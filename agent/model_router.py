"""
Intelligent Model Router — Escalation Strategy.

Always starts with the lowest-cost model capable of solving the task,
then escalates step by step when the current level proves insufficient.

Escalation levels (cost-ascending, effort-ascending):

  Level 0: DeepSeek V4 Flash Medium  — $0.14/M, effort=medium
  Level 1: DeepSeek V4 Flash High    — $0.14/M, effort=high   (same price!)
  Level 2: DeepSeek V4 Pro High      — $1.74/M, effort=high
  Level 3: DeepSeek V4 Pro Max       — $1.74/M, effort=max

Why Level 1 costs the same as Level 0: it's the SAME model
(deepseek-v4-flash) with higher reasoning effort — no extra token cost.

Config (``config.yaml`` → ``agent.model_router``)
-------------------------------------------------
.. code-block:: yaml

   agent:
     model_router:
       enabled: true
       auto_switch: true
       context_pressure_threshold: 0.70
       large_file_count_threshold: 100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Escalation levels (cost-ascending) ──────────────────────────────────────


@dataclass(frozen=True)
class EscalationLevel:
    """A step in the escalation ladder."""

    label: str          # User-facing name, e.g. "DeepSeek V4 Flash Medium"
    model_id: str       # Go API model ID, e.g. "deepseek-v4-flash"
    reasoning_effort: str  # "medium" | "high" | "max"
    cost: float         # $/M input tokens
    level: int          # 0-3


ESCALATION: list[EscalationLevel] = [
    EscalationLevel("DeepSeek V4 Flash Medium", "deepseek-v4-flash", "medium", 0.14, 0),
    EscalationLevel("DeepSeek V4 Flash Max",    "deepseek-v4-flash", "max",    0.14, 1),
    EscalationLevel("DeepSeek V4 Pro Medium",   "deepseek-v4-pro",   "medium", 1.74, 2),
    EscalationLevel("DeepSeek V4 Pro Max",      "deepseek-v4-pro",   "max",    1.74, 3),
]

_LEVEL_BY_MODEL_EFFORT: dict[tuple[str, str], int] = {
    (l.model_id, l.reasoning_effort): l.level for l in ESCALATION
}

_ROUND_ROBIN: dict[str, int] = {}  # session_id → next level index


def _next_rr(session_id: str, candidates: list[int]) -> int:
    """Round-robin pick from candidate levels for a session."""
    key = f"rr:{session_id}"

    if key not in _ROUND_ROBIN:
        _ROUND_ROBIN[key] = 0

    idx = _ROUND_ROBIN[key]
    chosen = candidates[idx % len(candidates)]
    _ROUND_ROBIN[key] = idx + 1

    return chosen


def reset_rr(session_id: str) -> None:
    """Reset round-robin state for a session."""
    _ROUND_ROBIN.pop(f"rr:{session_id}", None)


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class RouterConfig:
    """Runtime config for the model router."""

    enabled: bool = True
    auto_switch: bool = True
    context_pressure_threshold: float = 0.70
    large_file_count_threshold: int = 100

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RouterConfig":
        if not isinstance(config, dict):
            return cls()
        agent_cfg = config.get("agent", {})
        if not isinstance(agent_cfg, dict):
            agent_cfg = {}
        mr_cfg = agent_cfg.get("model_router", {})
        if not isinstance(mr_cfg, dict):
            mr_cfg = {}
        return cls(
            enabled=bool(mr_cfg.get("enabled", True)),
            auto_switch=bool(mr_cfg.get("auto_switch", True)),
            context_pressure_threshold=float(mr_cfg.get("context_pressure_threshold", 0.70)),
            large_file_count_threshold=int(mr_cfg.get("large_file_count_threshold", 100)),
        )


# ── Decision ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EscalationDecision:
    """The router's decision for this turn."""

    current_level: int         # 0-3 where we are now
    recommended_level: int     # 0-3 where we should be
    recommended_model: str     # model ID to switch to
    recommended_effort: str    # "medium" | "high" | "max"
    recommended_label: str     # user-facing label
    reason: str
    should_switch: bool
    auto_switch: bool
    is_upgrade: bool
    is_downgrade: bool
    cost_per_million: float = 0.0


# ── Signal detection ────────────────────────────────────────────────────────


def _detect_file_count_pressure(messages: List[Dict[str, Any]]) -> int:
    """Estimate how many files are being touched in this turn."""
    count = 0
    seen_paths: set[str] = set()
    file_tools = {"read_file", "write_file", "patch"}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            name = tc.get("function", {}).get("name", "")
            if name in file_tools:
                args_str = tc.get("function", {}).get("arguments", "")
                try:
                    import json
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    path = args.get("path", "")
                    if path and path not in seen_paths:
                        seen_paths.add(path)
                        count += 1
                except Exception:
                    pass
    return count


def _analyze_complexity(user_message: str) -> int:
    """Analyze user message and return complexity level 0-3.

    Reads the actual prompt content and scores it on:
      - Message length (long → complex)
      - Code blocks, diffs, stack traces
      - Multi-part requests (lists, numbered items)
      - Technical depth (file paths, error messages)
      - Semantic depth (debug, security, architecture keywords)

    Returns 0 (trivial) up to 3 (very complex).
    """
    msg = (user_message or "").strip()
    if not msg:
        return 0

    score = 0
    lines = msg.split("\n")
    words = msg.split()
    msg_lower = msg.lower()

    # ── Length signals ──────────────────────────────────────────
    word_count = len(words)
    line_count = len(lines)

    if word_count >= 500:
        score += 3
    elif word_count >= 200:
        score += 2
    elif word_count >= 50:
        score += 1

    if line_count >= 100:
        score += 2
    elif line_count >= 30:
        score += 1
    elif line_count >= 10:
        score += 1

    # ── Code blocks & technical structure ───────────────────────
    code_block_count = msg.count("```")
    if code_block_count >= 6:
        score += 2
    elif code_block_count >= 2:
        score += 1

    if "diff --git" in msg or "--- a/" in msg:
        score += 1

    # ── Multi-part / structured requests ────────────────────────
    if "\n-" in msg or "\n*" in msg:
        score += 1
    if "\n1." in msg or "\n2." in msg or "\n3." in msg:
        score += 1

    # ── Technical depth ─────────────────────────────────────────
    file_refs = sum(1 for w in words if "/" in w and "." in w)
    if file_refs >= 10:
        score += 2
    elif file_refs >= 3:
        score += 1

    if msg.count("?") >= 5:
        score += 1

    if "Traceback" in msg or "Error:" in msg or "Exception" in msg:
        score += 2

    # ── Semantic depth ──────────────────────────────────────────
    # Debugging / root cause / troubleshooting — strong signal
    has_debug = any(kw in msg_lower for kw in ("debug", "root cause", "troubleshoot", "diagnose"))
    has_security = any(kw in msg_lower for kw in ("security", "investigate", "vulnerability"))
    if has_debug:
        score += 3  # Debugging alone bumps to Level 2
    if has_security:
        score += 3  # Security alone bumps to Level 2
    if has_debug and "root cause" in msg_lower:
        score += 1  # Combo: debug + root cause → Pro Max territory

    # Architecture / research / analysis — needs analytical effort
    for kw in ("architecture", "proposal", "research", "codebase analysis",
               "analyze project", "repository-wide", "system design",
               "technical proposal"):
        if kw in msg_lower:
            score += 2
            break

    # Refactoring / restructuring — moderate complexity
    for kw in ("refactor", "restructure", "migrate code", "extract",
               "split file", "rewrite"):
        if kw in msg_lower:
            score += 1
            break

    # Multiple tasks = higher complexity
    task_indicators = msg_lower.count(" - check") + msg_lower.count(" - add") + \
                      msg_lower.count(" - update") + msg_lower.count(" - fix") + \
                      msg_lower.count(" - create")
    if task_indicators >= 3:
        score += 1

    # ── Map score to level ──────────────────────────────────────
    if score >= 8:
        return 3  # Very complex — Pro Max
    if score >= 3:
        return 2  # Complex — Pro Medium
    if score >= 2:
        return 1  # Moderate — Flash Max
    return 0  # Simple — Flash Medium


# ── Escalate ────────────────────────────────────────────────────────────────


def _current_level(agent_model: str, agent_effort: str) -> int:
    """Map current agent state to an escalation level."""
    key = (agent_model, agent_effort)
    return _LEVEL_BY_MODEL_EFFORT.get(key, 0)


def _target_level(
    current_level: int,
    estimated_tokens: int,
    estimated_files: int,
    complexity: int,
    cfg: RouterConfig,
) -> int:
    """Determine the appropriate escalation level based on actual analysis.

    Uses BOTH content analysis (complexity 0-3) and runtime metrics
    (tokens, files) to decide.
    """
    ctx_len = 1_000_000
    pressure = estimated_tokens / ctx_len if ctx_len > 0 else 0

    # ── Highest priority: runtime metrics ──
    if pressure >= cfg.context_pressure_threshold:
        return max(current_level, 3)  # Pro Max

    if estimated_files >= cfg.large_file_count_threshold:
        return max(current_level, 2)  # Pro Medium

    # ── Content analysis ──
    if complexity >= 3:
        return max(current_level, 3)  # Pro Max
    if complexity >= 2:
        return max(current_level, 2)  # Pro Medium
    if complexity >= 1:
        return max(current_level, 1)  # Flash Max

    # ── Secondary metrics ──
    if estimated_files >= 50:
        return max(current_level, 1)
    if estimated_tokens >= 100_000:
        return max(current_level, 1)

    return 0  # Flash Medium


# ── Main entry point ────────────────────────────────────────────────────────


def escalate(
    agent_model: str,
    agent_reasoning_effort: str,
    estimated_tokens: int,
    estimated_files: int,
    user_message: str,
    cfg: RouterConfig,
    session_id: str = "",
) -> EscalationDecision:
    """Determine the optimal escalation level for this turn.

    Args:
        agent_model: Current model ID (e.g. "deepseek-v4-flash").
        agent_reasoning_effort: Current effort level (e.g. "medium").
        estimated_tokens: Estimated tokens in the current request.
        estimated_files: Files touched in recent tool calls.
        user_message: The user's message for keyword analysis.
        cfg: Router config.
        session_id: Session ID for round-robin tracking.

    Returns:
        EscalationDecision with the recommended level and model.
    """
    # Bypassing escalation for custom 9Router / round-robin combo models
    model_lower = (agent_model or "").lower()
    if (
        "9router" in model_lower
        or "round-robin" in model_lower
        or "round_robin" in model_lower
    ):
        return EscalationDecision(
            current_level=0,
            recommended_level=0,
            recommended_model=agent_model,
            recommended_effort=agent_reasoning_effort,
            recommended_label=agent_model,
            reason="custom combo / round-robin model active, skipping escalation",
            should_switch=False,
            auto_switch=cfg.auto_switch,
            is_upgrade=False,
            is_downgrade=False,
            cost_per_million=0.0,
        )

    # Detect current level
    current = _current_level(agent_model, agent_reasoning_effort)
    complexity = _analyze_complexity(user_message)

    # Determine target level (complexity + runtime metrics)
    target = _target_level(current, estimated_tokens, estimated_files, complexity, cfg)

    # Round-robin within the same cost bracket (Level 0↔1 both $0.14, Level 2↔3 both $1.74)
    # Rotate when staying at same level; escalate UP when task demands;
    # descend DOWN when task becomes simple again (cost optimization).
    if target > current:
        chosen = target
    elif target < current:
        # Task became simpler — descend DOWN to save cost
        chosen = target
    elif target == 0:
        # Alternate between Flash Medium and Flash High (same price)
        candidates = [0, 1]
        if session_id:
            chosen = _next_rr(f"{session_id}/l0", candidates)
        else:
            chosen = candidates[0]
    elif target == 1:
        # Even when target is 1, still consider Flash Medium if it was working
        if current <= 1:
            candidates = [0, 1] if session_id else [1]
            chosen = _next_rr(f"{session_id}/l0", candidates) if session_id else 1
        else:
            chosen = target
    elif target == 2 or target == 3:
        # Pro tier — alternate between High and Max if needed
        if complexity >= 2 and current < 3 and estimated_tokens >= 700_000:
            chosen = 3  # Force Pro Max on high pressure + reasoning
        elif current == 2 and target == 2:
            candidates = [2, 3] if session_id else [2]
            chosen = _next_rr(f"{session_id}/l1", candidates) if session_id else 2
        elif current < 2 and target >= 2:
            # Escalating up: start at Pro High
            chosen = 2
        else:
            chosen = target
    else:
        chosen = target

    level = ESCALATION[chosen]
    should_switch = chosen != current
    is_upgrade = chosen > current
    is_downgrade = chosen < current

    # Build reason
    reasons = []
    if is_upgrade:
        if complexity >= 2:
            reasons.append("deep reasoning needed")
        if estimated_tokens >= 500_000:
            reasons.append(f"context pressure ({estimated_tokens:,} tokens)")
        if estimated_files >= cfg.large_file_count_threshold:
            reasons.append(f"{estimated_files} files touched")
    if is_downgrade:
        reasons.append("task completed, reducing cost")
    if not reasons:
        reasons.append("default level")

    return EscalationDecision(
        current_level=current,
        recommended_level=chosen,
        recommended_model=level.model_id,
        recommended_effort=level.reasoning_effort,
        recommended_label=level.label,
        reason=", ".join(reasons),
        should_switch=should_switch,
        auto_switch=cfg.auto_switch,
        is_upgrade=is_upgrade,
        is_downgrade=is_downgrade,
        cost_per_million=level.cost,
    )


def build_route_note(decision: EscalationDecision) -> str:
    """Build a system-prompt note for the model switch."""
    if not decision.should_switch:
        return ""

    pricing = f"${decision.cost_per_million:.2f}/M" if decision.cost_per_million > 0 else ""

    if decision.auto_switch:
        return (
            f"\n\n## Model Escalation\n"
            f"Auto-escalating to **{decision.recommended_label}** "
            f"(effort={decision.recommended_effort}"
            + (f", {pricing}" if pricing else "")
            + f") — {decision.reason}.\n"
        )

    return (
        f"\n\n## Model Escalation\n"
        f"**Recommendation:** {decision.recommended_label} "
        f"(effort={decision.recommended_effort}"
        + (f", {pricing}" if pricing else "")
        + f")\n"
        f"**Reason:** {decision.reason}\n\n"
        f"Call the `clarify` tool to ask the user:\n"
        f'- question: "I recommend **{decision.recommended_label}** '
        f"({decision.reason}). Switch before continuing?\"\n"
        f"- choices:\n"
        f'  - "Switch to {decision.recommended_label}"\n'
        f'  - "Continue with current"\n'
    )


def _cost_for_level(level: int) -> float:
    if 0 <= level < len(ESCALATION):
        return ESCALATION[level].cost
    return 0.0
