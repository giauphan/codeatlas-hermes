"""
Intelligent Model Router — Escalation Strategy.

Always starts with the lowest-cost model capable of solving the task,
then escalates step by step when the current level proves insufficient.

Escalation levels (cost-ascending, effort-ascending):

  Level 0: DeepSeek V4 Flash Medium  — $0.14/M, effort=low    (simple/low effort)
  Level 1: Mistral Codestral Latest  — $0.14/M, effort=medium (planning/architecture)
  Level 2: Mistral Large Latest      — $0.14/M, effort=high   (medium-high complexity)

**Implementation:** All tiers use 9router as the primary model, with 9router handling
internal round-robin logic across its model pool. The UI displays the original
model names for clarity, but all requests are routed through 9router.

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
    # Level 0: model-low-to-medium - Simple/low effort tasks
    EscalationLevel("DeepSeek V4 Flash Medium", "model-low-to-medium", "medium", 0.14, 0),

    # Level 1: model-only-plan - Planning/architecture tasks
    EscalationLevel("DeepSeek V4 Flash Max", "model-only-plan", "max", 0.14, 1),

    # Level 2: model-medium-high - Medium-high complexity tasks
    EscalationLevel("DeepSeek V4 Pro Medium", "model-medium-hight", "medium", 1.74, 2),

    # Level 3: model-high-pressure/very-long-complex
    EscalationLevel("GLM 5.2", "model-medium-hight", "max", 1.74, 3),
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

    # Debugging / root cause / troubleshooting — strong signal
    has_debug = any(kw in msg_lower for kw in ("debug", "root cause", "troubleshoot", "diagnose"))
    has_security = any(kw in msg_lower for kw in ("security", "investigate", "vulnerability"))
    if has_debug and "root cause" in msg_lower:
        score += 2  # Debug + root cause → Level 2 (DeepSeek V4 Pro)
    elif has_debug:
        score += 1  # Debug alone → Level 1 (9router)
    if has_security:
        score += 2  # Security bumps to Level 2 (DeepSeek V4 Pro)

    # Critical / crash / leak keywords
    for kw in ("critical", "crash", "leak", "severe", "urgent", "vulnerability"):
        if kw in msg_lower:
            score += 1
            break

    # Architecture / research / analysis — needs analytical effort
    for kw in ("architecture", "proposal", "research", "codebase analysis",
               "analyze project", "repository-wide", "system design",
               "technical proposal"):
        if kw in msg_lower:
            score += 1  # Architecture → Level 1 (9router)
            break

    # Planning — specific to 9router tier
    if "plan" in msg_lower or "planning" in msg_lower:
        score += 1  # Planning → Level 1 (9router)

    # Refactoring / restructuring — moderate complexity
    for kw in ("refactor", "restructure", "migrate code", "extract",
               "split file", "rewrite"):
        if kw in msg_lower:
            score += 1  # Refactoring → Level 1 (9router)
            break

    # Multiple tasks = higher complexity
    task_indicators = msg_lower.count(" - check") + msg_lower.count(" - add") + \
                      msg_lower.count(" - update") + msg_lower.count(" - fix") + \
                      msg_lower.count(" - create")
    if task_indicators >= 3:
        score += 1  # Multiple tasks → Level 1 (9router)


    # ── Map score to complexity level ──
    if score >= 8:
        return 3  # Very complex
    if score >= 3:
        return 2  # Complex (debugging, security)
    if score >= 1:
        return 1  # Moderate (planning, architecture)
    return 0  # Simple


# ── Escalate ────────────────────────────────────────────────────────────────


def _current_level(agent_model: str, agent_effort: str) -> int:
    """Map current agent state to an escalation level."""
    model_lower = (agent_model or "").lower()
    effort_lower = (agent_effort or "").lower()

    if "model-medium-hight" in model_lower:
        if effort_lower == "max":
            return 3
        return 2
    if "model-only-plan" in model_lower:
        return 1
    if "model-low-to-medium" in model_lower:
        return 0
    if "glm-5.2" in model_lower:
        return 3
    if "deepseek-v4-pro" in model_lower:
        return 2
    if "deepseek-v4-flash" in model_lower:
        if effort_lower == "max":
            return 1
        return 0
    if "9router" in model_lower or "round-robin" in model_lower or "round_robin" in model_lower:
        return 0
    return 0


def _target_level(
    current_level: int,
    estimated_tokens: int,
    estimated_files: int,
    user_message: str,
    cfg: RouterConfig,
) -> int:
    """Determine the appropriate escalation level based on actual analysis."""
    ctx_len = 1_000_000
    pressure = estimated_tokens / ctx_len if ctx_len > 0 else 0

    # 1. High context pressure -> Level 3 (GLM-5.2)
    if pressure >= cfg.context_pressure_threshold:
        return 3

    # 2. Large file count -> Level 2 (Pro Medium)
    if estimated_files >= cfg.large_file_count_threshold:
        return 2

    msg_lower = (user_message or "").lower()
    refactoring = any(kw in msg_lower for kw in (
        "refactor", "restructure", "rewrite", "reorganize",
        "migrate code", "extract", "split file"
    ))

    complexity = _analyze_complexity(user_message)

    if complexity >= 3:
        return 3
    if complexity >= 2:
        return 2
    if refactoring and (estimated_files >= 3 or estimated_tokens >= 30_000):
        return 1
    if complexity >= 1:
        return 1

    return 0


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
    if not cfg.enabled:
        current = _current_level(agent_model, agent_reasoning_effort)
        return EscalationDecision(
            current_level=current,
            recommended_level=current,
            recommended_model=agent_model,
            recommended_effort=agent_reasoning_effort,
            recommended_label=agent_model,
            reason="router disabled",
            should_switch=False,
            auto_switch=cfg.auto_switch,
            is_upgrade=False,
            is_downgrade=False,
        )

    current = _current_level(agent_model, agent_reasoning_effort)
    target = _target_level(current, estimated_tokens, estimated_files, user_message, cfg)

    # Escalation/downgrade decision
    chosen = target

    should_switch = chosen != current
    is_upgrade = chosen > current
    is_downgrade = chosen < current

    if should_switch:
        level = ESCALATION[chosen]
        rec_model = level.model_id
        rec_effort = level.reasoning_effort
        rec_label = level.label
        rec_cost = level.cost
    else:
        # Don't switch model name if we stay at the same level
        rec_model = agent_model
        rec_effort = agent_reasoning_effort
        rec_label = agent_model
        rec_cost = ESCALATION[current].cost

    # Build reason
    reasons = []
    if is_upgrade:
        complexity = _analyze_complexity(user_message)
        if complexity >= 2:
            reasons.append("deep reasoning needed")
        if estimated_tokens >= 500_000:
            reasons.append(f"context pressure ({estimated_tokens:,} tokens)")
        if estimated_files >= cfg.large_file_count_threshold:
            reasons.append(f"{estimated_files} files touched")
        if "plan" in user_message.lower() or "planning" in user_message.lower():
            reasons.append("planning task")
        if "architecture" in user_message.lower() or "design" in user_message.lower():
            reasons.append("architecture task")
        if not reasons:
            reasons.append("default level")
    elif is_downgrade:
        reasons.append("task completed, reducing cost")
    else:
        reasons.append("default level")

    return EscalationDecision(
        current_level=current,
        recommended_level=chosen,
        recommended_model=rec_model,
        recommended_effort=rec_effort,
        recommended_label=rec_label,
        reason=", ".join(reasons),
        should_switch=should_switch,
        auto_switch=cfg.auto_switch,
        is_upgrade=is_upgrade,
        is_downgrade=is_downgrade,
        cost_per_million=rec_cost,
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
