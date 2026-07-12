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
    EscalationLevel("DeepSeek V4 Flash High",   "deepseek-v4-flash", "high",   0.14, 1),
    EscalationLevel("DeepSeek V4 Pro High",     "deepseek-v4-pro",   "high",   1.74, 2),
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


def _requires_deep_reasoning(user_message: str) -> bool:
    """Check if the request requires deep reasoning (debug, security, etc.)."""
    msg = (user_message or "").lower()
    for kw in ("debug", "diagnose", "troubleshoot", "root cause",
               "security", "investigate", "complex algorithm",
               "vulnerability", "exploit", "reverse engineer",
               "cryptography", "formal verification"):
        if kw in msg:
            return True
    return False


# ── Escalate ────────────────────────────────────────────────────────────────


def _current_level(agent_model: str, agent_effort: str) -> int:
    """Map current agent state to an escalation level."""
    key = (agent_model, agent_effort)
    return _LEVEL_BY_MODEL_EFFORT.get(key, 0)


def _target_level(
    current_level: int,
    estimated_tokens: int,
    estimated_files: int,
    deep_reasoning: bool,
    cfg: RouterConfig,
) -> int:
    """Determine the appropriate escalation level for this request.

    Rules:
      - Deep reasoning + high token pressure → Level 3 (Pro Max)
      - Deep reasoning → Level 2 (Pro High) minimum
      - High token pressure or many files → Level 1 (Flash High) minimum
      - Everything else → Level 0 (Flash Medium)
    """
    ctx_len = 1_000_000
    pressure = estimated_tokens / ctx_len if ctx_len > 0 else 0

    # Detect "still insufficient" from repeated tool calls + failures
    # For now, use token pressure as a proxy

    if deep_reasoning and pressure >= cfg.context_pressure_threshold:
        return max(current_level, 3)  # Pro Max
    if deep_reasoning:
        return max(current_level, 2)  # Pro High
    if pressure >= cfg.context_pressure_threshold or estimated_files >= cfg.large_file_count_threshold:
        return max(current_level, 1)  # Flash High
    if estimated_files >= 50:
        return max(current_level, 1)  # Moderate file count → Flash High

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
    # Detect current level
    current = _current_level(agent_model, agent_reasoning_effort)
    deep_reasoning = _requires_deep_reasoning(user_message)

    # Determine target level
    target = _target_level(current, estimated_tokens, estimated_files, deep_reasoning, cfg)

    # Round-robin within the same cost bracket (Level 0↔1 both $0.14, Level 2↔3 both $1.74)
    # Only rotate when staying at the same level — escalating always goes to target.
    # Auto-switch NEVER downgrades: if user chose a stronger model, respect it.
    if target > current:
        chosen = target
    elif target < current:
        # User chose a stronger model — respect it, don't auto-downgrade
        chosen = current
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
        if deep_reasoning and current < 3 and estimated_tokens >= 700_000:
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
        if deep_reasoning:
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
