"""
Intelligent Model Router — automatic 3-tier model selection.

Replaces manual model selection by analysing the request and
automatically picking the most appropriate model from OpenCode Go.

Design
------
Three routing tiers, each with preferred models ordered by cost:

Tier 1 — Default (cheapest: $0.14/M)
    Chat, small code edits, syntax fixes, documentation, minor refactoring.
    Models: ``deepseek-v4-flash``, ``mimo-v2.5``

Tier 2 — Large Context (1M context, Flash preferred)
    Large repositories, multi-file changes, architecture analysis,
    repository-wide understanding, feature extension.
    Tries ``deepseek-v4-flash`` first (1M context, $0.14/M).
    Falls back to ``deepseek-v4-pro`` / ``glm-5.2`` / ``mimo-v2.5-pro``.

Tier 3 — Deep Reasoning ($1.74/M, max reasoning)
    Difficult debugging, root cause analysis, security investigation,
    complex architecture design, multi-step reasoning.
    Models: ``deepseek-v4-pro`` (reasoning mode)

The router also detects **context pressure** — when the conversation
is approaching the model's context window, it automatically suggests
stepping up to a larger-context model.

Config (``config.yaml`` → ``agent.model_router``)
-------------------------------------------------
.. code-block:: yaml

   agent:
     model_router:
       enabled: false          # opt-in (replaces model_recommendation)
       auto_switch: false      # true = skip clarify, just swap
       context_pressure_threshold: 0.70  # % of context window before warning
       large_file_count_threshold: 100   # files touched → Tier 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Tiers ───────────────────────────────────────────────────────────────────

TIER_DEFAULT = "default"
TIER_LARGE_CONTEXT = "large_context"
TIER_DEEP_REASONING = "deep_reasoning"

_TIER_DISPLAY: dict[str, str] = {
    TIER_DEFAULT: "Default (lightweight)",
    TIER_LARGE_CONTEXT: "Large Context",
    TIER_DEEP_REASONING: "Deep Reasoning",
}

# ── Tier → model mapping (ordered by cost ascending) ────────────────────────
# Each entry: (model_id, reasoning_mode)

_TIER_MODELS: dict[str, list[tuple[str, bool]]] = {
    TIER_DEFAULT: [
        ("deepseek-v4-flash", False),
        ("mimo-v2.5", False),
    ],
    TIER_LARGE_CONTEXT: [
        ("deepseek-v4-flash", False),  # 1M context, try cheap first
        ("glm-5.2", False),
        ("deepseek-v4-pro", False),
        ("mimo-v2.5-pro", False),
    ],
    TIER_DEEP_REASONING: [
        ("deepseek-v4-pro", True),  # reasoning mode
    ],
}

# ── Pricing lookup (from OpenCode Go docs) ──────────────────────────────────

_MODEL_PRICING: dict[str, float] = {
    # Lightweight
    "deepseek-v4-flash": 0.14,
    "mimo-v2.5": 0.14,
    "mimo-v2-omni": 0.25,
    "mimo-v2-pro": 1.74,
    # Balanced
    "minimax-m3": 0.30,
    "minimax-m2.7": 0.30,
    "minimax-m2.5": 0.30,
    "qwen3.7-plus": 0.40,
    "qwen3.6-plus": 0.50,
    "qwen3.5-plus": 0.50,
    "kimi-k2.7-code": 0.95,
    "kimi-k2.6": 0.95,
    "kimi-k2.5": 0.95,
    # Heavy
    "glm-5.2": 1.40,
    "glm-5.1": 1.40,
    "glm-5": 1.40,
    "deepseek-v4-pro": 1.74,
    "mimo-v2.5-pro": 1.74,
    "qwen3.7-max": 2.50,
    "hy3-preview": 1.40,
}

# ── Data structures ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RouteDecision:
    """The router's decision for this turn."""

    tier: str  # TIER_DEFAULT | TIER_LARGE_CONTEXT | TIER_DEEP_REASONING
    recommended_model: str  # e.g. "deepseek-v4-flash"
    recommended_mode: str  # "chat" | "reasoning"
    reason: str
    should_switch: bool = False
    auto_switch: bool = False
    upgrade: bool = False
    downgrade: bool = False


@dataclass
class RouterConfig:
    """Runtime config for the model router."""

    enabled: bool = False
    auto_switch: bool = False
    context_pressure_threshold: float = 0.70  # % of context window
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
            enabled=bool(mr_cfg.get("enabled", False)),
            auto_switch=bool(mr_cfg.get("auto_switch", False)),
            context_pressure_threshold=float(mr_cfg.get("context_pressure_threshold", 0.70)),
            large_file_count_threshold=int(mr_cfg.get("large_file_count_threshold", 100)),
        )


# ── Context pressure detection ───────────────────────────────────────────────


def _estimate_context_pressure(
    estimated_tokens: int,
    context_length: int,
) -> float:
    """Return 0.0 (no pressure) to 1.0 (critical).

    Uses the model's context length and current estimated token count.
    """
    if context_length <= 0:
        return 0.0
    ratio = estimated_tokens / context_length
    return min(ratio, 1.0)


def _detect_file_count_pressure(messages: List[Dict[str, Any]]) -> int:
    """Estimate how many files are being touched in this turn.

    Counts tool calls that reference file paths (read_file, write_file,
    patch, search_files, terminal) in recent messages.
    """
    count = 0
    seen_paths: set[str] = set()
    file_tools = {"read_file", "write_file", "patch"}

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
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


def _get_context_length(model: str) -> int:
    """Get known context length for a model (1M for all Go models)."""
    # All Go models currently support 1M context
    return 1_000_000


# ── Tier classification ─────────────────────────────────────────────────────


def classify_routing_tier(
    estimated_tokens: int,
    estimated_files: int,
    is_deep_reasoning_task: bool,
    current_model: str,
    cfg: RouterConfig,
) -> str:
    """Determine which routing tier this request needs.

    Priority:
    1. Deep reasoning keywords → Tier 3
    2. High context pressure OR many files → Tier 2
    3. Everything else → Tier 1
    """
    # Check deep reasoning signals first (highest priority)
    if is_deep_reasoning_task:
        return TIER_DEEP_REASONING

    # Check context pressure
    ctx_len = _get_context_length(current_model)
    pressure = _estimate_context_pressure(estimated_tokens, ctx_len)

    if pressure >= cfg.context_pressure_threshold:
        return TIER_LARGE_CONTEXT

    if estimated_files >= cfg.large_file_count_threshold:
        return TIER_LARGE_CONTEXT

    return TIER_DEFAULT


# ── Round-robin state per session ────────────────────────────────────────────


class RoundRobinTracker:
    """Rotates through models in each tier to distribute load.

    Maintains a per-tier index that advances every time a tier is
    selected.  The index resets when the tier changes.
    """

    def __init__(self):
        self._indices: dict[str, int] = {}
        self._last_tier: str | None = None

    def next(self, tier: str, candidates: list[tuple[str, bool]]) -> tuple[str, bool]:
        """Get the next model in round-robin order for this tier.

        Resets to 0 when the tier changes from the previous call.
        """
        # Reset index on tier change
        if self._last_tier is not None and self._last_tier != tier:
            self._indices.pop(tier, None)
        self._last_tier = tier

        if not candidates:
            return ("", False)

        idx = self._indices.get(tier, 0)
        if idx >= len(candidates):
            idx = 0

        chosen = candidates[idx]

        # Advance for next turn
        self._indices[tier] = (idx + 1) % len(candidates)

        return chosen


# Global round-robin state (one per session; turn_context creates per-session)
SESSION_RR: dict[str, RoundRobinTracker] = {}


def get_rr_tracker(session_id: str) -> RoundRobinTracker:
    """Get or create a round-robin tracker for a session."""
    if session_id not in SESSION_RR:
        SESSION_RR[session_id] = RoundRobinTracker()
    return SESSION_RR[session_id]


def reset_rr(session_id: str) -> None:
    """Reset round-robin state for a session."""
    SESSION_RR.pop(session_id, None)


# ── Main routing ─────────────────────────────────────────────────────────────


def route(
    current_model: str,
    estimated_tokens: int,
    estimated_files: int,
    is_deep_reasoning_task: bool,
    cfg: RouterConfig,
    session_id: str = "",
) -> RouteDecision:
    """Main entry point: determine the best model for this request.

    Uses round-robin selection when multiple models in a tier share
    the same price tier, spreading requests across them.

    Returns a RouteDecision indicating which model to use and whether
    to auto-switch.
    """
    tier = classify_routing_tier(
        estimated_tokens, estimated_files,
        is_deep_reasoning_task, current_model, cfg,
    )

    # Get current model info
    current_tier = _model_to_tier(current_model)

    # Determine if we need to switch
    should_switch = current_tier != tier

    # Pick the best model for this tier (round-robin)
    candidates = _TIER_MODELS.get(tier, [])
    if not candidates:
        return RouteDecision(
            tier=tier,
            recommended_model=current_model,
            recommended_mode="chat",
            reason="No models available for this tier",
        )

    # Round-robin selection
    if session_id:
        rr = get_rr_tracker(session_id)
        best_model, reasoning_mode = rr.next(tier, candidates)
    else:
        # No session — still rotate per call (stateless fallback)
        best_model, reasoning_mode = candidates[0]

    # Build reason
    reasons = []
    if is_deep_reasoning_task:
        reasons.append("deep reasoning required")
    if estimated_tokens >= 500_000:
        reasons.append(f"context pressure ({estimated_tokens:,} tokens)")
    if estimated_files >= cfg.large_file_count_threshold:
        reasons.append(f"{estimated_files} files touched")

    reason = ", ".join(reasons) if reasons else "default tier"

    mode_label = "reasoning" if reasoning_mode else "chat"
    display = _TIER_DISPLAY.get(tier, tier)

    return RouteDecision(
        tier=tier,
        recommended_model=best_model,
        recommended_mode=mode_label,
        reason=f"{display}: {reason}",
        should_switch=should_switch,
        auto_switch=cfg.auto_switch,
        upgrade=tier != TIER_DEFAULT and current_tier == TIER_DEFAULT,
        downgrade=tier == TIER_DEFAULT and current_tier != TIER_DEFAULT,
    )


def _model_to_tier(model: str) -> str:
    """Map a model ID to its routing tier."""
    m = (model or "").lower()
    for tier, models in _TIER_MODELS.items():
        for mid, _ in models:
            if mid in m:
                return tier
    return TIER_DEFAULT


def _get_model_price(model: str) -> Optional[float]:
    """Get pricing for a model ID."""
    m = (model or "").lower()
    for model_id, price in _MODEL_PRICING.items():
        if model_id in m:
            return price
    return None


def _get_display_name(model: str) -> str:
    """Get user-facing display name for a model ID."""
    names = {
        "deepseek-v4-flash": "DeepSeek V4 Flash",
        "deepseek-v4-pro": "DeepSeek V4 Pro",
        "mimo-v2.5": "MiMo V2.5",
        "mimo-v2.5-pro": "MiMo V2.5 Pro",
        "mimo-v2-pro": "MiMo V2 Pro",
        "mimo-v2-omni": "MiMo V2 Omni",
        "minimax-m3": "MiniMax M3",
        "minimax-m2.7": "MiniMax M2.7",
        "minimax-m2.5": "MiniMax M2.5",
        "qwen3.7-max": "Qwen 3.7 Max",
        "qwen3.7-plus": "Qwen 3.7 Plus",
        "qwen3.6-plus": "Qwen 3.6 Plus",
        "qwen3.5-plus": "Qwen 3.5 Plus",
        "glm-5.2": "GLM 5.2",
        "glm-5.1": "GLM 5.1",
        "glm-5": "GLM 5",
        "kimi-k2.7-code": "Kimi K2.7 Code",
        "kimi-k2.6": "Kimi K2.6",
        "kimi-k2.5": "Kimi K2.5",
        "hy3-preview": "Hy3 Preview",
    }
    m = (model or "").lower()
    for model_id, name in names.items():
        if model_id in m:
            return name
    return model


def build_route_note(decision: RouteDecision) -> str:
    """Build a system-prompt note that instructs the LLM to call ``clarify``
    or directly informs about the model switch.
    """
    if not decision.should_switch:
        return ""

    display = _get_display_name(decision.recommended_model)
    price = _get_model_price(decision.recommended_model)

    pricing = f"${price:.2f}/M" if price else ""

    if decision.auto_switch:
        return (
            f"\n\n## Model Routing\n"
            f"Auto-switching to **{display}** ({decision.recommended_mode} mode"
            + (f", {pricing})" if pricing else ")")
            + f" — {decision.reason}.\n"
        )

    return (
        f"\n\n## Model Routing\n"
        f"**Recommendation:** {display} ({decision.recommended_mode}"
        + (f", {pricing})" if pricing else ")")
        + f"\n**Reason:** {decision.reason}\n\n"
        f"Call the `clarify` tool to ask the user:\n"
        f'- question: "Based on your request, I recommend **{display}** '
        f"({decision.reason}). Would you like me to switch before continuing?\"\n"
        f"- choices:\n"
        f'  - "Switch to {display}"\n'
        f'  - "Continue with current model"\n'
    )


# ── Second Brain integration hook ────────────────────────────────────────────

def adjust_tier_with_knowledge(
    tier: str,
    knowledge_available: bool,
) -> str:
    """Downgrade tier if CodeAtlas Second Brain has relevant knowledge.

    When the Second Brain already contains project knowledge (Dreams,
    Skills, DNA, Immune Genes, Memories), a lower-cost model may be
    sufficient because the reasoning effort is reduced.

    Args:
        tier: Current routing tier.
        knowledge_available: True if Second Brain returned relevant data.

    Returns:
        Adjusted tier (may be downgraded if knowledge is rich).
    """
    if not knowledge_available:
        return tier

    # With rich knowledge, deep reasoning → large context
    if tier == TIER_DEEP_REASONING:
        logger.info("Second Brain knowledge available → downgrading from deep reasoning to large context")
        return TIER_LARGE_CONTEXT

    # With rich knowledge, large context → default
    if tier == TIER_LARGE_CONTEXT:
        logger.info("Second Brain knowledge available → downgrading from large context to default")
        return TIER_DEFAULT

    return tier
