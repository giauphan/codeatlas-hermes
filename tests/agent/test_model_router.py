"""
Tests for the Intelligent Model Router (agent/model_router.py).
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List

from agent.model_router import (
    RouterConfig,
    RouteDecision,
    route,
    build_route_note,
    classify_routing_tier,
    _estimate_context_pressure,
    _detect_file_count_pressure,
    _get_model_price,
    _get_display_name,
    adjust_tier_with_knowledge,
    _model_to_tier,
    TIER_DEFAULT,
    TIER_LARGE_CONTEXT,
    TIER_DEEP_REASONING,
)


# =============================================================================
# RouterConfig
# =============================================================================


class TestRouterConfig:
    def test_defaults(self):
        cfg = RouterConfig.from_config({})
        assert cfg.enabled is False
        assert cfg.auto_switch is False
        assert cfg.context_pressure_threshold == 0.70
        assert cfg.large_file_count_threshold == 100

    def test_from_config(self):
        cfg = RouterConfig.from_config({
            "agent": {
                "model_router": {
                    "enabled": True,
                    "auto_switch": True,
                    "context_pressure_threshold": 0.85,
                    "large_file_count_threshold": 50,
                }
            }
        })
        assert cfg.enabled is True
        assert cfg.auto_switch is True
        assert cfg.context_pressure_threshold == 0.85
        assert cfg.large_file_count_threshold == 50

    def test_graceful(self):
        cfg = RouterConfig.from_config(None)  # type: ignore
        assert cfg.enabled is False


# =============================================================================
# Context pressure detection
# =============================================================================


class TestContextPressure:
    def test_no_tokens(self):
        assert _estimate_context_pressure(0, 1_000_000) == 0.0

    def test_half_full(self):
        assert _estimate_context_pressure(500_000, 1_000_000) == 0.5

    def test_full(self):
        assert _estimate_context_pressure(1_000_000, 1_000_000) == 1.0

    def test_over_capacity(self):
        assert _estimate_context_pressure(2_000_000, 1_000_000) == 1.0

    def test_zero_context(self):
        assert _estimate_context_pressure(500_000, 0) == 0.0


class TestFileCountPressure:
    def test_no_messages(self):
        assert _detect_file_count_pressure([]) == 0

    def test_text_messages_only(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert _detect_file_count_pressure(msgs) == 0

    def test_read_file_calls(self):
        msgs = [{"role": "assistant", "tool_calls": [
            {"function": {"name": "read_file", "arguments": '{"path": "foo.py"}'}},
            {"function": {"name": "read_file", "arguments": '{"path": "bar.py"}'}},
        ]}]
        assert _detect_file_count_pressure(msgs) == 2

    def test_duplicate_paths_deduplicated(self):
        msgs = [{"role": "assistant", "tool_calls": [
            {"function": {"name": "read_file", "arguments": '{"path": "foo.py"}'}},
            {"function": {"name": "read_file", "arguments": '{"path": "foo.py"}'}},
            {"function": {"name": "patch", "arguments": '{"path": "foo.py"}'}},
        ]}]
        assert _detect_file_count_pressure(msgs) == 1  # same file

    def test_patch_and_write_included(self):
        msgs = [{"role": "assistant", "tool_calls": [
            {"function": {"name": "write_file", "arguments": '{"path": "a.py"}'}},
            {"function": {"name": "patch", "arguments": '{"path": "b.py"}'}},
        ]}]
        assert _detect_file_count_pressure(msgs) == 2


# =============================================================================
# Tier classification
# =============================================================================


class TestTierClassification:
    def test_default_tier(self):
        cfg = RouterConfig()
        tier = classify_routing_tier(1000, 5, False, "deepseek-v4-flash", cfg)
        assert tier == TIER_DEFAULT

    def test_deep_reasoning_triggers(self):
        cfg = RouterConfig()
        tier = classify_routing_tier(1000, 5, True, "deepseek-v4-flash", cfg)
        assert tier == TIER_DEEP_REASONING

    def test_context_pressure_triggers_high(self):
        cfg = RouterConfig(context_pressure_threshold=0.50)
        tier = classify_routing_tier(600_000, 5, False, "deepseek-v4-flash", cfg)
        assert tier == TIER_LARGE_CONTEXT

    def test_context_pressure_below_threshold(self):
        cfg = RouterConfig(context_pressure_threshold=0.70)
        tier = classify_routing_tier(600_000, 5, False, "deepseek-v4-flash", cfg)
        assert tier == TIER_DEFAULT  # 0.6 < 0.7

    def test_many_files_triggers_large_context(self):
        cfg = RouterConfig(large_file_count_threshold=50)
        tier = classify_routing_tier(1000, 200, False, "deepseek-v4-flash", cfg)
        assert tier == TIER_LARGE_CONTEXT


# =============================================================================
# route() — main entry point
# =============================================================================


class TestRoute:
    def test_default_stays_default(self):
        cfg = RouterConfig(enabled=True)
        d = route("deepseek-v4-flash", 1000, 5, False, cfg)
        assert d.tier == TIER_DEFAULT
        assert d.should_switch is False

    def test_deep_reasoning_from_flash(self):
        cfg = RouterConfig(enabled=True)
        d = route("deepseek-v4-flash", 1000, 5, True, cfg)
        assert d.tier == TIER_DEEP_REASONING
        assert d.should_switch is True
        assert d.upgrade is True

    def test_deep_reasoning_from_pro(self):
        cfg = RouterConfig(enabled=True)
        d = route("deepseek-v4-pro", 1000, 5, True, cfg)
        assert d.tier == TIER_DEEP_REASONING
        assert d.recommended_model == "deepseek-v4-pro"
        assert d.recommended_mode == "reasoning"

    def test_auto_switch_mode(self):
        cfg = RouterConfig(enabled=True, auto_switch=True)
        d = route("deepseek-v4-flash", 1000, 5, True, cfg)
        assert d.auto_switch is True

    def test_build_note_no_switch(self):
        cfg = RouterConfig(enabled=True)
        d = route("deepseek-v4-flash", 1000, 5, False, cfg)
        assert build_route_note(d) == ""

    def test_build_note_with_switch(self):
        cfg = RouterConfig(enabled=True)
        d = route("deepseek-v4-flash", 1000, 5, True, cfg)
        note = build_route_note(d)
        assert "Model Routing" in note
        assert "clarify" in note
        assert "DeepSeek V4 Pro" in note

    def test_build_note_auto_switch(self):
        cfg = RouterConfig(enabled=True, auto_switch=True)
        d = route("deepseek-v4-flash", 1000, 5, True, cfg)
        note = build_route_note(d)
        assert "Auto-switching" in note
        assert "clarify" not in note  # auto-switch skips clarify


# =============================================================================
# Round-robin
# =============================================================================


class TestRoundRobin:
    def test_rotates_through_default_tier(self):
        """Session-based round-robin rotates through Tier 1 models."""
        cfg = RouterConfig(enabled=True)
        models_seen: set[str] = set()
        for _ in range(6):
            d = route("deepseek-v4-flash", 1000, 5, False, cfg, session_id="rr-test-1")
            models_seen.add(d.recommended_model)
        # Should see multiple models in Tier 1
        assert len(models_seen) > 1

    def test_rotates_through_large_context_tier(self):
        """Context pressure triggers Large Context tier with round-robin."""
        cfg = RouterConfig(context_pressure_threshold=0.30)
        models_seen: set[str] = set()
        for _ in range(10):
            d = route("deepseek-v4-flash", 400_000, 5, False, cfg, session_id="rr-test-2")
            models_seen.add(d.recommended_model)
        # DeepSeek V4 Flash (cheapest in tier 2) should be first,
        # but round-robin should cycle through all in the tier
        assert "deepseek-v4-flash" in models_seen
        assert len(models_seen) >= 2  # at least Flash + one other

    def test_different_sessions_independent(self):
        """Two sessions have independent round-robin state."""
        cfg = RouterConfig(enabled=True)
        # Session A — first call
        a1 = route("deepseek-v4-flash", 1000, 5, True, cfg, session_id="rr-a")
        # Session B — first call
        b1 = route("deepseek-v4-flash", 1000, 5, True, cfg, session_id="rr-b")
        # Both should start at index 0 (same model for Tier 3, only 1 candidate)
        assert a1.recommended_model == b1.recommended_model

    def test_tier_change_resets_index(self):
        """Changing tier resets the round-robin index."""
        cfg = RouterConfig(enabled=True)

        # Call deep reasoning (Tier 3) — advances index
        for _ in range(5):
            route("deepseek-v4-flash", 1000, 5, True, cfg, session_id="rr-reset")

        # Switch to default (Tier 1) — should reset to 0
        d = route("deepseek-v4-flash", 1000, 5, False, cfg, session_id="rr-reset")
        # Default tier, first model
        assert d.tier == TIER_DEFAULT


# =============================================================================
# Pricing & display helpers
# =============================================================================


class TestHelpers:
    def test_pricing_lookup(self):
        assert _get_model_price("deepseek-v4-flash") == 0.14
        assert _get_model_price("deepseek-v4-pro") == 1.74

    def test_display_name(self):
        assert "DeepSeek" in _get_display_name("deepseek-v4-flash")
        assert "MiMo" in _get_display_name("mimo-v2.5")

    def test_pricing_unknown(self):
        assert _get_model_price("nonexistent-model") is None

    def test_model_to_tier(self):
        assert _model_to_tier("deepseek-v4-flash") == TIER_DEFAULT
        assert _model_to_tier("deepseek-v4-pro") == TIER_LARGE_CONTEXT
        assert _model_to_tier("unknown") == TIER_DEFAULT


# =============================================================================
# Second Brain integration
# =============================================================================


class TestSecondBrainIntegration:
    def test_knowledge_downgrades_deep_reasoning(self):
        tier = adjust_tier_with_knowledge(TIER_DEEP_REASONING, True)
        assert tier == TIER_LARGE_CONTEXT

    def test_knowledge_downgrades_large_context(self):
        tier = adjust_tier_with_knowledge(TIER_LARGE_CONTEXT, True)
        assert tier == TIER_DEFAULT

    def test_knowledge_no_effect_on_default(self):
        tier = adjust_tier_with_knowledge(TIER_DEFAULT, True)
        assert tier == TIER_DEFAULT

    def test_no_knowledge_no_change(self):
        assert adjust_tier_with_knowledge(TIER_DEEP_REASONING, False) == TIER_DEEP_REASONING
        assert adjust_tier_with_knowledge(TIER_LARGE_CONTEXT, False) == TIER_LARGE_CONTEXT
