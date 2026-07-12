"""
Tests for the Model Router — Escalation Strategy (agent/model_router.py).
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List

from agent.model_router import (
    RouterConfig,
    EscalationDecision,
    ESCALATION,
    escalate,
    build_route_note,
    _requires_deep_reasoning,
    _detect_file_count_pressure,
    reset_rr,
)


# =============================================================================
# Escalation levels
# =============================================================================


class TestEscalationLevels:
    def test_4_levels(self):
        assert len(ESCALATION) == 4

    def test_levels_cost_ascending(self):
        costs = [l.cost for l in ESCALATION]
        assert costs == sorted(costs)

    def test_level_0_and_1_same_price(self):
        assert ESCALATION[0].cost == ESCALATION[1].cost  # both $0.14
        assert ESCALATION[0].model_id == ESCALATION[1].model_id  # same model

    def test_level_2_and_3_same_price(self):
        assert ESCALATION[2].cost == ESCALATION[3].cost  # both $1.74
        assert ESCALATION[2].model_id == ESCALATION[3].model_id  # same model


# =============================================================================
# RouterConfig
# =============================================================================


class TestRouterConfig:
    def test_defaults(self):
        cfg = RouterConfig()
        assert cfg.enabled is True
        assert cfg.auto_switch is True
        assert cfg.context_pressure_threshold == 0.70

    def test_from_config(self):
        cfg = RouterConfig.from_config({
            "agent": {"model_router": {"enabled": False, "auto_switch": False}}
        })
        assert cfg.enabled is False
        assert cfg.auto_switch is False


# =============================================================================
# Signal detection
# =============================================================================


class TestDeepReasoning:
    def test_detects_debug(self):
        assert _requires_deep_reasoning("debug this crash") is True

    def test_detects_security(self):
        assert _requires_deep_reasoning("security audit needed") is True

    def test_normal_question(self):
        assert _requires_deep_reasoning("What is the capital?") is False

    def test_refactor_not_deep(self):
        assert _requires_deep_reasoning("Refactor the auth module") is False


# =============================================================================
# Escalate() — main entry point
# =============================================================================


class TestEscalate:
    def test_simple_question_stays_at_0(self):
        """Simple question on Flash Medium → no escalation."""
        d = escalate(
            agent_model="deepseek-v4-flash",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="What is the capital?",
            cfg=RouterConfig(),
            session_id="test-1",
        )
        assert d.current_level == 0
        assert d.recommended_level == 0
        assert d.should_switch is False
        assert d.recommended_label == "DeepSeek V4 Flash Medium"

    def test_debug_escalates_to_pro_high(self):
        """Deep reasoning from Flash → escalates to Level 2 (Pro High)."""
        d = escalate(
            agent_model="deepseek-v4-flash",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="Debug this crash and find root cause",
            cfg=RouterConfig(),
            session_id="test-2",
        )
        assert d.recommended_level >= 2
        assert d.should_switch is True
        assert d.is_upgrade is True
        assert "DeepSeek V4 Pro" in d.recommended_label

    def test_context_pressure_escalates_to_high(self):
        """High tokens + files → Level 1 (Flash High)."""
        d = escalate(
            agent_model="deepseek-v4-flash",
            agent_reasoning_effort="medium",
            estimated_tokens=800_000,
            estimated_files=50,
            user_message="Continue working on this repo",
            cfg=RouterConfig(context_pressure_threshold=0.50),
            session_id="test-3",
        )
        assert d.recommended_level >= 1
        assert d.should_switch is True

    def test_debug_with_high_pressure_goes_max(self):
        """Debug + max tokens → Level 3 (Pro Max)."""
        d = escalate(
            agent_model="deepseek-v4-flash",
            agent_reasoning_effort="medium",
            estimated_tokens=900_000,
            estimated_files=200,
            user_message="Debug this complex security vulnerability",
            cfg=RouterConfig(context_pressure_threshold=0.50),
            session_id="test-4",
        )
        assert d.recommended_level >= 3
        assert d.recommended_effort == "max"

    def test_already_on_pro_max_stays(self):
        """Already on Pro Max with deep reasoning → no switch."""
        d = escalate(
            agent_model="deepseek-v4-pro",
            agent_reasoning_effort="max",
            estimated_tokens=800_000,
            estimated_files=100,
            user_message="Debug this security issue",
            cfg=RouterConfig(context_pressure_threshold=0.50),
            session_id="test-5",
        )
        assert d.recommended_level == 3
        assert d.should_switch is False

    def test_auto_switch_note(self):
        """Auto-switch builds the right note."""
        d = escalate(
            agent_model="deepseek-v4-flash",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="Debug this crash",
            cfg=RouterConfig(auto_switch=True),
            session_id="test-6",
        )
        note = build_route_note(d)
        assert "Model Escalation" in note
        assert "Auto-escalating" in note
        assert "DeepSeek V4 Pro" in note

    def test_clarify_note(self):
        """Non-auto-switch mode uses clarify."""
        d = escalate(
            agent_model="deepseek-v4-flash",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="Debug this crash",
            cfg=RouterConfig(auto_switch=False),
            session_id="test-7",
        )
        note = build_route_note(d)
        assert "clarify" in note

    def test_no_switch_note_empty(self):
        """No switch → empty note."""
        d = escalate(
            agent_model="deepseek-v4-flash",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="What is the capital?",
            cfg=RouterConfig(),
            session_id="test-8",
        )
        assert build_route_note(d) == ""

    def test_round_robin_within_same_price(self):
        """Round-robin alternates between Level 0 and 1 (same price)."""
        d1 = escalate(
            agent_model="deepseek-v4-flash",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="Hello",
            cfg=RouterConfig(),
            session_id="rr-test",
        )
        d2 = escalate(
            agent_model="deepseek-v4-flash",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="Hello again",
            cfg=RouterConfig(),
            session_id="rr-test",
        )
        labels = {d1.recommended_label, d2.recommended_label}
        assert len(labels) > 1  # rotated between different effort levels

    def test_reset_rr(self):
        """Reset round-robin state."""
        reset_rr("reset-test")
        # Should work without errors
        assert True


# =============================================================================
# Build route note
# =============================================================================


class TestBuildRouteNote:
    def test_empty_when_no_switch(self):
        d = EscalationDecision(
            current_level=0, recommended_level=0,
            recommended_model="deepseek-v4-flash",
            recommended_effort="medium", recommended_label="Flash Medium",
            reason="", should_switch=False, auto_switch=False,
            is_upgrade=False, is_downgrade=False,
        )
        assert build_route_note(d) == ""

    def test_auto_switch_note_has_pricing(self):
        d = EscalationDecision(
            current_level=0, recommended_level=2,
            recommended_model="deepseek-v4-pro",
            recommended_effort="high", recommended_label="DeepSeek V4 Pro High",
            reason="deep reasoning needed",
            should_switch=True, auto_switch=True,
            is_upgrade=True, is_downgrade=False,
            cost_per_million=1.74,
        )
        note = build_route_note(d)
        assert "Auto-escalating" in note
        assert "$1.74" in note
        assert "Pro High" in note
