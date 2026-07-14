"""
Tests for the Model Router — Escalation Strategy (agent/model_router.py).
"""

from __future__ import annotations

import pytest

from agent.model_router import (
    RouterConfig,
    EscalationDecision,
    ESCALATION,
    escalate,
    build_route_note,
    _analyze_complexity,
    reset_rr,
)


# =============================================================================
# Escalation levels
# =============================================================================


class TestEscalationLevels:
    def test_4_levels(self):
        assert len(ESCALATION) == 4

    def test_escalation_models(self):
        assert ESCALATION[0].model_id == "model-low-to-medium"
        assert ESCALATION[1].model_id == "model-only-plan"
        assert ESCALATION[2].model_id == "model-medium-hight"
        assert ESCALATION[3].model_id == "model-medium-hight"

    def test_levels_effort_ascending(self):
        efforts = [l.reasoning_effort for l in ESCALATION]
        assert efforts == ["medium", "max", "medium", "max"]

    def test_levels_cost(self):
        costs = [l.cost for l in ESCALATION]
        assert costs == [0.14, 0.14, 1.74, 1.74]

    def test_level_labels_match_ui(self):
        """Labels should match the UI display names."""
        assert ESCALATION[0].label == "DeepSeek V4 Flash Medium"
        assert ESCALATION[1].label == "DeepSeek V4 Flash Max"
        assert ESCALATION[2].label == "DeepSeek V4 Pro Medium"
        assert ESCALATION[3].label == "GLM 5.2"


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


class TestAnalyzeComplexity:
    def test_simple_question_is_low(self):
        assert _analyze_complexity("What is the capital?") == 0

    def test_short_code_question_is_low(self):
        assert _analyze_complexity("How do I write a function?") == 0

    def test_debug_query_is_medium(self):
        """Debug keywords bump complexity to at least 1."""
        assert _analyze_complexity("Debug this crash and find root cause") >= 1

    def test_refactor_with_list_is_medium(self):
        msg = "Refactor the auth module\n- Move login to its own file\n- Update imports\n- Add error handling"
        assert _analyze_complexity(msg) >= 1

    def test_long_complex_analysis_is_high(self):
        """Long multi-part request with architecture keyword."""
        msg = (
            "Analyze the entire project architecture:\n"
            "- Check all modules\n"
            "- Identify bottlenecks\n"
            "- Suggest improvements\n"
            "- Review the auth module\n"
            "- Check database layer\n\n"
            "```python\ndef foo():\n    pass\n```\n\n"
            "```javascript\nfunction bar() {}\n```"
        )
        assert _analyze_complexity(msg) >= 2


# =============================================================================
# Escalate() — main entry point
# =============================================================================


class TestEscalate:
    def test_simple_question_stays_at_0(self):
        """Simple question → stays at Level 0 (9router medium)."""
        d = escalate(
            agent_model="9router",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="What is the capital?",
            cfg=RouterConfig(),
            session_id="test-1",
        )
        # No complexity → stays at current level
        assert d.recommended_model == "9router"
        assert d.should_switch is False

    def test_debug_escalates_to_deepseek_pro(self):
        """Debug + root cause → escalates to Level 2 (model-medium-hight)."""
        d = escalate(
            agent_model="9router",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="Debug this crash and find root cause",
            cfg=RouterConfig(),
            session_id="test-2",
        )
        # Complexity triggers escalation to model-medium-hight
        assert d.recommended_model == "model-medium-hight"
        assert d.recommended_level == 2
        assert d.recommended_effort == "medium"
        assert d.should_switch is True
        assert "deep reasoning needed" in d.reason

    def test_debug_with_high_pressure_escalates_to_glm(self):
        """Debug + max tokens → escalates to Level 3 (model-medium-hight)."""
        d = escalate(
            agent_model="9router",
            agent_reasoning_effort="medium",
            estimated_tokens=900_000,
            estimated_files=200,
            user_message="Debug this complex security vulnerability",
            cfg=RouterConfig(context_pressure_threshold=0.50),
            session_id="test-4",
        )
        # High complexity + pressure → Level 3
        assert d.recommended_model == "model-medium-hight"
        assert d.recommended_level == 3
        assert d.recommended_effort == "max"
        assert d.should_switch is True

    def test_already_on_glm_stays(self):
        """Already on GLM with deep reasoning → no switch."""
        d = escalate(
            agent_model="model-medium-hight",
            agent_reasoning_effort="max",
            estimated_tokens=800_000,
            estimated_files=100,
            user_message="Debug this security issue",
            cfg=RouterConfig(context_pressure_threshold=0.50),
            session_id="test-5",
        )
        assert d.recommended_model == "model-medium-hight"
        assert d.recommended_level == 3
        assert d.should_switch is False

    def test_9router_escalation(self):
        """9router model escalates to deepseek-pro when complexity requires."""
        d = escalate(
            agent_model="9router",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="Debug this crash and find root cause",
            cfg=RouterConfig(),
            session_id="test-9router",
        )
        assert d.recommended_model == "model-medium-hight"
        assert d.recommended_level == 2
        assert d.should_switch is True

    def test_round_robin_escalation(self):
        """nvidia-round-robin model also escalates under complexity."""
        d = escalate(
            agent_model="nvidia-round-robin",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="Debug this crash and find root cause",
            cfg=RouterConfig(),
            session_id="test-rr",
        )
        assert d.recommended_model == "model-medium-hight"
        assert d.recommended_level == 2
        assert d.should_switch is True

    def test_model_low_to_medium(self):
        """Low complexity stays at Level 0 (9router)."""
        d = escalate(
            agent_model="9router",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="What is the capital?",
            cfg=RouterConfig(),
            session_id="test-low-medium",
        )
        assert d.recommended_model == "9router"
        assert d.should_switch is False

    def test_model_only_plan(self):
        """Planning tasks escalate to Level 1 (model-only-plan)."""
        d = escalate(
            agent_model="9router",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="Plan the project architecture",
            cfg=RouterConfig(),
            session_id="test-plan",
        )
        assert d.recommended_model == "model-only-plan"
        assert d.recommended_level == 1
        assert d.should_switch is True

    def test_model_medium_high(self):
        """Medium-high complexity tasks escalate to Level 2."""
        d = escalate(
            agent_model="9router",
            agent_reasoning_effort="medium",
            estimated_tokens=100,
            estimated_files=0,
            user_message="Debug this crash and find root cause",
            cfg=RouterConfig(),
            session_id="test-medium-high",
        )
        assert d.recommended_model == "model-medium-hight"
        assert d.recommended_level == 2
        assert d.should_switch is True


# =============================================================================
# Build route note
# =============================================================================


class TestBuildRouteNote:
    def test_empty_when_no_switch(self):
        d = EscalationDecision(
            current_level=0, recommended_level=0,
            recommended_model="9router",
            recommended_effort="medium", recommended_label="DeepSeek V4 Flash Medium",
            reason="", should_switch=False, auto_switch=False,
            is_upgrade=False, is_downgrade=False,
        )
        assert build_route_note(d) == ""

    def test_auto_switch_note_has_pricing(self):
        d = EscalationDecision(
            current_level=0, recommended_level=2,
            recommended_model="model-medium-hight",
            recommended_effort="medium", recommended_label="DeepSeek V4 Pro Medium",
            reason="complexity requires higher effort level",
            should_switch=True, auto_switch=True,
            is_upgrade=True, is_downgrade=False,
            cost_per_million=1.74,
        )
        note = build_route_note(d)
        assert "Auto-escalating" in note
        assert "$1.74" in note
        assert "Pro Medium" in note
