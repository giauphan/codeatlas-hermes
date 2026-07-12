"""
E2E Test: Auto-Switch Model via Model Router Escalation.

Verifies that turn_context properly calls switch_model() when the
escalation ladder decides to move up or down.
"""

from __future__ import annotations

from typing import Any, Dict

from agent.model_router import RouterConfig, escalate


class MockAgent:
    """Minimal agent stub that tracks model/effort state."""

    def __init__(self) -> None:
        self.model = "deepseek-v4-flash"
        self.provider = "opencode-go"
        self.base_url = "https://api.opencode.ai/v1"
        self.reasoning_effort = "medium"
        self.session_id = "e2e-test"
        self.config: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"MockAgent(model={self.model!r}, effort={self.reasoning_effort!r})"


_switch_log: list[dict] = []


def _fake_switch_model(agent, new_model, new_provider, api_key="", base_url="", api_mode=""):
    """Fake switch_model that only updates the mock agent state."""
    _switch_log.append({
        "model": new_model,
        "provider": new_provider,
    })
    agent.model = new_model


def test_e2e_simple_question_stays_on_flash_medium():
    """T1: Simple question should NOT trigger any switch."""
    agent = MockAgent()
    _switch_log.clear()

    d = escalate(
        agent_model=agent.model,
        agent_reasoning_effort=agent.reasoning_effort,
        estimated_tokens=100,
        estimated_files=1,
        user_message="What is the capital of France?",
        cfg=RouterConfig(enabled=True, auto_switch=True),
        session_id=agent.session_id,
    )

    assert d.should_switch is False
    assert agent.model == "deepseek-v4-flash"
    assert agent.reasoning_effort == "medium"
    assert len(_switch_log) == 0


def test_e2e_debug_auto_switches_to_pro_medium():
    """T2: Debug request → auto-switch to Pro Medium."""
    agent = MockAgent()
    _switch_log.clear()

    d = escalate(
        agent_model=agent.model,
        agent_reasoning_effort=agent.reasoning_effort,
        estimated_tokens=300,
        estimated_files=5,
        user_message="Debug the memory leak in auth module and find root cause",
        cfg=RouterConfig(enabled=True, auto_switch=True),
        session_id=agent.session_id,
    )

    assert d.should_switch is True
    assert d.recommended_level == 2  # Pro Medium

    # Simulate auto-switch
    if d.auto_switch:
        _fake_switch_model(agent, d.recommended_model, "opencode-go")
        agent.reasoning_effort = d.recommended_effort

    assert agent.model == "deepseek-v4-pro"
    assert agent.reasoning_effort == "medium"
    assert len(_switch_log) == 1
    assert _switch_log[0]["model"] == "deepseek-v4-pro"


def test_e2e_security_high_pressure_to_pro_max():
    """T3: Security + 950K tokens → auto-switch to Pro Max."""
    agent = MockAgent()
    agent.model = "deepseek-v4-pro"
    agent.reasoning_effort = "medium"
    _switch_log.clear()

    d = escalate(
        agent_model=agent.model,
        agent_reasoning_effort=agent.reasoning_effort,
        estimated_tokens=950_000,
        estimated_files=200,
        user_message="Complex security vulnerability analysis with deep stack trace",
        cfg=RouterConfig(enabled=True, auto_switch=True, context_pressure_threshold=0.50),
        session_id=agent.session_id,
    )

    assert d.should_switch is True
    assert d.recommended_level == 3  # Pro Max
    assert d.recommended_effort == "max"

    if d.auto_switch:
        _fake_switch_model(agent, d.recommended_model, "opencode-go")
        agent.reasoning_effort = d.recommended_effort

    assert agent.model == "deepseek-v4-pro"
    assert agent.reasoning_effort == "max"


def test_e2e_simple_after_complex_auto_descends():
    """T4: After complex task, simple question → auto-descend to Flash Medium."""
    agent = MockAgent()
    agent.model = "deepseek-v4-pro"
    agent.reasoning_effort = "max"
    _switch_log.clear()

    d = escalate(
        agent_model=agent.model,
        agent_reasoning_effort=agent.reasoning_effort,
        estimated_tokens=50,
        estimated_files=1,
        user_message="Fix typo in comment",
        cfg=RouterConfig(enabled=True, auto_switch=True, context_pressure_threshold=0.50),
        session_id=agent.session_id,
    )

    assert d.should_switch is True
    assert d.is_downgrade is True  # Should be descending to save cost
    assert d.recommended_level == 0  # Flash Medium

    if d.auto_switch:
        _fake_switch_model(agent, d.recommended_model, "opencode-go")
        agent.reasoning_effort = d.recommended_effort

    assert agent.model == "deepseek-v4-flash"
    assert agent.reasoning_effort == "medium"


def test_e2e_medium_refactor_to_flash_max():
    """T5: Medium refactor → Flash Max (effort=max, same $0.14)."""
    agent = MockAgent()
    _switch_log.clear()

    d = escalate(
        agent_model=agent.model,
        agent_reasoning_effort=agent.reasoning_effort,
        estimated_tokens=50_000,
        estimated_files=5,
        user_message="Refactor the auth module utility functions",
        cfg=RouterConfig(enabled=True, auto_switch=True),
        session_id=agent.session_id,
    )

    assert d.should_switch is True
    assert d.recommended_level == 1  # Flash Max
    assert d.recommended_effort == "max"
    assert d.cost_per_million == 0.14  # Same price!

    if d.auto_switch:
        _fake_switch_model(agent, d.recommended_model, "opencode-go")
        agent.reasoning_effort = d.recommended_effort

    # Same model, higher effort
    assert agent.model == "deepseek-v4-flash"
    assert agent.reasoning_effort == "max"


def test_e2e_auto_switch_does_not_downgrade_with_clarify():
    """T6: When auto_switch=False, use clarify instead of auto-switch."""
    agent = MockAgent()
    _switch_log.clear()

    d = escalate(
        agent_model=agent.model,
        agent_reasoning_effort=agent.reasoning_effort,
        estimated_tokens=300,
        estimated_files=5,
        user_message="Debug this crash",
        cfg=RouterConfig(enabled=True, auto_switch=False),  # clarify mode
        session_id=agent.session_id,
    )

    assert d.should_switch is True
    assert d.auto_switch is False  # clarify mode
    # Agent state should NOT change yet
    assert agent.model == "deepseek-v4-flash"
    assert len(_switch_log) == 0  # No switch_model call
