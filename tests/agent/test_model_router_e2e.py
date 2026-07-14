"""End-to-end tests for the model router."""
import unittest
from unittest.mock import MagicMock, patch

from agent.model_router import RouterConfig, escalate

class TestModelRouterE2E(unittest.TestCase):
    def setUp(self):
        self.cfg = RouterConfig(enabled=True, auto_switch=True)

    def test_simple_task_stays_on_9router(self):
        """Simple questions using 9router — no escalation."""
        decision = escalate(
            agent_model="9router",
            agent_reasoning_effort="medium",
            estimated_tokens=5000,
            estimated_files=1,
            user_message="what is python?",
            cfg=self.cfg,
            session_id="e2e-test-1",
        )
        self.assertFalse(decision.should_switch)
        self.assertEqual(decision.recommended_model, "9router")
        self.assertEqual(decision.recommended_level, 0)

    def test_many_files_escalates_to_deepseek_pro(self):
        """150+ files → model-medium-hight (level 2, trigger: large_file_count_threshold)."""
        decision = escalate(
            agent_model="model-only-plan",
            agent_reasoning_effort="max",
            estimated_tokens=50000,
            estimated_files=150,  # > threshold 100
            user_message="update project files",
            cfg=self.cfg,
            session_id="e2e-test-3",
        )
        self.assertTrue(decision.should_switch)
        self.assertEqual(decision.recommended_model, "model-medium-hight")
        self.assertEqual(decision.recommended_level, 2)

    def test_complex_prompt_escalates_to_deepseek_pro(self):
        """Debug + root cause → complexity 2 → model-medium-hight (level 2)."""
        decision = escalate(
            agent_model="model-low-to-medium",
            agent_reasoning_effort="medium",
            estimated_tokens=50000,
            estimated_files=5,
            user_message="Debug this critical issue: find the root cause of the memory leak",
            cfg=self.cfg,
            session_id="e2e-test-4",
        )
        self.assertTrue(decision.should_switch)
        self.assertEqual(decision.recommended_model, "model-medium-hight")
        self.assertEqual(decision.recommended_level, 2)

    def test_very_long_complex_escalates_to_glm(self):
        """Very long + debug + multi-file → score ≥8 → complexity 3 → model-medium-hight max effort (level 3)."""
        decision = escalate(
            agent_model="model-medium-hight",
            agent_reasoning_effort="medium",
            estimated_tokens=50000,
            estimated_files=5,
            user_message="\n".join([
                "Debug this critical security vulnerability.",
                "Find the root cause by analyzing the entire architecture.",
                "Trace the call stack through all services:",
            ] + [f"  - check service/{i}/handler.ts for issues" for i in range(20)] + [
                "Verify the proposed fix against all edge cases.",
                "Add regression tests for each component.",
                "Also document the investigation process.",
            ]),
            cfg=self.cfg,
            session_id="e2e-test-5",
        )
        self.assertTrue(decision.should_switch)
        self.assertEqual(decision.recommended_model, "model-medium-hight")
        self.assertEqual(decision.recommended_level, 3)

    def test_high_pressure_escalates_to_glm(self):
        """Context pressure > 70% → model-medium-hight max effort (level 3)."""
        decision = escalate(
            agent_model="model-low-to-medium",
            agent_reasoning_effort="medium",
            estimated_tokens=800000,  # > 70% of 1M
            estimated_files=5,
            user_message="summarize this long conversation",
            cfg=self.cfg,
            session_id="e2e-test-6",
        )
        self.assertTrue(decision.should_switch)
        self.assertEqual(decision.recommended_model, "model-medium-hight")
        self.assertEqual(decision.recommended_level, 3)

    def test_downgrade_from_vip_to_9router(self):
        """Task done → downgrade to model-low-to-medium."""
        decision = escalate(
            agent_model="model-medium-hight",
            agent_reasoning_effort="max",
            estimated_tokens=2000,
            estimated_files=0,
            user_message="thanks, that's perfect",
            cfg=self.cfg,
            session_id="e2e-test-7",
        )
        self.assertTrue(decision.should_switch)
        self.assertEqual(decision.recommended_model, "model-low-to-medium")
        self.assertEqual(decision.recommended_level, 0)


if __name__ == '__main__':
    unittest.main()
