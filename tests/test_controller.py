from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from flywire_review.controller import (
    _action_contract_compliant,
    _parse_final_object,
    _worker_prompt,
    run_verifier,
    summarize_codex_events,
)


class LiveControllerTests(unittest.TestCase):
    def test_real_codex_jsonl_shape_is_summarized(self) -> None:
        response = {
            "status": "continue",
            "summary": "found the symbol",
            "evidence": ["src/auth.py:12"],
            "proposed_next": "inspect",
        }
        events = [
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "I will search."},
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "/bin/zsh -lc \"rg authenticate src\"",
                    "aggregated_output": "src/auth.py:12",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": json.dumps(response)},
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        ]
        summary = summarize_codex_events("\n".join(json.dumps(row) for row in events))
        self.assertEqual(summary["tool_actions"], ["search"])
        self.assertEqual(_parse_final_object(summary["final_message"]), response)
        self.assertEqual(summary["usage"]["input_tokens"], 100)

    def test_worker_prompt_grants_repository_capacity_but_one_action(self) -> None:
        prompt = _worker_prompt(
            goal="fix authentication",
            action="patch",
            mode="implement",
            observation_history=[{"previous_result": {"status": "started"}}],
        )
        self.assertIn("complete checkout", prompt)
        self.assertIn("Make the smallest justified source edit", prompt)
        self.assertIn("Perform one bounded step only", prompt)
        self.assertFalse(_action_contract_compliant("inspect", []))
        self.assertTrue(_action_contract_compliant("inspect", ["search"]))
        self.assertFalse(_action_contract_compliant("test", ["inspect"]))

    def test_verifier_reward_comes_from_exit_code(self) -> None:
        passed = run_verifier(
            Path.cwd(),
            [sys.executable, "-c", "raise SystemExit(0)"],
            timeout=30,
        )
        failed = run_verifier(
            Path.cwd(),
            [sys.executable, "-c", "raise SystemExit(3)"],
            timeout=30,
        )
        self.assertEqual((passed["passed"], passed["reward"]), (True, 1.0))
        self.assertEqual((failed["passed"], failed["reward"]), (False, 0.0))


if __name__ == "__main__":
    unittest.main()
