from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.summarize_controller_benchmark import (
    build_report,
    exact_paired_sign_test,
    exact_task_bootstrap,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]


class ControllerBenchmarkStatisticsTests(unittest.TestCase):
    def test_wilson_interval_is_bounded_and_contains_rate(self) -> None:
        low, high = wilson_interval(5, 6)
        self.assertEqual([low, high], [0.436497, 0.969947])
        self.assertLess(low, 5 / 6)
        self.assertGreater(high, 5 / 6)

    def test_exact_sign_test_counts_only_discordant_pairs(self) -> None:
        self.assertEqual(exact_paired_sign_test(1, 1), 1.0)
        self.assertEqual(exact_paired_sign_test(2, 0), 0.5)
        self.assertEqual(exact_paired_sign_test(0, 0), 1.0)

    def test_exact_task_bootstrap_is_deterministic(self) -> None:
        self.assertEqual(
            exact_task_bootstrap([1.0, 0.0, 0.0, 0.0, -1.0, 0.0]),
            [-0.5, 0.5],
        )

    def test_published_summary_rebuilds_from_immutable_receipts(self) -> None:
        rebuilt = build_report(
            ROOT / "benchmarks" / "controller" / "manifest.json",
            ROOT / "results" / "controller-policy-report.json",
            ROOT / "benchmarks" / "controller" / "trajectories.json",
            ROOT / "results" / "controller-runs" / "live",
            ROOT / "results" / "controller-runs" / "direct",
        )
        published = json.loads(
            (ROOT / "results" / "controller-benchmark.json").read_text(encoding="utf-8")
        )
        self.assertEqual(rebuilt, published)


if __name__ == "__main__":
    unittest.main()
