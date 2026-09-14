from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from flywire_review.standalone import (
    agentic_review_prompt,
    compare_with_competitor,
    parse_review_response,
    repository_receipt,
)


class StandaloneReviewTests(unittest.TestCase):
    def test_agentic_prompt_replaces_the_prompt_only_context_boundary(self) -> None:
        prompt = agentic_review_prompt(
            "Review only the supplied code and return a JSON array",
            base="base-sha",
            head="head-sha",
        )
        self.assertIn("read/search any repository file", prompt)
        self.assertIn("git diff base-sha..head-sha", prompt)
        self.assertNotIn("Review only the supplied code", prompt)

    def test_repository_receipt_requires_exact_clean_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Test"], check=True
            )
            (root / "a.txt").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "a.txt"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
            base = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            (root / "a.txt").write_text("head\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "commit", "-qam", "head"], check=True)
            head = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            receipt = repository_receipt(root, base=base, head=head)
            self.assertEqual(receipt["head"], head)
            self.assertEqual(receipt["changed_files"], 1)
            self.assertEqual(receipt["changed_file_paths"], ["a.txt"])

    def test_parser_accepts_fenced_array_and_enforces_changed_pr_files(self) -> None:
        response = """```json
[{"file":"src/a.py","line":12,"severity":"high","comment":"bug","evidence":"line 12"}]
```"""
        rows, error = parse_review_response(
            response,
            reviewer_id="fly-01",
            valid_files={"src/a.py"},
        )
        self.assertIsNone(error)
        self.assertEqual(rows[0]["reviewer_id"], "fly-01")
        rejected, error = parse_review_response(
            response.replace("src/a.py", "secret.py"),
            reviewer_id="fly-01",
            valid_files={"src/a.py"},
        )
        self.assertEqual(rejected, [])
        self.assertIn("outside", error)

    def test_comparison_requires_complete_explicit_grades(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review = root / "review.json"
            truth = root / "truth.json"
            competitor = root / "competitor.json"
            grades = root / "grades.json"
            output = root / "comparison.json"
            review.write_text(json.dumps({
                "contract": {"holdout": "case"},
                "repository_receipt": {"head": "abc"},
                "findings": [{"id": "F001"}],
            }), encoding="utf-8")
            truth.write_text(json.dumps({"prs": [{
                "key": "case",
                "truth": [{"truth_id": "T1", "truth_register": "pre"}],
            }]}), encoding="utf-8")
            competitor.write_text(json.dumps({"pull_requests": [{
                "key": "case",
                "repo": "org/repo",
                "pull": 1,
                "reviewed_head": "abc",
                "comparison_relation": "same_commit",
                "competitor": "Tool",
                "competitor_findings": [{"id": 9}],
            }]}), encoding="utf-8")
            grades.write_text(json.dumps({
                "key": "case",
                "reviewed_head": "abc",
                "bugbrain": [{"id": "F001", "truth_id": None, "reason": "false"}],
                "competitor": [{"id": "9", "truth_id": "T1", "reason": "confirmed"}],
            }), encoding="utf-8")
            result = compare_with_competitor(SimpleNamespace(
                review=review,
                truth_register=truth,
                competitor_evidence=competitor,
                grades=grades,
                out=output,
            ))
            self.assertEqual(result["bugbrain"]["recall"], 0.0)
            self.assertEqual(result["competitor"]["precision"], 1.0)
            self.assertEqual(result["competitor"]["recall"], 1.0)
            self.assertTrue(output.exists())

    def test_comparison_rejects_different_reviewed_heads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review = root / "review.json"
            truth = root / "truth.json"
            competitor = root / "competitor.json"
            grades = root / "grades.json"
            review.write_text(json.dumps({
                "contract": {"holdout": "case"},
                "repository_receipt": {"head": "fly-head"},
                "findings": [],
            }), encoding="utf-8")
            truth.write_text(json.dumps({"prs": [{
                "key": "case",
                "truth": [],
            }]}), encoding="utf-8")
            competitor.write_text(json.dumps({"pull_requests": [{
                "key": "case",
                "reviewed_head": "other-head",
                "comparison_relation": "same_commit",
                "competitor_findings": [],
            }]}), encoding="utf-8")
            grades.write_text(json.dumps({
                "key": "case",
                "reviewed_head": "fly-head",
                "bugbrain": [],
                "competitor": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact same head"):
                compare_with_competitor(SimpleNamespace(
                    review=review,
                    truth_register=truth,
                    competitor_evidence=competitor,
                    grades=grades,
                    out=root / "comparison.json",
                ))


if __name__ == "__main__":
    unittest.main()
