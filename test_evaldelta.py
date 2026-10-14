import contextlib
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from demo import fixture, write_rows
from evaldelta import InvalidRun, bootstrap, compare, load_run, main, quantile


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / "base.jsonl"
        self.cand = Path(self.temp.name) / "cand.jsonl"
        self.b, self.c = fixture()

    def run_compare(self, **kwargs):
        write_rows(self.base, self.b)
        write_rows(self.cand, self.c)
        return compare(self.base, self.cand, samples=200, **kwargs)

    def test_hidden_slice_regression(self):
        result = self.run_compare()
        self.assertEqual(result["decision"], "block")
        self.assertEqual(result["slices"][0]["quality_status"], "pass")
        self.assertAlmostEqual(result["slices"][0]["mean_delta"], 0.1)
        self.assertEqual(result["blockers"][0]["slice"], "tag:long_context")
        self.assertEqual(result["blockers"][0]["quality"], "regression")

    def test_identical_scores_pass(self):
        self.c = copy.deepcopy(self.b)
        self.assertEqual(self.run_compare()["decision"], "pass")

    def test_uniform_improvement_passes(self):
        for row in self.c:
            row["score"] = 0.9
        self.assertEqual(self.run_compare()["decision"], "pass")

    def test_inconclusive_is_not_a_pass(self):
        for i, row in enumerate(self.c):
            row["score"] = 1 if i % 2 else 0.4
        result = self.run_compare(margin=0)
        self.assertEqual(result["decision"], "block")
        self.assertEqual(result["slices"][0]["quality_status"], "inconclusive")

    def test_small_slice_blocks(self):
        self.b[0]["tags"].append("rare")
        self.c[0]["tags"].append("rare")
        rare = next(r for r in self.run_compare()["slices"] if r["slice"] == "tag:rare")
        self.assertEqual(rare["quality_status"], "insufficient_data")
        self.assertIsNone(rare["interval_95"])

    def test_latency_budget_independent_of_quality(self):
        self.c = copy.deepcopy(self.b)
        for row in self.c:
            row["latency_ms"] = 1200
        result = self.run_compare()
        self.assertEqual(result["slices"][0]["quality_status"], "pass")
        self.assertEqual(result["blockers"][0]["latency"], "over_budget")

    def test_latency_boundary_inclusive(self):
        self.c = copy.deepcopy(self.b)
        self.assertEqual(self.run_compare(max_p95_ms=200)["decision"], "pass")

    def test_reordered_rows_identical_report(self):
        expected = self.run_compare()
        random.Random(9).shuffle(self.b)
        self.c.reverse()
        self.assertEqual(expected, self.run_compare())

    def test_repeated_run_identical(self):
        self.assertEqual(self.run_compare(), self.run_compare())

    def test_missing_case_rejected(self):
        self.c.pop()
        with self.assertRaisesRegex(InvalidRun, "case IDs differ"):
            self.run_compare()

    def test_duplicate_case_rejected(self):
        self.c.append(self.c[0])
        with self.assertRaisesRegex(InvalidRun, "duplicate case"):
            self.run_compare()

    def test_changed_fingerprint_rejected(self):
        self.c[0]["case_hash"] = "a" * 64
        with self.assertRaisesRegex(InvalidRun, "case_hash differs"):
            self.run_compare()

    def test_changed_tags_rejected(self):
        self.c[0]["tags"] = ["other"]
        with self.assertRaisesRegex(InvalidRun, "tags differs"):
            self.run_compare()

    def test_scorer_version_mismatch_rejected(self):
        for row in self.c:
            row["scorer"] = "different-scorer"
        with self.assertRaisesRegex(InvalidRun, "scorer differs"):
            self.run_compare()

    def test_suite_version_mismatch_rejected(self):
        for row in self.c:
            row["suite"] = "different-suite"
        with self.assertRaisesRegex(InvalidRun, "suite differs"):
            self.run_compare()

    def test_mixed_runs_rejected(self):
        self.c[0]["run"] = "other"
        with self.assertRaisesRegex(InvalidRun, "mixed run"):
            self.run_compare()

    def test_invalid_scores_rejected(self):
        for value in (True, -1, 2, None, "0.5", float("nan"), float("inf"), 10**500):
            with self.subTest(value=repr(value)[:20]):
                self.c[0]["score"] = value
                with self.assertRaises(InvalidRun):
                    self.run_compare()

    def test_invalid_latencies_rejected(self):
        for value in (False, -1, None, float("nan"), float("inf")):
            self.c[0]["latency_ms"] = value
            with self.assertRaises(InvalidRun):
                self.run_compare()

    def test_unknown_fields_rejected(self):
        self.c[0]["secret_prompt"] = "not accepted"
        with self.assertRaisesRegex(InvalidRun, "incorrect fields"):
            self.run_compare()

    def test_invalid_tags_rejected(self):
        for value in ("tag", ["x", "x"], [None], ["<script>"], [str(i) for i in range(9)]):
            self.c[0]["tags"] = value
            with self.assertRaises(InvalidRun):
                self.run_compare()

    def test_duplicate_json_key_rejected(self):
        self.base.write_text('{"id":"a","id":"b"}\n')
        with self.assertRaisesRegex(InvalidRun, "duplicate JSON key"):
            load_run(self.base)

    def test_empty_run_rejected(self):
        self.base.write_text("\n ")
        with self.assertRaisesRegex(InvalidRun, "empty run"):
            load_run(self.base)

    def test_file_limit(self):
        write_rows(self.base, self.b)
        with patch("evaldelta.MAX_BYTES", 20):
            with self.assertRaisesRegex(InvalidRun, "8 MB"):
                load_run(self.base)

    def test_row_limit(self):
        write_rows(self.base, self.b)
        with patch("evaldelta.MAX_CASES", 2):
            with self.assertRaisesRegex(InvalidRun, "too many cases"):
                load_run(self.base)

    def test_config_rejected(self):
        for config in ({"margin": -1}, {"margin": float("nan")}, {"seed": True},
                       {"min_cases": 1}, {"max_p95_ms": -1}):
            with self.subTest(config=config), self.assertRaises(InvalidRun):
                self.run_compare(**config)

    def test_bootstrap_is_paired_and_degenerate_for_constant_deltas(self):
        self.assertEqual(bootstrap([0.25] * 30, 200, 3), [0.25, 0.25])

    def test_quantile_known_values(self):
        self.assertEqual(quantile([0, 10, 20, 30], .5), 15)
        self.assertAlmostEqual(quantile([30, 0, 10, 20], .95), 28.5)
        self.assertEqual(quantile([7], .95), 7)

    def test_worst_cases_are_sorted(self):
        result = self.run_compare()
        self.assertEqual(len(result["regressions"]), 10)
        self.assertEqual(result["regressions"][0]["id"], "case-000")

    def test_tag_order_normalized(self):
        for b, c in zip(self.b, self.c):
            b["tags"] = ["z", "a"]
            c["tags"] = ["a", "z"]
        self.assertEqual(len(self.run_compare()["slices"]), 3)

    def test_work_budget_rejected_before_bootstrap(self):
        self.b = [{**self.b[i % 100], "id": str(i)} for i in range(4000)]
        self.c = copy.deepcopy(self.b)
        write_rows(self.base, self.b)
        write_rows(self.cand, self.c)
        with self.assertRaisesRegex(InvalidRun, "work limit"):
            compare(self.base, self.cand, samples=10000)

    def test_too_many_tags_rejected(self):
        for i, (b, c) in enumerate(zip(self.b, self.c)):
            b["tags"] = c["tags"] = [f"tag-{i}"]
        with self.assertRaisesRegex(InvalidRun, "32 tag slices"):
            self.run_compare()

    def test_bootstrap_matches_simple_two_point_distribution(self):
        # Two draws from {-1, 1}: means -1, 0, 1 with probabilities 1/4, 1/2, 1/4.
        self.assertEqual(bootstrap([-1, 1], 1000, 42), [-1, 1])

    def test_content_hash_changes_with_score(self):
        first = self.run_compare()
        self.c[0]["score"] = 0.5
        second = self.run_compare()
        self.assertNotEqual(first["candidate_sha256"], second["candidate_sha256"])

    def test_cli_exit_codes_and_json(self):
        self.run_compare()
        proc = subprocess.run([sys.executable, "evaldelta.py", str(self.base), str(self.cand),
                               "--samples", "100"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(json.loads(proc.stdout)["decision"], "block")
        write_rows(self.cand, self.b)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.base), str(self.cand), "--samples", "100"]), 0)
        self.cand.write_text("not JSON")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(self.base), str(self.cand)]), 2)


if __name__ == "__main__":
    unittest.main()
