import io
import subprocess
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from run import cache_advice, execute, recall, recall_summary


class RecallTests(unittest.TestCase):
    def test_child_failure_preserves_exit_status_and_exposes_native_diagnostic(self):
        failed = subprocess.CompletedProcess(["binary", "build"], 1, "", "native contract failed\n")
        stderr = io.StringIO()
        with patch("run.subprocess.run", return_value=failed), redirect_stderr(stderr):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                execute(Path("binary"), ["build"], None, "build")
        self.assertEqual(caught.exception.returncode, 1)
        self.assertIn("native contract failed", stderr.getvalue())

    def test_missing_results_are_misses_and_duplicates_do_not_inflate_recall(self):
        self.assertEqual(recall([1, 2, 3, 4], [1, 1, 2]), 0.5)
        self.assertEqual(recall([1, 2, 3, 4], []), 0.0)

    def test_empty_filter_is_not_a_perfect_recall_sample(self):
        self.assertIsNone(recall([], []))
        with self.assertRaises(ValueError):
            recall([], [1])

    def test_denominator_is_eligible_oracle_count_and_query_counts_must_match(self):
        oracle = {"queries": [{"keys": [4, 5]}, {"keys": []}]}
        found = {"queries": [{"keys": [4]}, {"keys": []}]}
        summary = recall_summary(oracle, found)
        self.assertEqual(summary["mean"], 0.5)
        self.assertEqual(summary["queries"], 1)
        with self.assertRaises(ValueError):
            recall_summary(oracle, {"queries": []})

    def test_uncontrolled_cache_does_not_touch_a_file(self):
        with patch("run.os.open", side_effect=AssertionError("must not open")):
            self.assertEqual(cache_advice(Path("unused"), "oxv", "uncontrolled")["result"], "not_requested")

    @unittest.skipUnless(hasattr(__import__("os"), "posix_fadvise"), "Linux/Unix cache advice")
    def test_advice_is_scoped_to_the_selected_fixture_and_does_not_claim_eviction(self):
        with patch("run.os.open", return_value=42) as opened, patch("run.os.fsync") as sync, patch("run.os.posix_fadvise") as advise, patch("run.os.close") as close:
            result = cache_advice(Path("fixture"), "usearch-view", "advisory-cold")
            self.assertEqual(opened.call_args.args[0], Path("fixture/index.usearch"))
            sync.assert_called_once_with(42)
            self.assertEqual(advise.call_args.args[:3], (42, 0, 0))
            close.assert_called_once_with(42)
            self.assertFalse(result["eviction_verified"])
            self.assertEqual(result["result"], "accepted_advisory_no_guarantee")


if __name__ == "__main__":
    unittest.main()
