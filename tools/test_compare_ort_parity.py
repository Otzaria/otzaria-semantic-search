"""The comparison must reject corruption as well as a real ranking regression."""

import copy
import unittest

from compare_ort_parity import compare


class ParityComparisonTests(unittest.TestCase):
    def setUp(self):
        self.probe = {
            "schema": 1, "backend": "fixture", "max_tokens": 32,
            "texts": ["a", "b"], "tokens": [[1], [2]],
            "neighbours_at_5": [[1], [0]],
            "raw_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "normalized_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "measurement": {"iterations": 100, "threads": 1, "sessions": 1,
                            "native_allocations_measured": False, "p50_us": 10},
        }

    def test_timing_does_not_gate_numerical_parity(self):
        candidate = copy.deepcopy(self.probe)
        candidate["measurement"]["p50_us"] = 10000
        self.assertTrue(compare(self.probe, candidate)["raw_vectors"]["exactly_identical"])

    def test_small_delta_is_reported_and_larger_delta_rejected(self):
        candidate = copy.deepcopy(self.probe)
        candidate["raw_vectors"][0][1] = 1e-7
        report = compare(self.probe, candidate)
        self.assertEqual(report["raw_vectors"]["max_abs_delta"], 1e-7)
        candidate["raw_vectors"][0][1] = 1e-4
        with self.assertRaisesRegex(ValueError, "maximum absolute delta"):
            compare(self.probe, candidate)

    def test_tokens_ranking_shapes_and_nonfinite_values_fail(self):
        changes = [
            ("tokens", [[2], [1]]),
            ("neighbours_at_5", [[0], [1]]),
            ("raw_vectors", [[1.0, 0.0]]),
            ("raw_vectors", [[1.0], [0.0, 1.0]]),
            ("raw_vectors", [[float("nan"), 0.0], [0.0, 1.0]]),
            ("raw_vectors", [[0.0, 0.0], [0.0, 1.0]]),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                candidate = copy.deepcopy(self.probe)
                candidate[field] = value
                with self.assertRaises(ValueError):
                    compare(self.probe, candidate)


if __name__ == "__main__":
    unittest.main()
