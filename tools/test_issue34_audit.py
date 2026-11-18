import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import issue34_audit as audit


ROOT = Path(__file__).resolve().parents[1]


class AuditRecipeTests(unittest.TestCase):
    def test_legacy_window_port_matches_published_spans_and_keys(self):
        golden = json.loads((ROOT / "tests/data/long_line_windows_golden.json").read_text())
        for case in golden["cases"]:
            with self.subTest(source=case["source"]):
                spans = audit.legacy_spans(case["text"])
                self.assertEqual([list(span) for span in spans], case["spans"])
                keys = [hashlib.sha256(("[PASSAGE] " + case["text"][a:b].strip()).encode())
                        .hexdigest()[:32] for a, b in spans]
                self.assertEqual(keys, case["keys"])

    def test_short_anchor_failure_is_not_confused_with_substring_duplicates(self):
        lines = ["א" * 300, "ב" * 300, "עוגן קצר", "ג" * 300]
        passage = next(row for row in audit.passages(lines, [0] * 4) if row[0] == 2)
        self.assertTrue(passage[3])
        self.assertNotIn("עוגן קצר", passage[2])

    def test_context_cannot_cross_heading_boundary(self):
        rows = list(audit.passages(["א" * 400, "כותרת", "תוכן להקשר"], [0, 1, 1]))
        self.assertEqual(rows[1][2], "כותרת תוכן להקשר")

    def test_compiled_engine_normalizer_retains_legacy_note_markers(self):
        engine = ROOT.parent / "otzaria_search_engine"
        if not engine.exists():
            self.skipTest("engine checkout is required for exact source extraction")
        with tempfile.TemporaryDirectory() as directory:
            binary = audit.compile_normalizer(engine, Path(directory))
            actual = audit.normalize(binary, ["תורה<sup>30</sup>",
                "מְ<sup>נַ</sup>שֶּׁה", "א<br>ב", "\ufeffראשית", ""])
        self.assertEqual(actual, ["תורה 30", "מנשה", "א ב", "\ufeffראשית", ""])


if __name__ == "__main__":
    unittest.main()
