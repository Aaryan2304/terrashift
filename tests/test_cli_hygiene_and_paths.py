"""CLI help hygiene and output-path portability."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HELP_SCRIPTS = (
    "scripts/audit/verify_masks.py",
    "scripts/review/build_negative_review.py",
    "scripts/review/validate_review_locations.py",
    "scripts/utilities/query_candidate_osm.py",
)


class HelpHygieneTests(unittest.TestCase):
    def test_help_exits_zero_and_prints_usage(self):
        for rel in HELP_SCRIPTS:
            done = subprocess.run([sys.executable, str(ROOT / rel), "--help"],
                                  capture_output=True, text=True, cwd=ROOT, timeout=120,
                                  env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
            self.assertEqual(done.returncode, 0, "%s: %s" % (rel, done.stderr[-300:]))
            self.assertTrue(done.stdout.strip(), rel)

    def test_help_creates_no_output_directory(self):
        before = set(p.name for p in (ROOT / "outputs").iterdir()) if (ROOT / "outputs").exists() else set()
        for rel in HELP_SCRIPTS:
            subprocess.run([sys.executable, str(ROOT / rel), "--help"], capture_output=True,
                           text=True, cwd=ROOT, timeout=120,
                           env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
        after = set(p.name for p in (ROOT / "outputs").iterdir()) if (ROOT / "outputs").exists() else set()
        self.assertEqual(before, after)


class OutputPathPortabilityTests(unittest.TestCase):
    def setUp(self):
        import importlib
        self.mod = importlib.import_module("scripts.evaluation.evaluate_locked_test")

    def test_path_inside_the_repository_is_relative(self):
        inside = ROOT / "outputs" / "evaluation" / "_portability_probe"
        self.assertEqual(self.mod.report_path(inside), "outputs/evaluation/_portability_probe")

    def test_path_outside_the_repository_is_absolute(self):
        outside = Path(tempfile.mkdtemp()) / "locked_repro"
        self.assertEqual(self.mod.report_path(outside), str(outside.resolve()))

    def test_relative_input_is_resolved(self):
        got = self.mod.report_path("outputs/tmp_probe")
        self.assertIn("outputs", got)


if __name__ == "__main__":
    unittest.main()
