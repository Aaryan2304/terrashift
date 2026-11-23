"""CLI help must precede the overwrite guard; review paths stay independent."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class HelpOrderingTests(unittest.TestCase):
    def _run(self, script, *args):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / ("review" if script.startswith("build_review") else "misc") / script), *args],
                              capture_output=True, text=True, cwd=ROOT,
                              env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin",
                                   "SENTINEL_DATA_ROOT": "/home/aaryan/projects/skyserve"})

    def test_help_succeeds_before_the_guard(self):
        for script in ("build_review_package.py", "build_review_locations.py"):
            done = self._run(script, "--help")
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertNotIn("refusing to overwrite", done.stdout)

    def test_guard_still_refuses_on_normal_execution(self):
        out = ROOT / "outputs/submission_review_v1"
        before = sorted(p.name for p in out.iterdir()) if out.exists() else []
        done = self._run("build_review_package.py")
        self.assertEqual(done.returncode, 2)
        self.assertIn("refusing to overwrite", done.stdout)
        after = sorted(p.name for p in out.iterdir()) if out.exists() else []
        self.assertEqual(before, after)          # nothing deleted or added


class ReviewPathIndependenceTests(unittest.TestCase):
    def test_predictions_match_the_sibling_cli(self):
        import importlib
        pkg = importlib.import_module("scripts.review.build_review_package")
        loc = importlib.import_module("scripts.review.build_review_locations")
        self.assertEqual(pkg.PREDICTIONS, loc.PREDICTIONS)


if __name__ == "__main__":
    unittest.main()
