"""Contracts for the extracted validation-session and rendering modules."""

import importlib
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

import terrashift.data.local_patches as local_patches
import terrashift.data.oscd_original as oscd_original
import terrashift.data.patch_grid as patch_grid
import terrashift.data.patch_sampling as patch_sampling
import terrashift.evaluation.sessions as sessions
import terrashift.geospatial.rendering as rendering

ROOT = Path(__file__).resolve().parents[1]


class SessionContractTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(sessions.PAIRS, ("A", "B"))
        self.assertEqual((sessions.PATCH, sessions.STRIDE, sessions.BATCH), (256, 128, 4))

    def test_signatures_unchanged(self):
        self.assertEqual(str(inspect.signature(sessions.build_datasets)), "(args)")
        self.assertEqual(str(inspect.signature(sessions.load_model)), "(path, device)")
        self.assertEqual(str(inspect.signature(sessions.predict_pair)),
                         "(model, pair, entry, device)")

    def test_script_reexports_the_same_objects(self):
        script = importlib.import_module("scripts.evaluation.analyse_validation_thresholds")
        for name in ("PAIRS", "BATCH", "build_datasets", "load_model", "predict_pair"):
            self.assertIs(getattr(script, name), getattr(sessions, name), name)
        self.assertIs(script.rgb_background, rendering.rgb_background)

    def test_import_loads_no_data(self):
        probe = ("import sys, terrashift.evaluation.sessions, terrashift.geospatial.rendering;"
                 "print('rasterio' in sys.modules)")
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                              env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        self.assertEqual(done.returncode, 0, done.stderr)


class PatchGridOwnershipTests(unittest.TestCase):
    def test_one_object_across_four_modules(self):
        self.assertIs(patch_grid.patch_origins, oscd_original.patch_origins)
        self.assertIs(patch_grid.patch_origins, patch_sampling.patch_origins)
        self.assertIs(patch_grid.patch_origins, local_patches.patch_origins)

    def test_grid_module_does_not_import_oscd_or_torch(self):
        probe = ("import sys, terrashift.data.patch_grid as g;"
                 "print(any(m in sys.modules for m in ('torch','rasterio','terrashift.data.oscd_original')))")
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                              env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        self.assertEqual(done.stdout.strip(), "False", done.stderr)


if __name__ == "__main__":
    unittest.main()
