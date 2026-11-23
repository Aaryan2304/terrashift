"""Contracts for the P2a.6 extractions and the restored session APIs."""

import importlib
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

import terrashift.evaluation.sessions as sessions
import terrashift.geospatial.components as components
import terrashift.geospatial.rendering as rendering
import terrashift.inference.checkpoints as checkpoints

ROOT = Path(__file__).resolve().parents[1]


class RestoredSessionApiTests(unittest.TestCase):
    """P2a.5 briefly overwrote these; they must stay importable and unchanged."""

    def test_signatures(self):
        self.assertEqual(str(inspect.signature(sessions.evaluate_sessions)),
                         "(sessions, entries, thresholds, batch_size=4)")
        self.assertEqual(str(inspect.signature(sessions.evaluate_ort)),
                         "(session, loader, input_names, threshold)")

    def test_quantize_reexports_them(self):
        quantize = importlib.import_module("quantize")
        self.assertIs(quantize.evaluate_sessions, sessions.evaluate_sessions)
        self.assertIs(quantize.evaluate_ort, sessions.evaluate_ort)

    def test_validation_helpers_present(self):
        for name in ("PAIRS", "PATCH", "STRIDE", "BATCH", "build_datasets", "load_model", "predict_pair"):
            self.assertTrue(hasattr(sessions, name), name)


class ExtractionIdentityTests(unittest.TestCase):
    def test_component_extract_identity(self):
        script = importlib.import_module("scripts.registers.components_to_geojson")
        self.assertIs(script.extract, components.extract)

    def test_enumerate_support_identity(self):
        script = importlib.import_module("scripts.audit.check_checkpoint_input_shapes")
        self.assertIs(script.enumerate_support, checkpoints.enumerate_support)

    def test_panel_helpers_identity(self):
        script = importlib.import_module("scripts.reporting.make_evidence_panel")
        for name in ("_window_bounds", "read_rgb_window", "render_panel", "window_for_bounds", "polygon_rings"):
            self.assertIs(getattr(script, name), getattr(rendering, name), name)


class RenderingContractTests(unittest.TestCase):
    def test_rendering_import_loads_no_raster(self):
        probe = ("import sys, terrashift.geospatial.rendering;"
                 "print(any(m in sys.modules for m in ('torch',)))")
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                              env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
