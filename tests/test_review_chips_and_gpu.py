"""Contracts for the extracted review-chip and GPU-pipeline helpers."""

import importlib
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

import terrashift.inference.gpu as gpu
import terrashift.review.chips as chips

ROOT = Path(__file__).resolve().parents[1]

CHIP_SYMBOLS = ("component_scores", "render_chips", "reviewed_components",
                "rings_in_chip_pixels", "save_chip", "chip_bounds", "stretch",
                "MARGIN_PX", "STRUCTURE")


class ChipIdentityTests(unittest.TestCase):
    def test_script_reexports_the_package_objects(self):
        script = importlib.import_module("scripts.review.build_review_package")
        for name in CHIP_SYMBOLS:
            self.assertIs(getattr(script, name), getattr(chips, name), name)

    def test_cli_configuration_stays_in_the_script(self):
        # PREDICTIONS/PARTITIONS/LABELS derive from the repository root: configuration,
        # not reusable logic, so they deliberately remain in the CLI module.
        script = importlib.import_module("scripts.review.build_review_package")
        for name in ("PREDICTIONS", "PARTITIONS", "LABELS"):
            self.assertTrue(hasattr(script, name), name)
            self.assertFalse(hasattr(chips, name), name)

    def test_signatures_unchanged(self):
        self.assertEqual(str(inspect.signature(chips.chip_bounds)), "(state, component_id)")
        self.assertEqual(str(inspect.signature(chips.stretch)), "(before, after)")


class GpuHelperTests(unittest.TestCase):
    def test_identity(self):
        script = importlib.import_module("scripts.evaluation.gpu_inference_evaluation")
        self.assertIs(script.GpuSampler, gpu.GpuSampler)
        self.assertIs(script.build_entries, gpu.build_entries)

    def test_sampler_is_a_thread_and_constructible_without_cuda(self):
        import threading
        self.assertTrue(issubclass(gpu.GpuSampler, threading.Thread))
        sampler = gpu.GpuSampler(interval=0.1)
        self.assertFalse(sampler.is_alive())      # constructed, never started
        self.assertTrue(sampler.daemon)

    def test_import_does_not_initialise_cuda(self):
        probe = ("import torch, terrashift.inference.gpu; print(torch.cuda.is_initialized())")
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                              env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
