"""Tests for the optional CUDA/FP16 path in the production SAFE-pair inference script.

The heavy integration test is skipped unless the SAFE products are present.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inference" / "predict_safe_pair.py"


def _load():
    spec = importlib.util.spec_from_file_location("predict_safe_pair", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExecutionRequestTests(unittest.TestCase):
    """A CUDA or fp16 request must be validated, never silently downgraded."""

    def setUp(self):
        self.module = _load()

    def test_cpu_default_stays_batch_one(self):
        self.assertEqual(self.module.resolve_execution("cpu", "fp32", None,
                                                       cuda_available=False), 1)

    def test_cuda_default_batch_is_eight(self):
        self.assertEqual(self.module.resolve_execution("cuda", "fp16", None,
                                                       cuda_available=True), 8)

    def test_explicit_batch_size_wins(self):
        self.assertEqual(self.module.resolve_execution("cuda", "fp16", 4,
                                                       cuda_available=True), 4)

    def test_cuda_without_a_gpu_raises(self):
        with self.assertRaises(SystemExit):
            self.module.resolve_execution("cuda", "fp32", None, cuda_available=False)

    def test_fp16_on_cpu_raises(self):
        with self.assertRaises(SystemExit):
            self.module.resolve_execution("cpu", "fp16", None, cuda_available=False)

    def test_zero_batch_size_raises(self):
        with self.assertRaises(SystemExit):
            self.module.resolve_execution("cpu", "fp32", 0, cuda_available=False)


class HalfPrecisionWrapperTests(unittest.TestCase):
    """The wrapper must cast inputs to half, which is the method that was evaluated."""

    def setUp(self):
        self.module = _load()

    def test_inputs_arrive_in_half_precision(self):
        seen = {}

        class Inner(torch.nn.Module):
            def forward(self, a, b):
                seen["dtype"] = a.dtype
                seen["equal"] = bool(torch.equal(a, b))
                return (a + b)

        inner = Inner().half()
        wrapper = self.module.HalfInputModel(inner)
        output = wrapper(torch.ones(2, 4, 8, 8, dtype=torch.float32),
                         torch.ones(2, 4, 8, 8, dtype=torch.float32))
        self.assertEqual(seen["dtype"], torch.float16)
        self.assertTrue(seen["equal"])
        self.assertEqual(output.dtype, torch.float16)

    def test_output_shape_is_preserved(self):
        class TwoInput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 1, 1)

            def forward(self, a, b):
                return self.conv(a - b)

        wrapper = self.module.HalfInputModel(TwoInput().half())
        output = wrapper(torch.zeros(3, 4, 16, 16), torch.zeros(3, 4, 16, 16))
        self.assertEqual(tuple(output.shape), (3, 1, 16, 16))


class BatchingEquivalenceTests(unittest.TestCase):
    """Batching must not change the numbers or the coverage."""

    def setUp(self):
        from terrashift.inference.sliding import sliding_window_predict
        self.predict = sliding_window_predict

    def _toy(self):
        class Toy(torch.nn.Module):
            def forward(self, a, b):
                # a genuinely input-dependent, deterministic response
                return (a.mean(dim=1, keepdim=True) - b.mean(dim=1, keepdim=True)) * 4.0
        return Toy()

    def test_batch_size_does_not_change_the_probability_map(self):
        model, device = self._toy(), torch.device("cpu")
        x1 = torch.zeros(4, 200, 180)
        x2 = torch.zeros(4, 200, 180)
        x2[:, 40:90, 30:70] = 1.0
        single, _ = self.predict(model, x1, x2, patch_size=64, stride=32,
                                 device=device, batch_size=1)
        batched, _ = self.predict(model, x1, x2, patch_size=64, stride=32,
                                  device=device, batch_size=7)
        np.testing.assert_allclose(single, batched, rtol=0, atol=0)

    def test_incomplete_final_batch_still_covers_every_pixel(self):
        model, device = self._toy(), torch.device("cpu")
        x1 = torch.zeros(4, 130, 70)
        x2 = torch.zeros(4, 130, 70)
        x2[:, 10:20, 10:20] = 1.0
        probabilities, _ = self.predict(model, x1, x2, patch_size=64, stride=32,
                                        device=device, batch_size=5)
        self.assertEqual(probabilities.shape, (130, 70))
        self.assertTrue(np.isfinite(probabilities).all())
        # the block is brighter in date 2, so the change logit is negative there
        self.assertLess(probabilities[15, 15], probabilities[0, 0])

    def test_odd_window_count_with_batch_two_is_finite(self):
        model, device = self._toy(), torch.device("cpu")
        x1 = torch.zeros(4, 100, 100)
        x2 = torch.zeros(4, 100, 100)
        probabilities, _ = self.predict(model, x1, x2, patch_size=64, stride=32,
                                        device=device, batch_size=2)
        self.assertEqual(probabilities.shape, (100, 100))
        self.assertTrue(np.isfinite(probabilities).all())


class OverlapAccumulationTests(unittest.TestCase):
    def test_each_pixel_counts_once_and_averages_equally(self):
        from terrashift.evaluation.lattice import LatticeAccumulator
        accumulator = LatticeAccumulator((40, 40), col_offset=0)
        valid = np.ones((16, 16), dtype=bool)
        accumulator.add(0, 0, np.full((16, 16), 0.25, dtype=np.float32), valid)
        accumulator.add(8, 8, np.full((16, 16), 0.75, dtype=np.float32), valid)
        self.assertEqual(int(accumulator.count[12, 12]), 2)
        self.assertEqual(int(accumulator.count[2, 2]), 1)
        self.assertAlmostEqual(float(accumulator.total[12, 12] / accumulator.count[12, 12]),
                               0.5, places=6)


@unittest.skipUnless(os.environ.get("SENTINEL_DATA_ROOT"), "SAFE products are not configured")
class PipelineIntegrationTests(unittest.TestCase):
    """Run the real script on a small window and inspect the written rasters."""

    def _run(self, extra):
        import shutil
        import subprocess
        import sys
        # mkdtemp, not TemporaryDirectory: the caller must still see the files afterwards.
        directory = Path(tempfile.mkdtemp(prefix="predict_safe_pair_"))
        self.addCleanup(shutil.rmtree, directory, True)
        command = [sys.executable, str(SCRIPT), "--pair", "A",
                   "--checkpoint", str(ROOT / "outputs/checkpoints/"
                                       "local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"),
                   "--normalisation", str(ROOT / "outputs/dataset/v2/normalisation.json"),
                   "--output-dir", str(directory), "--window", "5000,5192,5000,5192",
                   "--status", "integration_test"] + extra
        result = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT))
        return result, directory

    def test_cpu_path_writes_the_expected_raster_contract(self):
        import json
        import rasterio
        result, directory = self._run([])
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        with rasterio.open(directory / "prediction_prob_A.tif") as handle:
            self.assertEqual(handle.dtypes[0], "float32")
            self.assertEqual(handle.nodata, -9999.0)
            self.assertEqual(str(handle.crs), "EPSG:32643")
            self.assertEqual((handle.height, handle.width), (192, 192))
            self.assertEqual(abs(handle.transform.a), 10.0)
            data = handle.read(1)
        self.assertTrue(np.isfinite(data[data != -9999]).all())
        config = json.loads((directory / "prediction_config.json").read_text())
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["precision"], "fp32")
        self.assertEqual(config["inference_batch_size"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device available")
    def test_cuda_fp16_matches_cpu_on_the_same_window(self):
        import rasterio
        cpu, cpu_dir = self._run([])
        self.assertEqual(cpu.returncode, 0, cpu.stderr[-800:])
        gpu, gpu_dir = self._run(["--device", "cuda", "--precision", "fp16",
                                  "--inference-batch-size", "8"])
        self.assertEqual(gpu.returncode, 0, gpu.stderr[-800:])
        with rasterio.open(cpu_dir / "prediction_prob_A.tif") as handle:
            reference = handle.read(1)
        with rasterio.open(gpu_dir / "prediction_prob_A.tif") as handle:
            candidate = handle.read(1)
            self.assertEqual(handle.dtypes[0], "float32")
            self.assertEqual(handle.nodata, -9999.0)
            self.assertEqual(str(handle.crs), "EPSG:32643")
        valid = (reference != -9999) & (candidate != -9999)
        np.testing.assert_array_equal(reference == -9999, candidate == -9999)
        difference = np.abs(reference[valid] - candidate[valid])
        # A loose bound on purpose: this window is outside the validation set, so it only
        # has to prove the two paths agree structurally. The authoritative parity figure is
        # the 159-window comparison (max |dP| 4.35e-3 on Pair A, 6.47e-3 on Pair B, AP
        # delta +0.0001).
        self.assertLess(float(difference.max()), 2e-2,
                        "FP16 diverged from the CPU reference by more than 2e-2")
        self.assertEqual(int((reference == -9999).sum()), int((candidate == -9999).sum()))


if __name__ == "__main__":
    unittest.main()
