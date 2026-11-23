"""Contract tests for the georeferenced inference path.

These tests cover the checkpoint contract and the preprocessing parity that the
inference script must respect. They do not run a model on Sentinel-2 data; the
full path is exercised by ``scripts/inference/predict_safe_pair.py`` on the real products
and that run is reported separately.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from terrashift.models.siamese_unet import SiameseUNet
from scripts.inference.predict_safe_pair import BANDS, load_checkpoint, normalise


def _save(root, **overrides):
    model = SiameseUNet(in_channels=len(BANDS), base=8)
    payload = {
        "model_state_dict": model.state_dict(),
        "bands": list(BANDS),
        "mean": [0.1, 0.2, 0.3, 0.4],
        "std": [0.05, 0.05, 0.05, 0.05],
        "base": 8,
        "run_config": {"patch_size": 256},
    }
    payload.update(overrides)
    path = Path(root) / "checkpoint.pt"
    torch.save(payload, path)
    return path


class CheckpointContractTests(unittest.TestCase):
    def test_compatible_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as temp:
            path = _save(temp)
            checkpoint, mean, std = load_checkpoint(path, BANDS)
            self.assertEqual(tuple(checkpoint["bands"]), tuple(BANDS))
            self.assertEqual(mean.shape, (4, 1, 1))
            self.assertEqual(std.shape, (4, 1, 1))

    def test_missing_model_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = {"bands": list(BANDS), "mean": [0.1] * 4, "std": [0.1] * 4, "base": 8}
            path = Path(temp) / "bad.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "model_state_dict"):
                load_checkpoint(path, BANDS)

    def test_band_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = _save(temp, bands=["B04", "B03", "B02", "B08"])
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_checkpoint(path, BANDS)

    def test_non_positive_std_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = _save(temp, std=[0.05, 0.0, 0.05, 0.05])
            with self.assertRaisesRegex(ValueError, "not usable"):
                load_checkpoint(path, BANDS)


class NormalisationContractTests(unittest.TestCase):
    def test_normalise_matches_the_training_formula(self):
        rng = np.random.default_rng(0)
        stack = rng.random((4, 8, 8)).astype(np.float32) * 0.5
        mean = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).reshape(-1, 1, 1)
        std = np.array([0.05, 0.1, 0.15, 0.2], dtype=np.float32).reshape(-1, 1, 1)
        out = normalise(stack, mean, std)
        expected = (stack - mean) / std
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_non_finite_values_become_zero_not_nan(self):
        stack = np.full((4, 4, 4), np.nan, dtype=np.float32)
        stack[0, 0, 0] = np.inf
        mean = np.zeros((4, 1, 1), dtype=np.float32)
        std = np.ones((4, 1, 1), dtype=np.float32)
        out = normalise(stack, mean, std)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all(out == 0.0))


if __name__ == "__main__":
    unittest.main()
