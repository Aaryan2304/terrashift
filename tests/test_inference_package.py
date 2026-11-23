"""Focused tests for the migrated inference package.

Each class covers a behaviour that the existing suite only reached indirectly: the sliding
window was exercised through the CLI, the export dummy and calibration reader had no direct
callers in the tests, and checkpoint loading was previously asserted through ``quantize``.

Synthetic tensors and small sizes only: no SAFE products, no ONNX artefacts written, no CUDA.
"""

from __future__ import annotations

import pathlib
import unittest

import numpy as np
import torch

from terrashift.inference.checkpoints import Wrapper, load_model
from terrashift.inference.export import PATCH_SIZE, export_dummy
from terrashift.inference.quantization import CalibrationReader
from terrashift.inference.sliding import _window_origins, sliding_window_predict

REPO = pathlib.Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"
BANDS = ("B02", "B03", "B04", "B08")


class _ConstantLogitModel(torch.nn.Module):
    """Returns a fixed logit everywhere, so aggregation weights can be checked exactly."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, a, b):
        return torch.full((a.shape[0], 1, a.shape[2], a.shape[3]), self.value)


class _CopyInputModel(torch.nn.Module):
    """Returns the first date's first band as the logit, to expose window placement."""

    def forward(self, a, b):
        return a[:, :1]


class SlidingWindowTests(unittest.TestCase):
    PATCH, STRIDE = 32, 16

    def _pair(self, height, width):
        g = torch.Generator().manual_seed(3)
        return (torch.randn(4, height, width, generator=g),
                torch.randn(4, height, width, generator=g))

    def test_output_shape_matches_a_non_divisible_input(self):
        x1, x2 = self._pair(70, 54)          # neither dimension divides evenly by patch/stride
        prob, mask = sliding_window_predict(_ConstantLogitModel(0.0), x1, x2,
                                            self.PATCH, self.STRIDE, torch.device("cpu"))
        self.assertEqual(prob.shape, (70, 54))
        # The in-memory map is float64; predict_safe_pair casts to float32 when writing
        # the GeoTIFF. Assert the actual contract rather than the documented one.
        self.assertTrue(np.issubdtype(prob.dtype, np.floating))
        self.assertIsNone(mask)

    def test_constant_logits_give_the_exact_mean_probability(self):
        x1, x2 = self._pair(40, 40)
        prob, _ = sliding_window_predict(_ConstantLogitModel(0.0), x1, x2,
                                        self.PATCH, self.STRIDE, torch.device("cpu"))
        # sigmoid(0) = 0.5, and overlap averaging must not change a constant field
        np.testing.assert_allclose(prob, 0.5, rtol=0, atol=1e-7)

    def test_every_pixel_is_covered_including_the_edges(self):
        x1, x2 = self._pair(33, 41)
        prob, _ = sliding_window_predict(_ConstantLogitModel(2.0), x1, x2,
                                        self.PATCH, self.STRIDE, torch.device("cpu"))
        self.assertTrue(np.isfinite(prob).all(), "uncovered pixels would be non-finite")
        np.testing.assert_allclose(prob, float(torch.sigmoid(torch.tensor(2.0))),
                                   rtol=0, atol=1e-6)

    def test_result_is_deterministic(self):
        x1, x2 = self._pair(50, 50)
        model = _CopyInputModel()
        first, _ = sliding_window_predict(model, x1, x2, self.PATCH, self.STRIDE,
                                          torch.device("cpu"))
        second, _ = sliding_window_predict(model, x1, x2, self.PATCH, self.STRIDE,
                                           torch.device("cpu"))
        np.testing.assert_array_equal(first, second)

    def test_batch_size_does_not_change_the_map(self):
        x1, x2 = self._pair(64, 48)
        model = _CopyInputModel()
        single, _ = sliding_window_predict(model, x1, x2, self.PATCH, self.STRIDE,
                                           torch.device("cpu"), batch_size=1)
        batched, _ = sliding_window_predict(model, x1, x2, self.PATCH, self.STRIDE,
                                            torch.device("cpu"), batch_size=5)
        np.testing.assert_allclose(single, batched, rtol=0, atol=1e-6)

    def test_threshold_returns_a_boolean_mask(self):
        x1, x2 = self._pair(40, 40)
        prob, mask = sliding_window_predict(_ConstantLogitModel(0.0), x1, x2,
                                            self.PATCH, self.STRIDE, torch.device("cpu"),
                                            threshold=0.5)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.dtype, np.bool_)
        np.testing.assert_array_equal(mask, prob >= 0.5)

    def test_window_origins_cover_the_axis_and_respect_the_stride(self):
        origins = _window_origins(70, 32, 16)
        self.assertEqual(origins[0], 0)
        self.assertEqual(origins[-1], 70 - 32, "the last window must end flush with the axis")
        differences = [b - a for a, b in zip(origins, origins[1:])]
        self.assertTrue(all(d == 16 for d in differences[:-1]),
                        "intermediate steps must equal the stride: %s" % differences)
        # the last window is clamped so it ends flush with the axis, so its step may be shorter
        self.assertLessEqual(differences[-1], 16)


class ExportDummyTests(unittest.TestCase):
    def test_dummy_has_the_exported_contract(self):
        a, b = export_dummy(BANDS)
        self.assertEqual(tuple(a.shape), (1, len(BANDS), PATCH_SIZE, PATCH_SIZE))
        self.assertEqual(tuple(b.shape), (1, len(BANDS), PATCH_SIZE, PATCH_SIZE))
        self.assertEqual(a.dtype, torch.float32)
        self.assertEqual(b.dtype, torch.float32)

    def test_size_is_configurable(self):
        a, b = export_dummy(BANDS, size=64)
        self.assertEqual(tuple(a.shape), (1, 4, 64, 64))
        self.assertEqual(tuple(b.shape), (1, 4, 64, 64))


class CalibrationReaderTests(unittest.TestCase):
    """Batching, key ordering and exhaustion, which nothing else asserts."""

    class _Loader:
        def __init__(self, samples):
            self.samples = samples

        def __iter__(self):
            return iter(self.samples)

    def _reader(self, count=4, limit=2):
        # the reader calls .numpy() on each value, so the loader must yield tensors
        samples = [{"image1": torch.full((1, 4, 8, 8), float(i)),
                    "image2": torch.full((1, 4, 8, 8), float(i))}
                   for i in range(count)]
        return CalibrationReader(self._Loader(samples), ["image1", "image2"], limit)

    def test_limit_caps_the_number_of_batches(self):
        reader = self._reader(count=6, limit=3)
        self.assertLessEqual(len(reader.batches), 3)

    def test_batches_carry_both_input_names_in_order(self):
        batch = self._reader().get_next()
        self.assertIsInstance(batch, dict)
        self.assertEqual(list(batch.keys()), ["image1", "image2"])

    def test_exhaustion_returns_none(self):
        reader = self._reader(count=2, limit=2)
        while reader.get_next() is not None:
            pass
        self.assertIsNone(reader.get_next())

    def test_rewind_restarts_the_sequence(self):
        reader = self._reader(count=2, limit=2)
        first = reader.get_next()
        while reader.get_next() is not None:
            pass
        reader.rewind()
        again = reader.get_next()
        np.testing.assert_array_equal(first["image1"], again["image1"])


class CheckpointLoadingTests(unittest.TestCase):
    def setUp(self):
        if not CHECKPOINT.is_file():
            self.skipTest("selected checkpoint not present")

    def test_load_model_returns_wrapper_bands_and_payload(self):
        wrapper, bands, payload = load_model(str(CHECKPOINT), "cpu")
        self.assertIsInstance(wrapper, Wrapper)
        self.assertEqual(tuple(bands), BANDS)
        self.assertIn("model_state_dict", payload)
        self.assertEqual(int(payload["base"]), 16)

    def test_wrapper_applies_sigmoid_and_stays_in_the_unit_interval(self):
        wrapper, _, _ = load_model(str(CHECKPOINT), "cpu")
        wrapper.eval()
        x = torch.zeros(1, 4, 64, 64)
        with torch.no_grad():
            probability = wrapper(x, x)
        self.assertEqual(tuple(probability.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(probability).all())
        self.assertGreaterEqual(float(probability.min()), 0.0)
        self.assertLessEqual(float(probability.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
