"""End-to-end tests for the inference-lattice validation path the trainer runs.

Synthetic imagery and synthetic split masks throughout. Nothing here loads a real
scene, and no model is trained: the "model" returns constant logits so the expected
probabilities can be written down by hand.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from terrashift.data.validation_windows import (IGNORE_VALUE, ValidationLatticeDataset,
                                     inference_strips, read_validation_target,
                                     select_windows, validation_stripe,
                                     validation_target, window_grid)
from terrashift.inference.sliding import _window_origins
from terrashift.evaluation.lattice import LatticeAccumulator, evaluate_lattice

PATCH = 16
STRIDE = 8
BLOCK = 32
HALO = 16
HEIGHT = 96
WIDTH = 96


def synthetic_split(validation_columns=(32, 64), reviewed=0.25, seed=7):
    """Labels and partition with train / guard / validation / test bands."""
    rng = np.random.default_rng(seed)
    labels = np.full((HEIGHT, WIDTH), IGNORE_VALUE, dtype=np.uint8)
    partition = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    partition[:, 0:24] = 1
    partition[:, 24:32] = 0
    partition[:, 32:64] = 2
    partition[:, 64:] = 3
    reviewed_mask = rng.random((HEIGHT, WIDTH)) < reviewed
    binary = rng.integers(0, 2, (HEIGHT, WIDTH))
    labels[reviewed_mask & (partition == 1)] = binary[reviewed_mask & (partition == 1)]
    labels[reviewed_mask & (partition == 2)] = binary[reviewed_mask & (partition == 2)]
    # Distinctive non-binary values in the test band, so a leak would be visible.
    labels[reviewed_mask & (partition == 3)] = rng.integers(2, 9, (HEIGHT, WIDTH))[
        reviewed_mask & (partition == 3)]
    return labels, partition


def write_raster(path, array, dtype="uint8"):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                       width=array.shape[1], count=1, dtype=dtype,
                       transform=from_origin(0, 0, 10, 10), crs="EPSG:32643") as dst:
        dst.write(array.astype(dtype), 1)


def synthetic_imagery(row_off, col_off, size, value=1.0):
    image1 = np.full((4, size, size), value, dtype=np.float32)
    image2 = np.full((4, size, size), value + 1.0, dtype=np.float32)
    # A spatial gradient so a constant-response model still produces varied output
    # only through the target, not through the image.
    return image1, image2


class ConstantModel:
    """Returns a fixed logit everywhere, ignoring the input."""

    def __init__(self, logit=0.0):
        self.logit = float(logit)
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, x1, x2):
        import torch
        self.calls += 1
        return torch.full((x1.shape[0], 1, x1.shape[2], x1.shape[3]), self.logit,
                          dtype=torch.float32, device=x1.device)


class TargetIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lattice_")
        self.labels_path = os.path.join(self.tmp, "labels.tif")
        self.partition_path = os.path.join(self.tmp, "partition.tif")
        self.labels, self.partition = synthetic_split()
        write_raster(self.labels_path, self.labels)
        write_raster(self.partition_path, self.partition)

    def test_target_is_validation_only(self):
        target, col0, tile_width = read_validation_target(self.labels_path,
                                                         self.partition_path)
        self.assertEqual((col0, tile_width), (32, WIDTH))
        expected = validation_target(self.labels, self.partition)[:, 32:64]
        np.testing.assert_array_equal(target, expected)
        self.assertTrue(np.all((target == 0) | (target == 1) | (target == IGNORE_VALUE)))
        self.assertGreater(int((target != IGNORE_VALUE).sum()), 0)

    def test_test_columns_are_never_read(self):
        seen = []
        real_open = rasterio.open

        class Recorder:
            def __init__(self, path, *args, **kwargs):
                self._handle = real_open(path, *args, **kwargs)

            def read(self, *args, **kwargs):
                window = kwargs.get("window", args[1] if len(args) > 1 else None)
                seen.append(window)
                return self._handle.read(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._handle, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._handle.close()

        rasterio.open = Recorder
        try:
            read_validation_target(self.labels_path, self.partition_path)
        finally:
            rasterio.open = real_open
        label_windows = [w for w in seen if w is not None]
        self.assertTrue(label_windows, "no windowed read was recorded")
        for window in label_windows:
            self.assertGreaterEqual(window.col_off, 32)
            self.assertLessEqual(window.col_off + window.width, 64)

    def test_changing_test_values_cannot_change_the_target(self):
        target_a, _, _ = read_validation_target(self.labels_path, self.partition_path)
        altered = self.labels.copy()
        altered[self.partition == 3] = 3          # test pixels renumbered
        altered[(self.partition == 3) & (altered == IGNORE_VALUE)] = 0
        write_raster(self.labels_path, altered)
        target_b, _, _ = read_validation_target(self.labels_path, self.partition_path)
        np.testing.assert_array_equal(target_a, target_b)

    def test_changing_test_values_cannot_change_validation_metrics(self):
        metrics = []
        for test_value in (7, 3):
            labels = self.labels.copy()
            test_pixels = self.partition == 3
            labels[test_pixels] = test_value
            write_raster(self.labels_path, labels)
            target, col0, tile_width = read_validation_target(self.labels_path,
                                                              self.partition_path)
            dataset = ValidationLatticeDataset(
                ["X"], target, col0, WIDTH, PATCH, STRIDE,
                image_loader=lambda r, c, s: synthetic_imagery(r, c, s),
                block=BLOCK, halo=HALO)
            import torch
            samples = [dataset[i] for i in range(len(dataset))]
            batch = {key: torch.stack([sample[key] for sample in samples])
                     for key in ("image1", "image2", "mask", "valid_mask")}
            batch.update({key: [sample[key] for sample in samples]
                          for key in ("row_off", "col_off", "core_offset", "core_count")})
            metrics.append(evaluate_lattice(
                ConstantModel(0.0), [batch], torch.device("cpu"),
                threshold=0.5, target=target, col_offset=col0))
        self.assertEqual(metrics[0]["unique_valid_pixels"], metrics[1]["unique_valid_pixels"])
        self.assertAlmostEqual(metrics[0]["ap"], metrics[1]["ap"], places=6)
        self.assertEqual(metrics[0]["tp"], metrics[1]["tp"])
        self.assertEqual(metrics[0]["fp"], metrics[1]["fp"])


class LatticeWiringTests(unittest.TestCase):
    def setUp(self):
        labels, partition = synthetic_split()
        self.target = validation_target(labels, partition)[:, 32:64]
        self.dataset = ValidationLatticeDataset(
            ["X"], self.target, 32, WIDTH, PATCH, STRIDE,
            image_loader=synthetic_imagery, block=BLOCK, halo=HALO)

    def test_windows_are_the_canonical_inference_origins(self):
        expected = set()
        for core_start, core_stop, read_start, read_stop in inference_strips(
                HEIGHT, BLOCK, HALO):
            for row0, col0 in window_grid(read_start, read_stop, WIDTH, PATCH, STRIDE):
                first, last = max(row0, core_start), min(row0 + PATCH, core_stop)
                if last <= first:
                    continue
                left = max(col0, 32)
                right = min(col0 + PATCH, 64)
                if right <= left:
                    continue
                if (self.target[first:last, left - 32:right - 32] != IGNORE_VALUE).any():
                    expected.add((row0, col0))
        mine = {(window["row0"], window["col0"]) for window in self.dataset.windows}
        self.assertEqual(mine, expected)

    def test_every_window_has_at_least_one_reviewed_pixel(self):
        for window in self.dataset.windows:
            self.assertGreater(window["reviewed_validation_px"], 0)

    def test_samples_carry_core_range_and_validation_only_mask(self):
        sample = self.dataset[0]
        self.assertIn("core_offset", sample)
        self.assertIn("core_count", sample)
        mask = np.asarray(sample["mask"])
        valid = np.asarray(sample["valid_mask"])
        self.assertFalse(np.any((mask != 0) & (mask != 1)))
        self.assertTrue(np.all((mask[valid] == 0) | (mask[valid] == 1)))
        self.assertTrue(np.all(mask[~valid] == 0))
        self.assertGreater(int(valid.sum()), 0)

    def test_accumulator_averages_overlapping_windows_equally(self):
        acc = LatticeAccumulator((8, 8), 0)
        probs_a = np.full((4, 4), 0.2)
        probs_b = np.full((4, 4), 0.4)
        acc.add(0, 0, probs_a, np.ones((4, 4), bool))
        acc.add(0, 2, probs_b, np.ones((4, 4), bool))
        self.assertEqual(int(acc.count[0, 2]), 2)
        self.assertAlmostEqual(acc.total[0, 2] / acc.count[0, 2], 0.3, places=6)
        self.assertEqual(int(acc.count[0, 0]), 1)
        self.assertAlmostEqual(acc.total[0, 0] / acc.count[0, 0], 0.2, places=6)

    def test_core_clipping_matches_inference(self):
        acc = LatticeAccumulator((8, 4), 0)
        probs = np.full((4, 4), 0.9)
        acc.add(2, 0, probs, np.ones((4, 4), bool), core_offset=2, core_count=2)
        self.assertTrue(np.all(acc.count[0:4] == 0), "rows before the core were counted")
        self.assertTrue(np.all(acc.count[4:6] == 1))
        self.assertTrue(np.all(acc.count[6:] == 0), "rows after the core were counted")

    def test_metrics_count_each_pixel_once(self):
        import torch
        samples = [self.dataset[i] for i in range(len(self.dataset))]
        batch = {key: torch.stack([s[key] for s in samples])
                 for key in ("image1", "image2", "mask", "valid_mask")}
        batch.update({key: [s[key] for s in samples]
                      for key in ("row_off", "col_off", "core_offset", "core_count")})
        result = evaluate_lattice(ConstantModel(0.0), [batch], torch.device("cpu"),
                                  threshold=0.5, target=self.target, col_offset=32)
        reviewed = int((self.target != IGNORE_VALUE).sum())
        self.assertEqual(result["unique_valid_pixels"], reviewed)
        self.assertLessEqual(result["unique_valid_pixels"], self.target.size)
        self.assertGreaterEqual(result["accumulated_pixels"], reviewed)
        # Constant 0.5 probability: every pixel ties at the threshold, so the
        # positive count must equal the reviewed positive count.
        self.assertEqual(result["unique_positive_pixels"],
                         int((self.target == 1).sum()))

    def test_edge_padding_and_final_window(self):
        # A grid narrower than a patch still yields one window per strip.
        small = np.full((HEIGHT, 8), IGNORE_VALUE, dtype=np.uint8)
        small[0:8, 2:6] = 1
        dataset = ValidationLatticeDataset(
            ["X"], small, 0, 8, PATCH, STRIDE, image_loader=synthetic_imagery,
            block=BLOCK, halo=HALO)
        self.assertTrue(dataset.windows)
        for window in dataset.windows:
            self.assertEqual(window["col0"], 0)


class ImageCoordinateTests(unittest.TestCase):
    def test_image_loader_receives_tile_coordinates(self):
        labels, partition = synthetic_split()
        target = validation_target(labels, partition)[:, 32:64]
        calls = []

        def recording_loader(row_off, col_off, size):
            calls.append((row_off, col_off))
            return synthetic_imagery(row_off, col_off, size)

        dataset = ValidationLatticeDataset(
            ["X"], target, 32, WIDTH, PATCH, STRIDE,
            image_loader=recording_loader, block=BLOCK, halo=HALO)
        for index in range(len(dataset)):
            dataset[index]
        self.assertTrue(calls)
        for row_off, col_off in calls:
            with self.subTest(call=(row_off, col_off)):
                # Tile columns, never stripe-relative ones.
                self.assertGreaterEqual(col_off, 0)
                self.assertLessEqual(col_off + PATCH, WIDTH)
                self.assertIn(col_off, [w["col0"] for w in dataset.windows])


class StatisticsTests(unittest.TestCase):
    def test_train_statistics_divide_by_pixels_not_channel_pixels(self):
        """A shared counter across channels makes every mean too small by 4x."""
        import numpy as np
        from scripts.inference.validation_dry_run import _mean_std_from_channels

        planes = [np.full((10, 10), value, dtype=np.float32) for value in (1.0, 2.0, 3.0, 4.0)]
        mean, std = _mean_std_from_channels(planes)
        np.testing.assert_allclose(mean, [1.0, 2.0, 3.0, 4.0], rtol=1e-6)
        np.testing.assert_allclose(std, [0.0, 0.0, 0.0, 0.0], atol=1e-6)


class TrainerPathTests(unittest.TestCase):
    def test_trainer_uses_the_lattice_and_not_the_old_index(self):
        import inspect
        import train_local_manual

        source = inspect.getsource(train_local_manual)
        self.assertIn("evaluate_lattice_all", source)
        self.assertIn("ValidationLatticeDataset", source)
        self.assertIn("read_validation_target", source)
        self.assertNotIn("evaluate_unique", source)
        self.assertNotIn('LocalManualPatchDataset(\n        index_path, "validation"', source)

    def test_training_split_still_comes_from_index(self):
        import inspect
        import train_local_manual

        source = inspect.getsource(train_local_manual)
        self.assertIn('LocalManualPatchDataset(\n        index_path, "train"', source)
        self.assertNotIn('"validation", manifest_path', source)


if __name__ == "__main__":
    unittest.main()
