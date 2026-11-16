"""Regression tests for unique-geographic-pixel statistics and metrics.

These tests build overlapping patch layouts whose exact per-pixel values are
known, then check that the weighted statistics and the stitched metrics agree
with a direct per-pixel computation, and that the patch-weighted version does
not. No imagery is read.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

from data.local_patches import local_training_statistics
from utils.unique_metrics import (UniqueValidationAccumulator, average_precision,
                                  coverage_1d, unique_validation_metrics,
                                  weighted_training_statistics)

CRS = "EPSG:32643"
ORIGIN = (500000.0, 3000000.0)


class _StubDataset:
    """Minimal LocalManualPatchDataset stand-in built from arrays."""

    def __init__(self, patches, size, split="train", bands=("B02",)):
        self.split = split
        self.bands = tuple(bands)
        self._size = size
        self._patches = patches
        self.rows = [
            {"row_off": p["row_off"], "col_off": p["col_off"],
             "height": size, "width": size} for p in patches
        ]

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, index):
        patch = self._patches[index]
        return {
            "image1": torch.from_numpy(patch["image1"].copy()),
            "image2": torch.from_numpy(patch["image2"].copy()),
            "mask": torch.from_numpy(patch["mask"].copy()),
            "valid_mask": torch.from_numpy(patch["valid_mask"].copy()),
        }


def _overlapping_layout(size=4, stride=2, grid=8):
    origins = list(range(0, grid - size + 1, stride))
    return [(r, c) for r in origins for c in origins]


def _field(grid, seed=0):
    """A smooth-free test field; a linear ramp would hide the weighting bias."""
    rng = np.random.default_rng(seed)
    return rng.random((grid, grid)).astype(np.float64)


def _build_dataset(grid=8, size=4, stride=2, n_bands=1, seed=0):
    field = _field(grid, seed)
    labels = (np.arange(grid * grid).reshape(grid, grid) % 2).astype(np.int64)
    patches = []
    for r0, c0 in _overlapping_layout(size, stride, grid):
        sub = field[r0:r0 + size, c0:c0 + size].astype(np.float32)
        image = np.repeat(sub[None, :, :], n_bands, axis=0).astype(np.float32)
        patches.append({
            "row_off": r0, "col_off": c0,
            "image1": image, "image2": image.copy(),
            "mask": labels[r0:r0 + size, c0:c0 + size].copy(),
            "valid_mask": np.ones((size, size), dtype=bool),
        })
    dataset = _StubDataset(patches, size, bands=tuple("B02" for _ in range(n_bands)))
    return dataset, field, labels


class CoverageTests(unittest.TestCase):
    def test_coverage_counts_overlapping_windows(self):
        counts = coverage_1d([0, 2, 4], 4, 8)
        np.testing.assert_array_equal(counts, [1, 1, 2, 2, 2, 2, 1, 1])

    def test_non_overlapping_windows_have_unit_coverage(self):
        counts = coverage_1d([0, 4], 4, 8)
        np.testing.assert_array_equal(counts, [1, 1, 1, 1, 1, 1, 1, 1])

    def test_origin_outside_the_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            coverage_1d([9], 4, 8)


class WeightedStatisticsTests(unittest.TestCase):
    def test_unique_mean_and_std_match_a_direct_computation(self):
        grid, size, stride = 8, 4, 2
        dataset, field, _ = _build_dataset(grid, size, stride)
        stats = weighted_training_statistics(dataset)

        # Date 1 and date 2 are identical, so the pooled statistic equals the
        # single-date statistic.
        expected_mean = field.mean()
        expected_std = field.std()

        # 9 patches of 4x4 slots over 64 unique pixels.
        self.assertEqual(stats["patch_pixels"], 9 * size * size)
        self.assertAlmostEqual(stats["unique_pixels"], float(grid * grid), places=9)
        self.assertAlmostEqual(stats["pooled_values"], float(2 * grid * grid), places=9)
        self.assertLess(stats["weighting_rounding_error"], 1e-9)
        self.assertAlmostEqual(float(stats["mean"][0]), expected_mean, places=5)
        self.assertAlmostEqual(float(stats["std"][0]), expected_std, places=5)

    def test_patch_weighted_mean_is_biased_and_the_fix_removes_the_bias(self):
        dataset, field, _ = _build_dataset(8, 4, 2)
        weighted = weighted_training_statistics(dataset)
        naive_mean, _ = local_training_statistics(dataset)
        # The naive computation counts interior pixels several times, so it
        # must differ from the unique-pixel mean. This is the defect the
        # weighted path removes.
        self.assertGreater(abs(float(naive_mean[0]) - float(weighted["mean"][0])), 1e-3)
        self.assertAlmostEqual(float(weighted["mean"][0]), field.mean(), places=5)

    def test_class_counts_are_unique_pixel_counts(self):
        grid, size, stride = 8, 4, 2
        dataset, _, labels = _build_dataset(grid, size, stride)
        stats = weighted_training_statistics(dataset)
        self.assertAlmostEqual(stats["positive"], float((labels == 1).sum()), places=9)
        self.assertAlmostEqual(stats["negative"], float((labels == 0).sum()), places=9)
        self.assertAlmostEqual(stats["positive"] + stats["negative"], float(grid * grid),
                               places=9)

    def test_statistics_reject_the_validation_split(self):
        dataset, _, _ = _build_dataset()
        dataset.split = "validation"
        with self.assertRaises(ValueError):
            weighted_training_statistics(dataset)


class AccumulatorTests(unittest.TestCase):
    def _label_raster(self, root, array):
        path = Path(root) / "labels.tif"
        with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                           width=array.shape[1], count=1, dtype="uint8",
                           crs=CRS, transform=Affine(10.0, 0.0, ORIGIN[0],
                                                     0.0, -10.0, ORIGIN[1])) as ds:
            ds.write(array.astype(np.uint8), 1)
        return path

    def test_overlapping_patches_score_each_pixel_once(self):
        with tempfile.TemporaryDirectory() as temp:
            grid, size, stride = 6, 4, 2
            labels = np.zeros((grid, grid), dtype=np.uint8)
            labels[0:3, 0:3] = 1
            path = self._label_raster(temp, labels)

            accumulator = UniqueValidationAccumulator("A", (grid, grid), path)
            constants = {(0, 0): 0.2, (0, 2): 0.4, (2, 0): 0.6, (2, 2): 0.8}
            for (r0, c0), value in constants.items():
                probs = np.full((size, size), value, dtype=np.float32)
                accumulator.add(r0, c0, probs, np.ones((size, size), dtype=bool))

            self.assertEqual(accumulator.covered_pixels, grid * grid)
            scores, truth = accumulator.finalize()
            self.assertEqual(scores.size, grid * grid)
            self.assertEqual(truth.size, grid * grid)

            # A pixel covered by all four patches gets the mean 0.5; a corner
            # pixel covered by one patch keeps that patch's value.
            scores_grid = scores.reshape(grid, grid)
            self.assertAlmostEqual(float(scores_grid[0, 0]), 0.2, places=6)
            self.assertAlmostEqual(float(scores_grid[0, 1]), 0.2, places=6)
            self.assertAlmostEqual(float(scores_grid[2, 2]), 0.5, places=6)
            self.assertAlmostEqual(float(scores_grid[3, 3]), 0.5, places=6)
            self.assertAlmostEqual(float(scores_grid[5, 5]), 0.8, places=6)
            np.testing.assert_array_equal(truth.reshape(grid, grid), labels)

    def test_naive_patch_counting_would_inflate_the_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            grid, size = 6, 4
            labels = np.ones((grid, grid), dtype=np.uint8)
            path = self._label_raster(temp, labels)
            accumulator = UniqueValidationAccumulator("A", (grid, grid), path)
            for (r0, c0) in ((0, 0), (0, 2), (2, 0), (2, 2)):
                accumulator.add(r0, c0, np.full((size, size), 0.9, np.float32),
                                np.ones((size, size), dtype=bool))
            scores, truth = accumulator.finalize()
            self.assertEqual(scores.size, grid * grid)
            self.assertEqual(4 * size * size, 64)   # the patch-weighted sample size
            self.assertLess(scores.size, 4 * size * size)

    def test_metrics_on_unique_pixels_match_a_hand_computation(self):
        scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7], dtype=np.float32)
        labels = np.array([0, 1, 1, 0, 0], dtype=np.int64)
        result = unique_validation_metrics(scores, labels, threshold=0.5)
        self.assertEqual(result["unique_valid_pixels"], 5)
        self.assertEqual(result["unique_positive_pixels"], 2)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]),
                         (2, 1, 0, 2))
        self.assertAlmostEqual(result["precision"], 2 / 3, places=9)
        self.assertAlmostEqual(result["recall"], 1.0, places=9)
        self.assertAlmostEqual(result["f1"], 2 * (2 / 3) / (1 + 2 / 3), places=9)
        self.assertAlmostEqual(result["ap"], average_precision(labels, scores), places=12)

    def test_accumulator_rejects_a_grid_over_budget(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            UniqueValidationAccumulator("A", (100, 100), "missing.tif", max_pixels=1000)

    def test_accumulator_rejects_a_patch_outside_the_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self._label_raster(temp, np.zeros((6, 6), np.uint8))
            accumulator = UniqueValidationAccumulator("A", (6, 6), path)
            with self.assertRaisesRegex(ValueError, "outside the validation grid"):
                accumulator.add(4, 4, np.zeros((4, 4), np.float32),
                                np.ones((4, 4), dtype=bool))


class _StubLoaderDataset(torch.utils.data.Dataset):
    """Torch dataset that emits the keys the stitched evaluator needs."""

    def __init__(self, grid, size, label_path, value=0.9):
        self.items = []
        for r0, c0 in ((0, 0), (0, 2), (2, 0), (2, 2)):
            self.items.append({
                "image1": torch.full((1, size, size), value),
                "image2": torch.full((1, size, size), value),
                "mask": torch.ones((size, size), dtype=torch.long),
                "valid_mask": torch.ones((size, size), dtype=torch.bool),
                "pair": "A", "row_off": r0, "col_off": c0,
                "label_path": str(label_path),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _ConstantModel(torch.nn.Module):
    def forward(self, a, b):
        return torch.zeros((a.shape[0], 1, a.shape[2], a.shape[3]))


class EvaluateUniqueIntegrationTests(unittest.TestCase):
    def test_data_loader_path_returns_unique_pixel_counts(self):
        from torch.utils.data import DataLoader
        from utils.unique_metrics import evaluate_unique

        with tempfile.TemporaryDirectory() as temp:
            grid, size = 6, 4
            labels = np.ones((grid, grid), dtype=np.uint8)
            path = Path(temp) / "labels.tif"
            with rasterio.open(path, "w", driver="GTiff", height=grid, width=grid,
                               count=1, dtype="uint8", crs=CRS,
                               transform=Affine(10.0, 0.0, ORIGIN[0],
                                                0.0, -10.0, ORIGIN[1])) as ds:
                ds.write(labels, 1)
            dataset = _StubLoaderDataset(grid, size, path)
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            result = evaluate_unique(_ConstantModel(), loader, torch.device("cpu"),
                                     threshold=0.5)
            # 4 patches x 16 px = 64 patch slots, but only 36 unique pixels.
            self.assertEqual(result["validation_patches"], 4)
            self.assertEqual(result["unique_valid_pixels"], grid * grid)
            self.assertEqual(result["unique_positive_pixels"], grid * grid)
            self.assertLess(result["unique_valid_pixels"], 4 * size * size)
            self.assertAlmostEqual(result["recall"], 1.0, places=9)


class EmptyValidationTests(unittest.TestCase):
    def test_no_reviewed_pixels_reports_nan_metrics(self):
        result = unique_validation_metrics(np.empty(0), np.empty(0))
        self.assertEqual(result["unique_valid_pixels"], 0)
        self.assertTrue(np.isnan(result["ap"]))
        self.assertTrue(np.isnan(result["f1"]))


if __name__ == "__main__":
    unittest.main()
