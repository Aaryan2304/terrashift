"""Tests for the inference-compatible validation window selection.

Synthetic grids only. The reference implementation of the tiling is written inline
here with ``terrashift.inference._window_origins`` so a divergence in the new wrappers
shows up as a failing assertion rather than being mirrored by both sides.
"""

from __future__ import annotations

import unittest

import numpy as np

from terrashift.data.validation_windows import (IGNORE_VALUE, core_slice, inference_strips,
                                     origins_by_pixel, select_windows, strip_origins,
                                     validation_target, window_grid)
from terrashift.inference.sliding import _window_origins

PATCH = 32
STRIDE = 16
BLOCK = 64
HALO = 32
HEIGHT = 200
WIDTH = 128


def _reference_origins(height, width, patch, stride, block, halo):
    """The inference tiling, written straight from the canonical tiler."""
    grid = {}
    for core_start in range(0, height, block):
        core_stop = min(core_start + block, height)
        read_start = max(0, core_start - halo)
        read_stop = min(height, core_stop + halo)
        rows = [read_start + o for o in _window_origins(read_stop - read_start, patch, stride)]
        cols = list(_window_origins(width, patch, stride))
        for row0 in rows:
            for col0 in cols:
                first = max(row0, core_start)
                last = min(row0 + patch, core_stop)
                for row in range(first, last):
                    for col in range(col0, min(col0 + patch, width)):
                        grid.setdefault((row, col), set()).add((row0, col0))
    return grid


def _grid(labels_value=IGNORE_VALUE):
    labels = np.full((HEIGHT, WIDTH), labels_value, dtype=np.uint8)
    partition = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    partition[:, 0:40] = 1        # train
    partition[:, 40:48] = 0       # guard
    partition[:, 48:96] = 2       # validation
    partition[:, 96:128] = 3      # test
    return labels, partition


class TilingFidelityTests(unittest.TestCase):
    def test_per_pixel_origin_sets_match_the_inference_tiler(self):
        labels, partition = _grid()
        labels[60:70, 60:70] = 1              # reviewed validation pixels
        target = validation_target(labels, partition)
        windows = select_windows(target, PATCH, STRIDE, BLOCK, HALO)
        mine = origins_by_pixel(windows, PATCH, target.shape)
        reference = _reference_origins(HEIGHT, WIDTH, PATCH, STRIDE, BLOCK, HALO)
        rows, cols = np.nonzero(target != IGNORE_VALUE)
        self.assertGreater(rows.size, 0)
        for row, col in zip(rows, cols):
            with self.subTest(pixel=(int(row), int(col))):
                self.assertEqual(mine.get((int(row), int(col)), set()),
                                 reference[(int(row), int(col))])

    def test_every_window_origin_comes_from_the_canonical_tiler(self):
        labels, partition = _grid()
        labels[::5, ::5] = 0                  # plenty of reviewed validation pixels
        target = validation_target(labels, partition)
        windows = select_windows(target, PATCH, STRIDE, BLOCK, HALO)
        allowed = {}
        for core_start, core_stop, read_start, read_stop in inference_strips(HEIGHT, BLOCK, HALO):
            allowed[core_start] = (
                {read_start + o for o in _window_origins(read_stop - read_start, PATCH, STRIDE)},
                set(_window_origins(WIDTH, PATCH, STRIDE)))
        for window in windows:
            rows, cols = allowed[window["core_start"]]
            with self.subTest(window=(window["row0"], window["col0"])):
                self.assertIn(window["row0"], rows)
                self.assertIn(window["col0"], cols)

    def test_final_window_is_flushed_to_the_edge(self):
        # width 128, patch 32, stride 16 -> origins 0..96, and 96 is exactly the edge.
        self.assertEqual(list(_window_origins(WIDTH, PATCH, STRIDE)), [0, 16, 32, 48, 64, 80, 96])
        # A width that is not a multiple of the stride must still end at the edge.
        origins = _window_origins(100, PATCH, STRIDE)
        self.assertEqual(origins[-1], 100 - PATCH)
        self.assertEqual(strip_origins(0, 100, PATCH, STRIDE)[-1], 100 - PATCH)

    def test_shorter_than_a_patch_gives_one_origin(self):
        self.assertEqual(strip_origins(0, 20, PATCH, STRIDE), [0])
        self.assertEqual(_window_origins(20, PATCH, STRIDE), [0])

    def test_strip_halo_windows_are_clipped_to_the_core(self):
        # A window from the halo extends outside its core; only core rows count.
        window = {"row0": 40, "col0": 0, "core_start": 64, "core_stop": 96}
        offset, count = core_slice(window, PATCH)
        self.assertEqual((offset, count), (24, 8))
        self.assertEqual(window["row0"] + offset, 64)
        self.assertEqual(window["row0"] + offset + count, 72)

    def test_window_outside_its_core_is_rejected(self):
        window = {"row0": 0, "col0": 0, "core_start": 64, "core_stop": 96}
        with self.assertRaisesRegex(ValueError, "does not overlap its core"):
            core_slice(window, PATCH)


class TargetTests(unittest.TestCase):
    def test_train_and_test_labels_are_ignored(self):
        labels, partition = _grid()
        labels[:, :] = 1                      # a review everywhere
        target = validation_target(labels, partition)
        self.assertTrue(np.all(target[:, 48:96] == 1))
        self.assertTrue(np.all(target[:, 0:48] == IGNORE_VALUE))
        self.assertTrue(np.all(target[:, 96:] == IGNORE_VALUE))

    def test_unreviewed_pixels_are_ignored(self):
        labels, partition = _grid()
        labels[:, :] = 0
        labels[10:20, 60:70] = IGNORE_VALUE
        target = validation_target(labels, partition)
        self.assertTrue(np.all(target[10:20, 60:70] == IGNORE_VALUE))
        self.assertTrue(np.all(target[0:10, 60:70] == 0))

    def test_non_binary_reviewed_label_is_rejected(self):
        labels, partition = _grid()
        labels[5, 60] = 7
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            validation_target(labels, partition)

    def test_shape_mismatch_is_rejected(self):
        labels, partition = _grid()
        with self.assertRaisesRegex(ValueError, "share a shape"):
            validation_target(labels, partition[:, :-1])


class SelectionTests(unittest.TestCase):
    def test_windows_may_cross_a_split_boundary(self):
        labels, partition = _grid()
        labels[0:HEIGHT, 48:96] = 1
        target = validation_target(labels, partition)
        windows = select_windows(target, PATCH, STRIDE, BLOCK, HALO)
        crossing = [w for w in windows if w["col0"] < 48 < w["col0"] + PATCH]
        self.assertTrue(crossing, "no window spans the train/validation boundary")

    def test_window_with_no_reviewed_validation_pixels_is_excluded(self):
        labels, partition = _grid()
        labels[0:10, 50:60] = 1               # one small reviewed island
        target = validation_target(labels, partition)
        windows = select_windows(target, PATCH, STRIDE, BLOCK, HALO)
        self.assertTrue(windows)
        for window in windows:
            self.assertGreater(window["reviewed_validation_px"], 0)

    def test_train_only_review_reaches_no_validation_window(self):
        labels, partition = _grid()
        labels[0:HEIGHT, 0:40] = 1            # reviewed, but all in train
        target = validation_target(labels, partition)
        self.assertEqual(select_windows(target, PATCH, STRIDE, BLOCK, HALO), [])

    def test_test_only_review_reaches_no_validation_window(self):
        labels, partition = _grid()
        labels[0:HEIGHT, 96:] = 1             # reviewed, but all in test
        target = validation_target(labels, partition)
        self.assertEqual(select_windows(target, PATCH, STRIDE, BLOCK, HALO), [])

    def test_min_reviewed_is_enforced(self):
        labels, partition = _grid()
        labels[0:2, 50:52] = 1                # four reviewed pixels
        target = validation_target(labels, partition)
        one = select_windows(target, PATCH, STRIDE, BLOCK, HALO, min_reviewed=1)
        many = select_windows(target, PATCH, STRIDE, BLOCK, HALO, min_reviewed=5)
        self.assertTrue(one)
        self.assertEqual(many, [])


class AccumulationEquivalenceTests(unittest.TestCase):
    def test_validation_and_inference_see_the_same_windows_per_pixel(self):
        # Same grid, same tiler: the selection must not drop or add a window for any
        # reviewed validation pixel.
        labels, partition = _grid()
        labels[50:70, 55:85] = 1
        target = validation_target(labels, partition)
        windows = select_windows(target, PATCH, STRIDE, BLOCK, HALO)
        mine = origins_by_pixel(windows, PATCH, target.shape)
        reference = _reference_origins(HEIGHT, WIDTH, PATCH, STRIDE, BLOCK, HALO)
        rows, cols = np.nonzero(target != IGNORE_VALUE)
        self.assertTrue(rows.size > 0)
        for row, col in zip(rows, cols):
            self.assertEqual(len(mine.get((int(row), int(col)), set())),
                             len(reference[(int(row), int(col))]))


if __name__ == "__main__":
    unittest.main()
