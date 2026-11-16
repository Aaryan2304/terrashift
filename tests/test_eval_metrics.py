"""Tests for the object-level evaluation logic in eval.py."""

from __future__ import annotations

import unittest

import numpy as np

from eval import component_table, evaluate_patch


def _mask(shape=(16, 16), boxes=()):
    mask = np.zeros(shape, dtype=bool)
    for (r0, r1, c0, c1) in boxes:
        mask[r0:r1, c0:c1] = True
    return mask


class ComponentTableTests(unittest.TestCase):
    def test_empty_mask_has_no_components(self):
        labels, areas = component_table(np.zeros((8, 8), dtype=bool))
        self.assertEqual(areas.size, 0)
        self.assertEqual(int(labels.max()), 0)

    def test_two_separate_blocks_are_two_components(self):
        _, areas = component_table(_mask(boxes=[(0, 3, 0, 3), (8, 10, 8, 10)]))
        self.assertEqual(sorted(areas.tolist()), [4, 9])


class ObjectMatchingTests(unittest.TestCase):
    def test_identical_masks_match_perfectly(self):
        truth = _mask(boxes=[(2, 6, 2, 6), (10, 13, 10, 13)])
        result = evaluate_patch(truth.copy(), truth.copy())
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)
        self.assertTrue(all(abs(iou - 1.0) < 1e-9 for iou in result["ious"]))

    def test_shifted_prediction_matches_nothing_at_half_iou(self):
        truth = _mask(boxes=[(2, 6, 2, 6)])
        pred = _mask(boxes=[(2, 6, 4, 8)])   # half overlap -> IoU 8/24
        result = evaluate_patch(pred, truth)
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)

    def test_fragmented_prediction_counts_extra_false_positives(self):
        truth = _mask(boxes=[(2, 8, 2, 8)])
        pred = _mask(boxes=[(2, 8, 2, 4), (2, 8, 5, 8)])
        result = evaluate_patch(pred, truth)
        # One fragment can match the whole truth; the other is a false positive.
        self.assertEqual(result["tp"] + result["fn"], 1)
        self.assertGreaterEqual(result["fp"], 1)

    def test_missing_component_is_a_false_negative(self):
        truth = _mask(boxes=[(2, 5, 2, 5), (10, 14, 10, 14)])
        pred = _mask(boxes=[(2, 5, 2, 5)])
        result = evaluate_patch(pred, truth)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["fp"], 0)


class SizeBinTests(unittest.TestCase):
    def test_components_are_attributed_to_area_bins(self):
        truth = _mask(shape=(32, 32), boxes=[(0, 2, 0, 2), (4, 20, 4, 20)])  # 4 px and 256 px
        result = evaluate_patch(truth.copy(), truth.copy())
        self.assertIn("4_16_px", result["size_bins"])
        self.assertIn("256_plus_px", result["size_bins"])
        self.assertEqual(result["size_bins"]["4_16_px"]["components"], 1)
        self.assertEqual(result["size_bins"]["4_16_px"]["matched"], 1)
        self.assertEqual(result["size_bins"]["256_plus_px"]["components"], 1)

    def test_unmatched_small_component_lowers_its_bin_recall(self):
        truth = _mask(boxes=[(0, 2, 0, 2), (10, 14, 10, 14)])
        pred = _mask(boxes=[(10, 14, 10, 14)])
        result = evaluate_patch(pred, truth)
        self.assertEqual(result["size_bins"]["4_16_px"]["components"], 1)
        self.assertEqual(result["size_bins"]["4_16_px"]["matched"], 0)


if __name__ == "__main__":
    unittest.main()
