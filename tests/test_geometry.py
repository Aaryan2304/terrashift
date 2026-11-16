"""Tests for the detection geometry: area range and skeleton length."""

from __future__ import annotations

import unittest

import numpy as np
from rasterio.transform import Affine

from utils.geometry import (boundary_area_range, clean_binary_mask, components,
                            pixel_area, skeleton_length)


def _mask(shape=(20, 20), boxes=()):
    mask = np.zeros(shape, dtype=bool)
    for (r0, r1, c0, c1) in boxes:
        mask[r0:r1, c0:c1] = True
    return mask


class ComponentTests(unittest.TestCase):
    def test_diagonal_touching_pixels_are_one_component(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        _, areas = components(mask)
        self.assertEqual(areas.tolist(), [2])


class AreaRangeTests(unittest.TestCase):
    def test_range_brackets_the_nominal_area(self):
        mask = _mask(boxes=[(5, 10, 5, 10)])   # 5x5 = 25 px
        result = boundary_area_range(mask, 100.0)
        self.assertEqual(result["nominal_px"], 25)
        self.assertEqual(result["nominal_m2"], 2500)
        self.assertLess(result["min_m2"], result["nominal_m2"])
        self.assertGreater(result["max_m2"], result["nominal_m2"])
        # One pixel of erosion and dilation on a 5x5 square is exact.
        self.assertEqual(result["eroded_px"], 9)
        self.assertEqual(result["dilated_px"], 49)

    def test_single_pixel_erodes_to_nothing(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        result = boundary_area_range(mask, 100.0)
        self.assertEqual(result["eroded_px"], 0)
        self.assertEqual(result["min_m2"], 0)

    def test_pixel_area_reads_the_transform(self):
        self.assertAlmostEqual(pixel_area(Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)), 100.0)
        self.assertAlmostEqual(pixel_area(Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0)), 400.0)


class SkeletonLengthTests(unittest.TestCase):
    def test_horizontal_line_length(self):
        mask = np.zeros((5, 11), dtype=bool)
        mask[2, 0:11] = True
        result = skeleton_length(mask, 10.0)
        self.assertEqual(result["orthogonal_edges"], 10)
        self.assertEqual(result["diagonal_edges"], 0)
        self.assertAlmostEqual(result["length_m"], 100.0)

    def test_diagonal_line_uses_the_diagonal_weight(self):
        mask = np.zeros((11, 11), dtype=bool)
        for i in range(11):
            mask[i, i] = True
        result = skeleton_length(mask, 10.0)
        self.assertEqual(result["orthogonal_edges"], 0)
        self.assertEqual(result["diagonal_edges"], 10)
        self.assertAlmostEqual(result["length_m"], 10 * np.sqrt(2.0) * 10.0, places=6)

    def test_pixel_count_overestimates_a_wide_band(self):
        from scipy import ndimage as ndi

        line = np.zeros((11, 11), dtype=bool)
        for i in range(11):
            line[i, i] = True
        band = ndi.binary_dilation(line, structure=np.ones((3, 3), dtype=bool))
        measured = skeleton_length(band, 10.0)["length_m"]
        naive = int(band.sum()) * 10.0
        # The pixel count measures area, so a 3 px wide band looks 3x too long.
        self.assertGreater(naive, measured)
        self.assertGreater(naive / measured, 2.0)

    def test_pixel_count_underestimates_a_one_pixel_diagonal(self):
        mask = np.zeros((11, 11), dtype=bool)
        for i in range(11):
            mask[i, i] = True
        measured = skeleton_length(mask, 10.0)["length_m"]
        naive = int(mask.sum()) * 10.0
        self.assertLess(naive, measured)

    def test_empty_mask_returns_zero(self):
        result = skeleton_length(np.zeros((4, 4), dtype=bool), 10.0)
        self.assertEqual(result["length_m"], 0.0)
        self.assertEqual(result["skeleton_px"], 0)


class CleanBinaryMaskTests(unittest.TestCase):
    def test_matches_the_previous_inline_implementation(self):
        from scipy import ndimage as ndi

        rng = np.random.default_rng(0)
        mask = rng.random((40, 40)) > 0.55
        valid = rng.random((40, 40)) > 0.1

        manual = ndi.binary_opening(mask, np.ones((3, 3), dtype=bool))
        manual = ndi.binary_closing(manual, np.ones((5, 5), dtype=bool))
        manual &= valid
        labels, count = ndi.label(manual)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        keep = sizes >= 4
        keep[0] = False
        manual = keep[labels]

        self.assertTrue(np.array_equal(
            clean_binary_mask(mask, valid=valid), manual))

    def test_zero_min_area_keeps_singletons(self):
        # Blocks must sit away from the array border and further apart than the
        # 5x5 closing, otherwise closing bridges them or the border erodes them.
        mask = np.zeros((30, 30), dtype=bool)
        mask[5:8, 5:8] = True
        mask[18:21, 18:21] = True
        kept = clean_binary_mask(mask, min_area_px=0)
        self.assertTrue(kept[5:8, 5:8].all())
        self.assertTrue(kept[18:21, 18:21].all())
        self.assertEqual(int(kept.sum()), 18)

    def test_validity_is_reapplied_after_closing(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[4:8, 4:8] = True
        valid = np.ones((12, 12), dtype=bool)
        valid[4:8, 4:6] = False       # half the block is invalid
        cleaned = clean_binary_mask(mask, valid=valid, min_area_px=0)
        self.assertFalse(cleaned[4:8, 4:6].any())

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape does not match"):
            clean_binary_mask(np.ones((5, 5), dtype=bool),
                              valid=np.ones((4, 4), dtype=bool))


if __name__ == "__main__":
    unittest.main()
