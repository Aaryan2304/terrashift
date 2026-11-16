"""Tests for mask-to-component vectorisation."""

from __future__ import annotations

import unittest

import numpy as np
from rasterio.transform import Affine

from scripts.components_to_geojson import extract

TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)


def _mask(shape=(20, 20), boxes=()):
    mask = np.zeros(shape, dtype=bool)
    for (r0, r1, c0, c1) in boxes:
        mask[r0:r1, c0:c1] = True
    return mask


class ExtractTests(unittest.TestCase):
    def test_one_component_per_blob(self):
        mask = _mask(boxes=[(2, 6, 2, 6), (10, 14, 10, 14)])
        records = extract(mask, TRANSFORM, 10.0, min_area_px=4)
        self.assertEqual(len(records), 2)
        self.assertEqual(sorted(r["area_px"] for r in records), [16, 16])

    def test_small_components_are_dropped(self):
        mask = _mask(boxes=[(2, 6, 2, 6)])
        mask[15, 15] = True          # one pixel, below the default minimum
        records = extract(mask, TRANSFORM, 10.0, min_area_px=4)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["area_px"], 16)

    def test_area_range_brackets_the_nominal_area(self):
        mask = _mask(boxes=[(5, 9, 5, 9)])   # 4x4 = 16 px = 1600 m2 at 10 m
        record = extract(mask, TRANSFORM, 10.0, min_area_px=1)[0]
        self.assertEqual(record["area_m2"], 1600)
        self.assertLess(record["area_min_m2"], record["area_m2"])
        self.assertGreater(record["area_max_m2"], record["area_m2"])

    def test_linear_component_reports_a_length(self):
        mask = _mask(shape=(10, 11))
        mask[5, 0:11] = True
        record = extract(mask, TRANSFORM, 10.0, min_area_px=1)[0]
        self.assertAlmostEqual(record["skeleton_length_m"], 100.0, places=6)

    def test_geometry_is_a_polygon_in_map_units(self):
        mask = _mask(boxes=[(4, 8, 4, 8)])
        record = extract(mask, TRANSFORM, 10.0, min_area_px=1)[0]
        self.assertIn(record["geometry"]["type"], ("Polygon", "MultiPolygon"))
        xs = [point[0] for point in record["geometry"]["coordinates"][0]]
        self.assertGreaterEqual(min(xs), 500000.0)
        self.assertLessEqual(max(xs), 500100.0)

    def test_empty_mask_returns_no_records(self):
        self.assertEqual(extract(np.zeros((8, 8), dtype=bool), TRANSFORM, 10.0), [])


if __name__ == "__main__":
    unittest.main()
