"""Tests for mask-to-component vectorisation."""

from __future__ import annotations

import unittest

import numpy as np
from rasterio.transform import Affine

from terrashift.geospatial.components import extract

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

    def test_diagonal_pair_keeps_both_pixels_in_the_geometry(self):
        # Regression: labelling used 8-connectivity while polygonisation used
        # rasterio's 4-connectivity default and kept only the first part, so the
        # record said 200 m2 against a polygon of 100 m2.
        mask = np.zeros((8, 8), dtype=bool)
        mask[2, 2] = True
        mask[3, 3] = True
        records = extract(mask, TRANSFORM, 10.0, min_area_px=1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["area_px"], 2)
        self.assertAlmostEqual(records[0]["area_m2"], 200.0, places=6)
        self.assertAlmostEqual(records[0]["geometry_area_m2"], 200.0, places=6)

    def test_raster_area_and_polygon_area_agree_for_a_block(self):
        mask = _mask(boxes=[(4, 9, 4, 9)])   # 5x5
        record = extract(mask, TRANSFORM, 10.0, min_area_px=1)[0]
        self.assertAlmostEqual(record["geometry_area_m2"], record["area_m2"], places=6)

    def test_a_hole_survives_polygonisation(self):
        mask = _mask(boxes=[(3, 9, 3, 9)])
        mask[5:7, 5:7] = False
        records = extract(mask, TRANSFORM, 10.0, min_area_px=1)
        self.assertEqual(len(records), 1)
        # 6x6 outer minus 2x2 inner = 36 - 4 = 32 px = 3,200 m2
        self.assertAlmostEqual(records[0]["geometry_area_m2"], 3200.0, places=6)
        # A polygon with one hole has two rings: the exterior and the interior.
        self.assertEqual(len(records[0]["geometry"]["coordinates"]), 2)

    def test_length_envelope_contains_the_nominal(self):
        # Keep the line away from the array border: dilation is clipped there,
        # which bends the skeleton and shortens the measurement by about 6 %.
        mask = _mask(shape=(12, 15))
        mask[6, 2:13] = True
        record = extract(mask, TRANSFORM, 10.0, min_area_px=1)[0]
        self.assertAlmostEqual(record["skeleton_length_m"], 100.0, places=6)
        self.assertEqual(record["skeleton_length_min_m"], 0.0)   # one pixel wide
        self.assertGreaterEqual(record["skeleton_length_max_m"], record["skeleton_length_m"])
        self.assertEqual(record["skeleton_endpoints"], 2)
        # The raw values stay visible, so the envelope is never read alone.
        self.assertEqual(record["skeleton_length_eroded_m"], 0.0)
        self.assertGreater(record["skeleton_length_dilated_m"], 0.0)

    def test_skeleton_fields_are_named_as_diagnostics_not_road_length(self):
        record = extract(_mask(boxes=[(4, 8, 4, 8)]), TRANSFORM, 10.0, min_area_px=1)[0]
        for forbidden in ("road_length_m", "length_min_m", "length_max_m",
                          "linear_feature_length_m"):
            self.assertNotIn(forbidden, record)
        self.assertEqual(record["geometry_class"], "unclassified")


if __name__ == "__main__":
    unittest.main()
