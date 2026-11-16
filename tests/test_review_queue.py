"""Tests for the review-queue builder.

Synthetic fixtures only: a small partition raster and hand-made candidate
polygons. No real candidate layer and no real partition is read, and nothing
here creates a label.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from scripts.build_review_queue import (FIELDS, candidate_rows, classify_partition,
                                        order_rows)

TRANSFORM = from_origin(500000.0, 1000000.0, 20.0, 20.0)
SHAPE = (100, 100)


def write_partition(path, left_code=1, right_code=2, guard_cols=()):
    data = np.full(SHAPE, left_code, dtype=np.uint8)
    data[:, 50:] = right_code
    for col in guard_cols:
        data[:, col] = 0
    profile = dict(driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1,
                   dtype="uint8", crs="EPSG:32643", transform=TRANSFORM)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data, 1)


def polygon_feature(candidate_id, col0, col1, row0, row1, area_px=10):
    """A candidate whose geometry is a map-coordinate box, written in WGS84."""
    to_wgs84 = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    left = TRANSFORM.c + col0 * TRANSFORM.a
    right = TRANSFORM.c + col1 * TRANSFORM.a
    top = TRANSFORM.f + row0 * TRANSFORM.e
    bottom = TRANSFORM.f + row1 * TRANSFORM.e
    ring = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
    coords = [to_wgs84.transform(x, y) for x, y in ring]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {"candidate_id": candidate_id, "pair": "A", "component_id": 1,
                       "area_px_20m": area_px, "mean_score": 9.0, "max_score": 12.0,
                       "score_threshold": 7.7, "priority_percentile": 99.9},
    }


class ClassifyPartitionTests(unittest.TestCase):
    def test_single_split(self):
        codes = np.array([1, 1, 1], dtype=np.uint8)
        self.assertEqual(classify_partition(codes, codes.size),
                         ("train", False, False, 0.0))

    def test_guard_only(self):
        codes = np.zeros(4, dtype=np.uint8)
        partition, crosses, touches, fraction = classify_partition(codes, codes.size)
        self.assertEqual(partition, "guard")
        self.assertTrue(touches)
        self.assertEqual(fraction, 1.0)

    def test_split_plus_guard_is_not_mixed_but_is_flagged(self):
        codes = np.array([1, 1, 0, 0], dtype=np.uint8)
        partition, crosses, touches, fraction = classify_partition(codes, codes.size)
        self.assertEqual(partition, "train")
        self.assertFalse(crosses)
        self.assertTrue(touches)
        self.assertEqual(fraction, 0.5)

    def test_two_splits_is_mixed_and_crosses(self):
        codes = np.array([1, 2, 2], dtype=np.uint8)
        partition, crosses, _, _ = classify_partition(codes, codes.size)
        self.assertEqual(partition, "mixed")
        self.assertTrue(crosses)

    def test_empty_codes_fall_back_to_guard(self):
        partition, _, touches, fraction = classify_partition(
            np.zeros(0, dtype=np.uint8), 0)
        self.assertEqual(partition, "guard")
        self.assertEqual(fraction, 0.0)


class CandidateRowTests(unittest.TestCase):
    def build(self, features, **raster):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = os.path.join(temporary.name, "partition.tif")
        write_partition(path, **raster)
        rows = candidate_rows("A", {"features": features}, path)
        return rows

    def test_row_has_every_field_and_blank_human_columns(self):
        rows = self.build([polygon_feature("A_1", 10, 14, 10, 14)])
        row = rows[0]
        self.assertEqual(set(row), set(FIELDS))
        self.assertEqual(row["review_status"], "unreviewed")
        for column in ("human_decision", "evidence", "confidence", "note"):
            self.assertEqual(row[column], "")

    def test_train_and_validation_are_separated(self):
        rows = self.build([polygon_feature("A_1", 10, 14, 10, 14),
                           polygon_feature("A_2", 60, 64, 10, 14)])
        self.assertEqual([r["partition"] for r in rows], ["train", "validation"])

    def test_polygon_across_the_boundary_is_mixed(self):
        rows = self.build([polygon_feature("A_1", 46, 56, 10, 14)])
        self.assertEqual(rows[0]["partition"], "mixed")
        self.assertEqual(rows[0]["crosses_partition_boundary"], True)
        self.assertEqual(rows[0]["n_partitions"], 2)

    def test_polygon_in_the_guard_band_is_flagged(self):
        # Guard columns 48-51, so a box spanning columns 48..52 lies wholly in it.
        rows = self.build([polygon_feature("A_1", 48, 52, 10, 14)],
                          guard_cols=(48, 49, 50, 51))
        self.assertEqual(rows[0]["partition"], "guard")
        self.assertEqual(rows[0]["touches_guard_band"], True)

    def test_measurements(self):
        rows = self.build([polygon_feature("A_1", 10, 14, 10, 14, area_px=9)])
        row = rows[0]
        self.assertAlmostEqual(row["bbox_width_m"], 80.0, places=6)
        self.assertAlmostEqual(row["bbox_height_m"], 80.0, places=6)
        self.assertAlmostEqual(row["component_area_m2"], 9 * 400.0, places=1)
        self.assertEqual(row["bbox_width_px_20m"], 4)
        # The synthetic raster origin is 500 000 / 1 000 000 in EPSG:32643, so the
        # centroid round-trips to a fixed longitude and latitude; check the
        # round-trip rather than a hard-coded degree value.
        to_wgs84 = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
        expected_lon, expected_lat = to_wgs84.transform(500000.0 + 12 * 20.0,
                                                        1000000.0 - 12 * 20.0)
        self.assertAlmostEqual(row["centroid_lon"], expected_lon, places=5)
        self.assertAlmostEqual(row["centroid_lat"], expected_lat, places=5)


class OrderTests(unittest.TestCase):
    def row(self, candidate_id, pair, partition, area, score, crosses=False,
            guard=False):
        return {"candidate_id": candidate_id, "pair": pair, "partition": partition,
                "component_area_px_20m": area, "max_score": score,
                "crosses_partition_boundary": crosses,
                "touches_guard_band": guard, "review_order": ""}

    def test_pairs_then_partitions_then_flags_then_area(self):
        rows = [
            self.row("B_1", "B", "train", 500, 10.0),
            self.row("A_1", "A", "validation", 100, 9.0),
            self.row("A_2", "A", "train", 50, 12.0),
            self.row("A_3", "A", "train", 10, 13.0, guard=True),
            self.row("A_4", "A", "train", 200, 11.0),
        ]
        order_rows(rows)
        self.assertEqual([r["candidate_id"] for r in rows],
                         ["A_3", "A_4", "A_2", "A_1", "B_1"])
        self.assertEqual([r["review_order"] for r in rows], [1, 2, 3, 1, 1])

    def test_mixed_sorts_after_split_groups(self):
        rows = [self.row("A_1", "A", "mixed", 999, 14.0),
                self.row("A_2", "A", "guard", 999, 14.0),
                self.row("A_3", "A", "train", 1, 9.0)]
        order_rows(rows)
        self.assertEqual([r["partition"] for r in rows], ["train", "guard", "mixed"])


if __name__ == "__main__":
    unittest.main()
