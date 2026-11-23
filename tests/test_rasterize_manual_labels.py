"""Contract tests for the annotation rasteriser, on a synthetic fixture.

The rasteriser produced every label mask in this project and had no test of its
own. These pin the properties that the reported numbers depend on:

* label values: change 1, no_change 0, uncertain and unlabelled 255;
* the output grid is the partition grid, not a grid of its own;
* ``all_touched=False``: a polygon must cover a pixel centre to burn it;
* overlapping polygons, split crossings and invalid observations are rejected.

Nothing here is real imagery or a real annotation. The fixture is built in the
test: one 600 m tile at 10 m with a train and a validation half, and a 20 m
validity raster with one deliberately invalid block.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform

from scripts.review.rasterize_manual_labels import rasterize_annotations

PAIR = "B"
X0, Y0 = 700000.0, 3100000.0      # top-left corner, EPSG:32643
CRS_UTM = "EPSG:32643"
CRS_LL = "EPSG:4326"
SIZE_10M = 60                      # 600 m
SIZE_20M = 30

_TO_LL = Transformer.from_crs(CRS_UTM, CRS_LL, always_xy=True)


def _ll(geometry):
    return shapely_transform(_TO_LL.transform, geometry)


def _cell_box(row0, row1, col0, col1, resolution=10.0):
    """Box covering a row/column range of the 10 m partition grid."""
    left = X0 + col0 * resolution
    right = X0 + col1 * resolution
    top = Y0 - row0 * resolution
    bottom = Y0 - row1 * resolution
    return box(left, bottom, right, top)


def _write(path, array, transform, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                       width=array.shape[1], count=1, dtype=array.dtype,
                       crs=CRS_UTM, transform=transform, nodata=nodata) as dataset:
        dataset.write(array, 1)


class Fixture:
    """A tiny but complete set of rasters for one pair."""

    def __init__(self, root):
        self.root = root
        self.partition_dir = os.path.join(root, "partitions")
        self.score_dir = os.path.join(root, "score")
        self.validity_dir = os.path.join(root, "validity")
        self.label_dir = os.path.join(root, "out")
        for directory in (self.partition_dir, self.score_dir, self.validity_dir):
            os.makedirs(directory, exist_ok=True)

        partition = np.zeros((SIZE_10M, SIZE_10M), dtype=np.uint8)
        partition[:, :30] = 1          # train
        partition[:, 30:] = 2          # validation
        # from_origin(west, north, xsize, ysize) already negates ysize, so a
        # positive ysize gives a north-up grid whose top row sits at Y0.
        self.partition_transform = from_origin(X0, Y0, 10.0, 10.0)
        _write(os.path.join(self.partition_dir, "pair_%s_partition_10m.tif" % PAIR),
               partition, self.partition_transform)

        validity = np.ones((SIZE_20M, SIZE_20M), dtype=np.uint8)
        validity[2:4, 2:4] = 0         # invalid block, 40-80 m from both edges
        score_transform = from_origin(X0, Y0, 20.0, 20.0)
        _write(os.path.join(self.validity_dir, "observation_valid_%s.tif" % PAIR),
               validity, score_transform)
        _write(os.path.join(self.score_dir, "change_score_%s.tif" % PAIR),
               np.zeros((SIZE_20M, SIZE_20M), dtype=np.float32), score_transform)

    def labels(self, geometries, name="labels.geojson"):
        features = []
        for index, (record_id, label_class, geometry) in enumerate(geometries, start=1):
            features.append({
                "type": "Feature",
                "geometry": mapping(_ll(geometry)),
                "properties": {
                    "record_id": record_id,
                    "pair": PAIR,
                    "label_class": label_class,
                    "reviewer": "synthetic",
                    "reviewed_at": "2026-11-20",
                    "evidence_source": "before_after_visual",
                    "evidence": "synthetic fixture for the rasteriser contract",
                    "confidence": "medium",
                },
            })
        path = Path(self.root) / name
        with open(path, "w") as handle:
            json.dump({"type": "FeatureCollection", "features": features}, handle)
        return path

    def run(self, geometries, name="labels.geojson", output="out"):
        path = self.labels(geometries, name)
        out_dir = Path(self.root) / output
        result = rasterize_annotations(PAIR, path, out_dir, Path(self.partition_dir),
                                       Path(self.score_dir),
                                       validity_dir=Path(self.validity_dir))
        with rasterio.open(result["mask"]) as dataset:
            mask = dataset.read(1)
            grid = (dataset.crs, dataset.transform, dataset.shape)
        return mask, grid, result


class LabelValueTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_values_and_unlabelled_default(self):
        mask, _, _ = self.fixture.run([
            ("B_1", "change", _cell_box(20, 24, 2, 6)),
            ("B_2", "no_change", _cell_box(20, 24, 32, 36)),
            ("B_3", "uncertain", _cell_box(30, 32, 10, 12)),
        ])
        self.assertEqual(int(mask[20, 2]), 1)
        self.assertEqual(int(mask[20, 32]), 0)
        self.assertEqual(int(mask[30, 10]), 255)     # uncertain is not a negative
        self.assertEqual(int(mask[50, 50]), 255)     # unreviewed
        self.assertEqual(set(np.unique(mask).tolist()), {0, 1, 255})

    def test_output_grid_is_the_partition_grid(self):
        _, grid, _ = self.fixture.run([("B_1", "change", _cell_box(20, 24, 2, 6))])
        crs, transform, shape = grid
        self.assertEqual(str(crs), CRS_UTM)
        self.assertEqual(shape, (SIZE_10M, SIZE_10M))
        self.assertEqual(transform.a, 10.0)
        self.assertEqual(transform.c, X0)
        self.assertEqual(transform.f, Y0)

    def test_audit_row_records_split_and_pixel_count(self):
        _, _, result = self.fixture.run([("B_1", "change", _cell_box(20, 24, 2, 6))])
        self.assertEqual(result["feature_count"], 1)
        self.assertEqual(result["pixels"], 16)


class RejectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_sub_pixel_polygon_covers_no_centre(self):
        # 4 m x 4 m inside a single 10 m pixel: with all_touched=False nothing
        # burns. With all_touched=True this would have burned a pixel, so the
        # test pins the pixel-centre rule.
        # Pixel centres sit at X0 + 5 + 10k. This 3 m box falls between two
        # centres in both axes, so no centre is inside it.
        sliver = box(X0 + 21.0, Y0 - 24.0, X0 + 24.0, Y0 - 21.0)
        with self.assertRaisesRegex(ValueError, "covers no pixel centres"):
            self.fixture.run([("B_1", "change", sliver)])

    def test_overlapping_polygons_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlaps another reviewed polygon"):
            self.fixture.run([
                ("B_1", "change", _cell_box(20, 26, 2, 8)),
                ("B_2", "no_change", _cell_box(22, 28, 4, 10)),
            ])

    def test_touching_polygons_are_not_an_overlap(self):
        # The guard is pixel-level, not bounding-box level: two polygons sharing
        # an edge burn their own pixels and both survive. Without this, the
        # overlap test above could pass for the wrong reason.
        mask, _, result = self.fixture.run([
            ("B_1", "change", _cell_box(20, 24, 2, 6)),
            ("B_2", "no_change", _cell_box(20, 24, 6, 10)),
        ])
        self.assertEqual(result["feature_count"], 2)
        self.assertEqual(int(mask[20, 2]), 1)
        self.assertEqual(int(mask[20, 6]), 0)
        self.assertEqual(int(mask[20, 5]), 1)
        self.assertEqual(int(mask[20, 9]), 0)

    def test_split_crossing_polygon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "crosses a split boundary"):
            self.fixture.run([("B_1", "change", _cell_box(2, 6, 28, 32))])

    def test_invalid_observation_is_rejected(self):
        # The validity block at 20 m rows 2:4, cols 2:4 is 10 m rows 4:8, cols 4:8.
        with self.assertRaisesRegex(ValueError, "not valid observations"):
            self.fixture.run([("B_1", "change", _cell_box(4, 8, 4, 8))])

    def test_polygon_outside_the_tile_is_rejected(self):
        outside = box(X0 - 200.0, Y0 - 200.0, X0 - 100.0, Y0 - 100.0)
        with self.assertRaisesRegex(ValueError, "outside the pair tile"):
            self.fixture.run([("B_1", "change", outside)])


if __name__ == "__main__":
    unittest.main()
