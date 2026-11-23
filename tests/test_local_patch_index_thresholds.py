"""Tests for split-specific minimum labelled fractions in the local indexer.

Two layers:

* ``resolve_split_thresholds`` as a pure function: scalar compatibility, mappings,
  malformed input, unknown splits, the test-split ban.
* ``build_local_patch_index`` end to end on a synthetic 128 x 128 fixture with a
  train / guard / validation / guard / test partition and hand-placed labels, so
  the selection rules are exercised rather than re-implemented in the test.

Everything is synthetic. No real imagery, labels or partitions are read.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin

from terrashift.data.local_patches import (build_local_patch_index, resolve_split_thresholds,
                                write_local_patch_index)

CRS = "EPSG:32643"
X0, Y0 = 500000.0, 3000000.0
GRID = 128
PATCH = 32
STRIDE = 8


def _write_raster(path, array, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                       width=array.shape[1], count=1, dtype=array.dtype,
                       crs=CRS, transform=from_origin(X0, Y0, 10.0, 10.0),
                       nodata=nodata) as dataset:
        dataset.write(array, 1)


class ThresholdResolverTests(unittest.TestCase):
    SPLITS = {"train": 1, "validation": 2}

    def test_scalar_applies_to_every_split(self):
        self.assertEqual(resolve_split_thresholds(0.05, self.SPLITS),
                         {"train": 0.05, "validation": 0.05})

    def test_scalar_zero_is_still_rejected(self):
        # Historical behaviour: a scalar call never accepted zero.
        with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
            resolve_split_thresholds(0.0, self.SPLITS)

    def test_scalar_out_of_range_is_rejected(self):
        for value in (-0.1, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_split_thresholds(value, self.SPLITS)

    def test_non_numeric_is_rejected(self):
        for value in ("0.002", None, [0.1]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be a number or a split mapping"):
                    resolve_split_thresholds(value, self.SPLITS)

    def test_mapping_allows_zero_for_validation(self):
        resolved = resolve_split_thresholds({"train": 0.002, "validation": 0.0},
                                            self.SPLITS)
        self.assertEqual(resolved, {"train": 0.002, "validation": 0.0})

    def test_mapping_with_unknown_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown splits"):
            resolve_split_thresholds({"train": 0.002, "validation": 0.0,
                                      "holdout": 0.1}, self.SPLITS)

    def test_mapping_missing_a_required_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no threshold for split"):
            resolve_split_thresholds({"train": 0.002}, self.SPLITS)

    def test_mapping_with_bad_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be in \[0, 1\]"):
            resolve_split_thresholds({"train": 0.002, "validation": 1.2}, self.SPLITS)
        with self.assertRaisesRegex(ValueError, "must be a number"):
            resolve_split_thresholds({"train": "0.002", "validation": 0.0}, self.SPLITS)

    def test_mapping_may_not_configure_test(self):
        with self.assertRaisesRegex(ValueError, "test split must stay excluded"):
            resolve_split_thresholds({"train": 0.002, "validation": 0.0, "test": 0.0},
                                     {"train": 1, "validation": 2, "test": 3})


class IndexFixture:
    """128 x 128 synthetic pair with a five-stripe partition."""

    def __init__(self, root):
        root = Path(root)
        self.root = root
        self.label_dir = root / "labels"
        self.partition_dir = root / "partitions"
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.partition_dir.mkdir(parents=True, exist_ok=True)

        partition = np.zeros((GRID, GRID), dtype=np.uint8)
        partition[:, 0:48] = 1        # train
        partition[:, 48:56] = 0       # guard
        partition[:, 56:88] = 2       # validation
        partition[:, 88:96] = 0       # guard
        partition[:, 96:128] = 3      # test
        for pair in ("A", "B"):
            _write_raster(self.partition_dir / ("pair_%s_partition_10m.tif" % pair),
                          partition)

        # Pair A: mostly unreviewed (255), with hand-placed witnesses. Keeping the
        # background at 255 is what makes the labelled-fraction tests meaningful: a
        # fully reviewed stripe would give every window fraction 1.0.
        labels_a = np.full((GRID, GRID), 255, dtype=np.uint8)
        labels_a[0:8, 0:8] = 0        # reviewed no-change block, train
        labels_a[8:12, 8:16] = 1      # dense change block, train (96 reviewed in window 0,0)
        labels_a[100, 4] = 1          # single change pixel, train: fraction 1/1024
        labels_a[10, 60] = 1          # single change pixel, validation
        labels_a[0:8, 96:104] = 0     # reviewed block in test, above the 0.002 threshold
        labels_a[10, 100] = 1         # single change pixel, test

        # Pair B: nothing reviewed anywhere, so it must contribute no windows.
        labels_b = np.full((GRID, GRID), 255, dtype=np.uint8)

        for pair, array in (("A", labels_a), ("B", labels_b)):
            _write_raster(self.label_dir / ("manual_labels_%s.tif" % pair), array)

        self.manifest = root / "manifest.yaml"
        self.manifest.write_text(yaml.safe_dump({
            "products": [
                {"id": "a1", "path": str(root / "a1.SAFE")},
                {"id": "a2", "path": str(root / "a2.SAFE")},
                {"id": "b1", "path": str(root / "b1.SAFE")},
                {"id": "b2", "path": str(root / "b2.SAFE")},
            ],
            "pairs": {
                "A": {"date1": "a1", "date2": "a2", "tile": "T43QBB", "crs": CRS},
                "B": {"date1": "b1", "date2": "b2", "tile": "T43RGM", "crs": CRS},
            },
        }), encoding="utf-8")

    def build(self, threshold, include_test=False):
        return build_local_patch_index(
            self.manifest, self.label_dir, self.partition_dir,
            patch_size=PATCH, stride=STRIDE, min_labeled_fraction=threshold,
            include_test=include_test)


def _offsets(rows, split=None):
    return {(row["row_off"], row["col_off"]) for row in rows
            if split is None or row["split"] == split}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fixture = IndexFixture(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_different_train_and_validation_thresholds(self):
        rows = self.fixture.build({"train": 0.002, "validation": 0.0})
        train = _offsets(rows, "train")
        validation = _offsets(rows, "validation")
        # The dense block is selected in train; the single pixel in validation is
        # selected only because validation's threshold is zero.
        self.assertIn((8, 8), train)
        self.assertIn((8, 56), validation)
        # The single train pixel is below 0.002 and must be absent.
        self.assertNotIn((96, 0), train)

    def test_validation_zero_keeps_a_window_with_one_reviewed_pixel(self):
        rows = self.fixture.build({"train": 0.002, "validation": 0.0})
        single = [row for row in rows
                  if row["split"] == "validation" and row["labeled_pixels"] == 1]
        # One pixel, and any window that contains it is a legitimate selection: with
        # stride 8 and size 32 more than one window can see the same pixel.
        self.assertGreaterEqual(len(single), 1)
        for row in single:
            self.assertEqual(row["manual_label_fraction"], 1 / (PATCH * PATCH))

    def test_scalar_threshold_excludes_that_validation_window(self):
        # Legacy behaviour: the same 0.002 applied everywhere drops it.
        rows = self.fixture.build(0.002)
        self.assertEqual(_offsets(rows, "validation"), set())

    def test_low_threshold_keeps_the_single_train_pixel(self):
        rows = self.fixture.build({"train": 0.0005, "validation": 0.0})
        self.assertIn((96, 0), _offsets(rows, "train"))

    def test_uncertain_only_window_is_never_selected(self):
        # (40, 16) sees only 255 pixels; both thresholds are zero-ish for it.
        for threshold in ({"train": 0.0, "validation": 0.0}, 0.002):
            with self.subTest(threshold=threshold):
                rows = self.fixture.build(threshold)
                self.assertNotIn((40, 16), _offsets(rows, "train"))

    def test_no_window_without_reviewed_pixels_is_indexed(self):
        rows = self.fixture.build({"train": 0.0, "validation": 0.0})
        self.assertTrue(rows)
        self.assertTrue(all(row["labeled_pixels"] > 0 for row in rows))

    def test_zero_threshold_still_rejects_empty_windows(self):
        # Pair B is entirely 255, so a zero threshold must yield nothing for it.
        rows = self.fixture.build({"train": 0.0, "validation": 0.0})
        self.assertEqual({row["pair"] for row in rows}, {"A"})

    def test_test_split_is_excluded(self):
        rows = self.fixture.build({"train": 0.002, "validation": 0.0})
        self.assertEqual(_offsets(rows, "test"), set())
        self.assertFalse(any(row["split"] == "test" for row in rows))

    def test_test_split_can_still_be_unlocked_explicitly(self):
        rows = self.fixture.build({"train": 0.002, "validation": 0.0}, include_test=True)
        self.assertTrue(_offsets(rows, "test"))


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fixture = IndexFixture(self._tmp.name)
        self.output = Path(self._tmp.name) / "index.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def test_metadata_records_the_resolved_mapping(self):
        rows = self.fixture.build({"train": 0.002, "validation": 0.0})
        write_local_patch_index(rows, self.output, PATCH, STRIDE,
                                {"train": 0.002, "validation": 0.0})
        meta = json.loads(self.output.with_suffix(".json").read_text())
        self.assertEqual(meta["min_labeled_fraction_by_split"],
                         {"train": 0.002, "validation": 0.0})
        self.assertIn("empty_window_policy", meta)
        self.assertIn("reviewed binary pixel", meta["empty_window_policy"])
        self.assertFalse(meta["test_index_included"])
        self.assertFalse(meta["test_labels_used_for_training"])

    def test_metadata_records_a_scalar_policy_too(self):
        rows = self.fixture.build(0.002)
        write_local_patch_index(rows, self.output, PATCH, STRIDE, 0.002)
        meta = json.loads(self.output.with_suffix(".json").read_text())
        self.assertEqual(meta["min_labeled_fraction"], 0.002)
        self.assertEqual(meta["min_labeled_fraction_by_split"],
                         {"train": 0.002, "validation": 0.002})

    def test_writer_refuses_to_overwrite(self):
        rows = self.fixture.build(0.002)
        write_local_patch_index(rows, self.output, PATCH, STRIDE, 0.002)
        with self.assertRaises(FileExistsError):
            write_local_patch_index(rows, self.output, PATCH, STRIDE, 0.002)


if __name__ == "__main__":
    unittest.main()
