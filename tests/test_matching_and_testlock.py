"""Synthetic tests for the extracted matching and locked-test helpers."""

import importlib
import unittest

import numpy as np
from shapely.geometry import box

from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.matching import (
    CANDIDATE_COVERED_CONFIDENT,
    CANDIDATE_COVERED_WEAK,
    match_candidates,
)
from terrashift.evaluation.testlock import read_test_target


class Row(dict):
    """Dict that also allows attribute access, like a GeoDataFrame row."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class Frame:
    """Minimal stand-in for a GeoDataFrame: iterrows over dict-backed rows."""
    def __init__(self, rows):
        self._rows = [Row(row) for row in rows]
    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, row


class MatchingTests(unittest.TestCase):
    def test_exact_overlap_gives_full_coverage(self):
        annotations = Frame([{"record_id": "A_1", "geometry": box(0, 0, 10, 10)}])
        candidates = Frame([{"candidate_id": 7, "geometry": box(0, 0, 10, 10)}])
        match = match_candidates(annotations, candidates)["A_1"]
        self.assertEqual(len(match), 1)
        self.assertAlmostEqual(match[0]["iou"], 1.0)
        self.assertAlmostEqual(match[0]["annotation_covered"], 1.0)
        self.assertEqual(match[0]["candidate_id"], 7)

    def test_disjoint_geometry_yields_no_match(self):
        annotations = Frame([{"record_id": "A_2", "geometry": box(0, 0, 1, 1)}])
        candidates = Frame([{"candidate_id": 1, "geometry": box(5, 5, 6, 6)}])
        self.assertEqual(match_candidates(annotations, candidates)["A_2"], [])

    def test_half_coverage_and_iou_are_calculable(self):
        # candidate covers the right half of the annotation: IoU = 0.5, covered = 0.5
        annotations = Frame([{"record_id": "A_3", "geometry": box(0, 0, 10, 10)}])
        candidates = Frame([{"candidate_id": 2, "geometry": box(5, 0, 15, 10)}])
        row = match_candidates(annotations, candidates)["A_3"][0]
        # intersection 50 m2, union 150 m2 -> IoU 1/3, not 1/2
        self.assertAlmostEqual(row["iou"], 50/150)
        self.assertAlmostEqual(row["annotation_covered"], 0.5)
        self.assertAlmostEqual(row["candidate_covered"], 0.5)

    def test_multiple_candidates_are_ranked_by_iou(self):
        annotations = Frame([{"record_id": "A_4", "geometry": box(0, 0, 10, 10)}])
        candidates = Frame([
            {"candidate_id": 10, "geometry": box(0, 0, 2, 2)},    # IoU 0.04
            {"candidate_id": 11, "geometry": box(0, 0, 10, 10)},  # IoU 1.0
        ])
        rows = match_candidates(annotations, candidates)["A_4"]
        self.assertEqual([r["candidate_id"] for r in rows], [11, 10])

    def test_empty_inputs_return_empty_mapping(self):
        self.assertEqual(match_candidates(Frame([]), Frame([])), {})

    def test_confidence_constants_are_unchanged(self):
        self.assertEqual(CANDIDATE_COVERED_CONFIDENT, 0.90)
        self.assertEqual(CANDIDATE_COVERED_WEAK, 0.10)

    def test_script_reexports_the_package_objects(self):
        script = importlib.import_module("scripts.audit.audit_annotations_vs_candidates")
        self.assertIs(script.match_candidates, match_candidates)


class TestLockTests(unittest.TestCase):
    def test_test_code_matches_the_specification(self):
        from terrashift.evaluation.testlock import TEST_CODE
        self.assertEqual(TEST_CODE, 3)

    def test_script_reexports_the_package_object(self):
        script = importlib.import_module("scripts.evaluation.evaluate_locked_test")
        self.assertIs(script.read_test_target, read_test_target)

    def test_target_uses_ignore_value_outside_the_test_stripe(self):
        import tempfile
        from pathlib import Path
        import rasterio
        from rasterio.transform import from_origin

        tmp = Path(tempfile.mkdtemp())
        partition_path, labels_path = tmp / "p.tif", tmp / "l.tif"
        partition = np.zeros((8, 30), dtype=np.uint8)
        partition[:, 20:] = 3                     # right third is the test stripe
        labels = np.ones((8, 30), dtype=np.uint8)
        for path, arr in ((partition_path, partition), (labels_path, labels)):
            with rasterio.open(path, "w", driver="GTiff", height=8, width=30, count=1,
                               dtype="uint8", transform=from_origin(0, 0, 1, 1)) as dst:
                dst.write(arr, 1)

        target, col0, width = read_test_target(labels_path, partition_path)
        self.assertEqual(col0, 20)
        self.assertEqual(width, 30)
        self.assertEqual(target.shape, (8, 10))
        self.assertTrue((target[partition[:, 20:] != 3] == IGNORE_VALUE).all())


if __name__ == "__main__":
    unittest.main()
