"""Tests for the annotation-to-candidate crosswalk.

The crosswalk must match by geometry and never by record-id suffix: a record
numbered B_151 can belong to candidate B_161, and candidate B_151 can be
unannotated. These tests pin that behaviour, plus unmatched records, shared
candidates, and determinism.

The last test reads the real crosswalk if it is present, so the six known
holdout remappings stay checked without requiring the data to exist.
"""

from __future__ import annotations

import csv
import os
import unittest

import geopandas as gpd
from shapely.geometry import box

from scripts.registers.build_candidate_crosswalk import build_rows, partition_lookup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROSSWALK = os.path.join(ROOT, "outputs", "labels", "pair_b_v2", "candidate_crosswalk.csv")


def _frame(records, crs="EPSG:32643"):
    """GeoDataFrame of (record_id, label_class, geometry) tuples."""
    return gpd.GeoDataFrame(
        [{"record_id": record_id, "pair": "B", "label_class": label_class}
         for record_id, label_class, _ in records],
        geometry=[geometry for _, _, geometry in records], crs=crs)


def _candidates(rows, crs="EPSG:32643"):
    return gpd.GeoDataFrame(
        [{"candidate_id": candidate_id} for candidate_id, _ in rows],
        geometry=[geometry for _, geometry in rows], crs=crs)


PIECES = {"train": box(0.0, 0.0, 1000.0, 1000.0),
          "validation": box(1200.0, 0.0, 2000.0, 1000.0)}


class GeometricMappingTests(unittest.TestCase):
    def test_record_numbers_do_not_decide_the_match(self):
        # Candidate B_161 sits at x 100, candidate B_151 at x 500. The record
        # numbered B_151 is drawn over B_161, so that is the match.
        candidates = _candidates([("B_161", box(100.0, 100.0, 140.0, 140.0)),
                                  ("B_151", box(500.0, 100.0, 540.0, 140.0))])
        annotations = _frame([("B_151", "change", box(90.0, 90.0, 150.0, 150.0))])
        rows = build_rows(annotations, candidates, PIECES)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["annotation_record_id"], "B_151")
        self.assertEqual(rows[0]["candidate_id"], "B_161")
        self.assertEqual(rows[0]["match_type"], "confident_one_to_one")
        self.assertGreaterEqual(rows[0]["association_value"], 0.90)

    def test_partition_comes_from_the_geometry(self):
        candidates = _candidates([("B_9", box(1300.0, 100.0, 1340.0, 140.0))])
        annotations = _frame([("B_9", "no_change", box(1290.0, 90.0, 1350.0, 150.0))])
        rows = build_rows(annotations, candidates, PIECES)
        self.assertEqual(rows[0]["partition"], "validation")


class UnmatchedTests(unittest.TestCase):
    def test_annotation_away_from_every_candidate(self):
        candidates = _candidates([("B_1", box(100.0, 100.0, 140.0, 140.0))])
        annotations = _frame([("B_190", "change", box(900.0, 900.0, 950.0, 950.0))])
        rows = build_rows(annotations, candidates, PIECES)
        self.assertEqual(rows[0]["candidate_id"], "")
        self.assertEqual(rows[0]["match_type"], "unmatched_no_overlap")
        self.assertEqual(rows[0]["association_value"], 0.0)
        self.assertEqual(rows[0]["candidate_shared_by"], 0)


class SharedCandidateTests(unittest.TestCase):
    def test_two_annotations_inside_one_candidate_are_both_reported(self):
        candidates = _candidates([("B_7", box(100.0, 100.0, 200.0, 200.0))])
        annotations = _frame([
            ("B_7", "change", box(100.0, 100.0, 140.0, 140.0)),
            ("B_8", "change", box(160.0, 160.0, 200.0, 200.0)),
        ])
        rows = build_rows(annotations, candidates, PIECES)
        self.assertEqual({row["candidate_id"] for row in rows}, {"B_7"})
        self.assertEqual([row["candidate_shared_by"] for row in rows], [2, 2])

    def test_distinct_candidates_report_one(self):
        candidates = _candidates([("B_1", box(100.0, 100.0, 140.0, 140.0)),
                                  ("B_2", box(300.0, 100.0, 340.0, 140.0))])
        annotations = _frame([("B_1", "change", box(95.0, 95.0, 145.0, 145.0)),
                              ("B_2", "change", box(295.0, 95.0, 345.0, 145.0))])
        rows = build_rows(annotations, candidates, PIECES)
        self.assertEqual([row["candidate_shared_by"] for row in rows], [1, 1])


class DeterminismTests(unittest.TestCase):
    def test_repeated_builds_are_identical(self):
        candidates = _candidates([("B_1", box(100.0, 100.0, 140.0, 140.0)),
                                  ("B_2", box(300.0, 100.0, 340.0, 140.0))])
        annotations = _frame([("B_1", "change", box(90.0, 90.0, 150.0, 150.0)),
                              ("B_9", "no_change", box(600.0, 600.0, 650.0, 650.0))])
        first = build_rows(annotations, candidates, PIECES)
        second = build_rows(annotations, candidates, PIECES)
        self.assertEqual(first, second)


class RealCrosswalkTests(unittest.TestCase):
    """Checks against the generated file when it exists."""

    def setUp(self):
        if not os.path.exists(CROSSWALK):
            self.skipTest("crosswalk not generated in this tree")
        with open(CROSSWALK, newline="") as handle:
            self.rows = {row["annotation_record_id"]: row
                         for row in csv.DictReader(handle)}

    def test_holdout_remappings_are_recorded(self):
        expected = {"B_151": "B_161", "B_152": "B_163", "B_153": "B_169",
                    "B_154": "B_174", "B_155": "B_175", "B_156": "B_179"}
        for record_id, candidate_id in expected.items():
            with self.subTest(record=record_id):
                self.assertIn(record_id, self.rows)
                self.assertEqual(self.rows[record_id]["candidate_id"], candidate_id)

    def test_every_record_has_a_row(self):
        self.assertEqual(len(self.rows), 190)

    def test_shared_candidates_are_flagged_consistently(self):
        counts = {}
        for row in self.rows.values():
            if row["candidate_id"]:
                counts[row["candidate_id"]] = counts.get(row["candidate_id"], 0) + 1
        for row in self.rows.values():
            if row["candidate_id"]:
                self.assertEqual(int(row["candidate_shared_by"]),
                                 counts[row["candidate_id"]])


if __name__ == "__main__":
    unittest.main()
