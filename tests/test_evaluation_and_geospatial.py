"""Focused tests for the migrated evaluation, geospatial and register surfaces.

Expected values are hand-computable from the synthetic inputs. The register checks read the
existing CSVs and never write them.
"""

from __future__ import annotations

import csv
import pathlib
import unittest

import numpy as np

import quantize
import terrashift.evaluation.lattice as lattice
import terrashift.evaluation.prediction as prediction
import terrashift.evaluation.sessions as sessions
import terrashift.evaluation.thresholds as thresholds
import terrashift.evaluation.unique_regions as unique_regions
import terrashift.geospatial.geometry as geometry
import terrashift.geospatial.imagery as imagery

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "terrashift"


class PackageResidenceTests(unittest.TestCase):
    def test_evaluation_modules_live_in_the_package(self):
        for module, expected in {
            prediction: "evaluation/prediction.py",
            thresholds: "evaluation/thresholds.py",
            sessions: "evaluation/sessions.py",
            lattice: "evaluation/lattice.py",
            unique_regions: "evaluation/unique_regions.py",
        }.items():
            self.assertEqual(pathlib.Path(module.__file__).resolve(),
                             (PACKAGE / expected).resolve(), module.__name__)

    def test_geospatial_modules_live_in_the_package(self):
        for module, expected in {geometry: "geospatial/geometry.py",
                                 imagery: "geospatial/imagery.py"}.items():
            self.assertEqual(pathlib.Path(module.__file__).resolve(),
                             (PACKAGE / expected).resolve(), module.__name__)

    def test_legacy_utils_modules_are_gone(self):
        for legacy in ("utils/geometry.py", "utils/imagery.py", "utils/lattice_metrics.py",
                       "utils/unique_metrics.py"):
            self.assertFalse((REPO / legacy).exists(), "duplicate left at %s" % legacy)

    def test_quantize_shares_the_package_implementation(self):
        # A re-export must be the same object, not a second definition.
        self.assertIs(quantize.evaluate_sessions, sessions.evaluate_sessions)
        self.assertIs(quantize.metrics_at_thresholds, thresholds.metrics_at_thresholds)
        self.assertIs(quantize.predict_pair_onnx, prediction.predict_pair_onnx)


class ThresholdMetricsTests(unittest.TestCase):
    """Six synthetic pixels whose confusion counts are countable by hand."""

    SCORES = np.array([0.04, 0.05, 0.10, 0.49, 0.50, 0.90])
    LABELS = np.array([0, 1, 0, 1, 1, 1])

    def setUp(self):
        self.out = thresholds.metrics_at_thresholds(self.SCORES, self.LABELS,
                                                    (0.05, 0.10, 0.50))

    def test_counts_at_exactly_the_threshold_are_positive(self):
        # score 0.05 with label 1 counts as a true positive at threshold 0.05
        at = self.out["thresholds"]["0.05"]
        self.assertEqual((at["tp"], at["fp"], at["fn"], at["tn"]), (4.0, 1.0, 0.0, 1.0))

    def test_below_threshold_pixel_is_a_false_positive(self):
        # 0.04 (negative) is excluded at 0.05; 0.10 (negative) becomes a false positive there
        at = self.out["thresholds"]["0.10"]
        self.assertEqual((at["tp"], at["fp"], at["fn"], at["tn"]), (3.0, 1.0, 1.0, 1.0))

    def test_above_threshold_pixels_are_dropped_from_the_positive_set(self):
        at = self.out["thresholds"]["0.50"]
        self.assertEqual((at["tp"], at["fp"], at["fn"], at["tn"]), (2.0, 0.0, 2.0, 2.0))

    def test_precision_and_recall_are_derivable_from_the_counts(self):
        at = self.out["thresholds"]["0.05"]
        self.assertAlmostEqual(at["precision"], at["tp"] / (at["tp"] + at["fp"]), places=6)
        self.assertAlmostEqual(at["recall"], at["tp"] / (at["tp"] + at["fn"]), places=6)

    def test_pixel_and_positive_totals_are_reported(self):
        self.assertEqual(self.out["pixels"], 6)
        self.assertEqual(self.out["positives"], 4)

    def test_ap_and_auc_are_present_and_bounded(self):
        self.assertTrue(0.0 <= self.out["ap"] <= 1.0)
        self.assertTrue(0.0 <= self.out["auc"] <= 1.0)


class RegisterContractTests(unittest.TestCase):
    """Read-only checks of the registers that already exist on disk."""

    REGISTERS = {
        "outputs/register/detections.csv": 26,
        "outputs/register/reviewed_cases_v2/reviewed_cases.csv": 38,
        "outputs/register/candidate_provenance.csv": 52,
    }

    def _rows(self, relative):
        path = REPO / relative
        if not path.is_file():
            self.skipTest("%s not present" % relative)
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_record_counts_are_unchanged(self):
        for relative, expected in self.REGISTERS.items():
            self.assertEqual(len(self._rows(relative)), expected, relative)

    def test_primary_identifiers_are_unique(self):
        for relative, key in (("outputs/register/detections.csv", "detection_id"),
                              ("outputs/register/reviewed_cases_v2/reviewed_cases.csv", "case_id"),
                              ("outputs/register/candidate_provenance.csv", "source_candidate_id")):
            rows = self._rows(relative)
            identifiers = [row[key] for row in rows]
            self.assertEqual(len(set(identifiers)), len(identifiers),
                             "%s has duplicate %s" % (relative, key))

    def test_class_encodings_are_the_reviewer_vocabulary(self):
        rows = self._rows("outputs/register/reviewed_cases_v2/reviewed_cases.csv")
        statuses = {row["decision_status"] for row in rows}
        self.assertTrue(statuses <= {"accepted_man_made", "rejected_natural",
                                     "rejected_non_target", "rejected_acquisition",
                                     "uncertain", "unassessable"}, statuses)

    def test_area_semantics_column_is_present_and_consistent(self):
        rows = self._rows("outputs/register/reviewed_cases_v2/reviewed_cases.csv")
        for row in rows:
            self.assertIn(row["area_semantics"],
                          {"annotated_change_area_of_reviewed_component",
                           "model_flagged_area_within_reviewed_no_change"})
            if row["area_semantics"] == "annotated_change_area_of_reviewed_component":
                self.assertTrue(row["area_m2"])
            else:
                self.assertTrue(row["flagged_area_m2"])


if __name__ == "__main__":
    unittest.main()
