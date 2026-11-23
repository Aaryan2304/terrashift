"""The extracted component helpers must be the same objects, and stay in sync."""

import importlib
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.components import (
    KNOWN_COMPONENT,
    STRUCTURE,
    component_recall,
    identify_known_component,
)

ROOT = Path(__file__).resolve().parents[1]


class ExtractionIdentityTests(unittest.TestCase):
    def test_script_reexports_the_package_functions(self):
        script = importlib.import_module("scripts.evaluation.compare_quantized_validation")
        self.assertIs(script.component_recall, component_recall)
        self.assertIs(script.identify_known_component, identify_known_component)

    def test_constant_matches_the_root_cli_copy(self):
        # The package may not import quantize, so the value is duplicated; fail loudly
        # if the two ever disagree instead of letting recall numbers drift.
        import quantize
        self.assertIs(KNOWN_COMPONENT, quantize.KNOWN_COMPONENT)

    def test_module_imports_without_a_dataset(self):
        probe = ("import terrashift.evaluation.components as c; "
                 "assert c.KNOWN_COMPONENT['pixels'] == %d" % KNOWN_COMPONENT["pixels"])
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                              env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        self.assertEqual(done.returncode, 0, done.stderr)


class ComponentRecallSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.target = np.full((6, 6), IGNORE_VALUE, dtype=np.int16)
        self.target[1:3, 1:3] = 1          # 4-pixel change component
        self.target[4:6, 4:6] = 1          # second 4-pixel component
        covered = self.target != IGNORE_VALUE
        self.labels = self.target[covered].astype(np.int64)
        self.scores = np.zeros(int(covered.sum()), dtype=np.float64)

    def test_no_component_when_mask_is_empty(self):
        empty = np.full((4, 4), IGNORE_VALUE, dtype=np.int16)
        result = component_recall(np.zeros(0), np.zeros(0, dtype=np.int64), empty, 0.5)
        self.assertEqual(result, {"components": 0, "rows": [], "groups": {}})

    def test_all_below_threshold_gives_zero_recall(self):
        result = component_recall(self.scores, self.labels, self.target, 0.5)
        self.assertEqual(result["components"], 2)
        self.assertTrue(all(row["recall"] == 0.0 for row in result["rows"]))

    def test_min_size_filters_small_components(self):
        result = component_recall(self.scores, self.labels, self.target, 0.5, min_size=5)
        self.assertEqual(result["components"], 2)
        self.assertEqual(result["rows"], [])

    def test_partial_hit_is_pixel_fraction(self):
        scores = self.scores.copy()
        first = np.nonzero(self.target[self.target != IGNORE_VALUE] == 1)[0]
        scores[first[:4]] = 1.0            # cover exactly one 4-pixel component
        result = component_recall(scores, self.labels, self.target, 0.5)
        recalls = sorted(row["recall"] for row in result["rows"])
        self.assertEqual(recalls, [0.0, 1.0])

    def test_mismatched_ordering_raises(self):
        # Mixed labels so a reversal is genuinely a different ordering: with an
        # all-positive mask the reversed vector would be identical and prove nothing.
        target = np.full((4, 4), IGNORE_VALUE, dtype=np.int16)
        target[0, :] = 1
        target[1, :] = 0
        covered = target != IGNORE_VALUE
        labels = target[covered].astype(np.int64)
        self.assertFalse(np.array_equal(labels, labels[::-1]))
        with self.assertRaises(SystemExit):
            component_recall(np.zeros(labels.size), labels[::-1].copy(), target, 0.5)


class IdentifyKnownComponentTests(unittest.TestCase):
    def _rows(self, component_id, pixels):
        return [{"component": component_id, "pixels": pixels, "recall": 0.5,
                 "mean_score": 0.4, "median_score": 0.4, "rank_by_area": 1}]

    def test_verified_when_area_and_id_both_match(self):
        rows = self._rows(KNOWN_COMPONENT["component_id"], KNOWN_COMPONENT["pixels"])
        self.assertEqual(identify_known_component(rows, KNOWN_COMPONENT["pair"])["status"], "verified")

    def test_id_mismatch_is_reported_not_hidden(self):
        rows = self._rows(KNOWN_COMPONENT["component_id"] + 1, KNOWN_COMPONENT["pixels"])
        result = identify_known_component(rows, KNOWN_COMPONENT["pair"])
        self.assertEqual(result["status"], "id_mismatch")
        self.assertEqual(result["expected_id"], KNOWN_COMPONENT["component_id"])

    def test_no_match_and_wrong_pair(self):
        self.assertEqual(identify_known_component(self._rows(1, 5), KNOWN_COMPONENT["pair"])["status"], "not_identified")
        self.assertEqual(identify_known_component(self._rows(1, 10749), "ZZ")["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
