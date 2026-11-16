"""Contract tests for the normalisation positive-control comparison.

The metrics are the point of the experiment, so they get their own tests: a
correction that preserves change must report preservation 1.0, and one that
suppresses change must report below 1. Synthetic arrays only.
"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_normalization_positive_controls import evaluate_method
from utils.metrics import roc_auc


CHANGE = 0.04


def _scene(seed=0, size=64):
    """One band, a stable background, a changed block with headroom left in AUROC."""
    rng = np.random.default_rng(seed)
    ref = (0.2 + 0.02 * rng.standard_normal((size, size))).astype(np.float32)
    mov = (ref + 0.02 * rng.standard_normal((size, size))).astype(np.float32)
    labels = np.zeros((size, size), dtype=np.int64)
    block = (slice(10, 26), slice(10, 26))
    labels[block] = 1
    # A moderate reflectance increase: AUROC is about 0.85, so a correction can
    # move it up or down instead of saturating at 1.0.
    mov[block] += CHANGE
    return [ref], [mov], labels


def _score(ref, mov):
    from preprocess.normalize import reference_scaled_change_score
    valid = np.ones(ref[0].shape, dtype=bool)
    score, _ = reference_scaled_change_score(ref, mov, valid)
    return score


class PositiveControlMetricTests(unittest.TestCase):
    def test_unchanged_input_reports_preservation_one(self):
        ref, mov, labels = _scene()
        valid = np.ones(labels.shape, dtype=bool)
        before = _score(ref, mov)
        result = evaluate_method("none", ref, [m.copy() for m in mov], labels,
                                 valid, before)
        self.assertAlmostEqual(result["preservation"], 1.0, places=9)
        self.assertAlmostEqual(result["delta_auroc"], 0.0, places=9)
        self.assertGreater(result["auroc_after"], 0.5)
        self.assertEqual(result["n_change_pixels"], int(labels.sum()))

    def test_suppressed_change_lowers_preservation(self):
        ref, mov, labels = _scene()
        valid = np.ones(labels.shape, dtype=bool)
        before = _score(ref, mov)
        suppressed = [m.copy() for m in mov]
        block = (slice(10, 26), slice(10, 26))
        # Halve the change amplitude the way an over-aggressive correction would.
        suppressed[0][block] = mov[0][block] - CHANGE / 2.0
        result = evaluate_method("histogram", ref, suppressed, labels, valid, before)
        self.assertLess(result["preservation"], 1.0)
        self.assertLess(result["delta_auroc"], 0.0)

    def test_amplified_change_raises_auroc(self):
        ref, mov, labels = _scene()
        valid = np.ones(labels.shape, dtype=bool)
        before = _score(ref, mov)
        boosted = [m.copy() for m in mov]
        block = (slice(10, 26), slice(10, 26))
        boosted[0][block] = mov[0][block] + CHANGE / 2.0
        result = evaluate_method("global", ref, boosted, labels, valid, before)
        self.assertGreater(result["preservation"], 1.0)
        self.assertGreater(result["delta_auroc"], 0.0)

    def test_auroc_matches_the_documented_definition(self):
        ref, mov, labels = _scene(seed=3)
        valid = np.ones(labels.shape, dtype=bool)
        before = _score(ref, mov)
        result = evaluate_method("none", ref, [m.copy() for m in mov], labels,
                                 valid, before)
        expected = roc_auc(labels.ravel(), before.ravel())
        self.assertAlmostEqual(result["auroc_before"], expected, places=12)


if __name__ == "__main__":
    unittest.main()
