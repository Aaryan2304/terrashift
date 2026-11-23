"""Tests for the single-source water thresholds and the class encoding."""

from __future__ import annotations

import unittest

import numpy as np

from terrashift.preprocessing.water import classify, thresholds_from_config, water_masks


def _cfg(ndwi=0.0, nir=0.15):
    return {"preprocess": {"thresholds": {"water_ndwi": ndwi, "water_nir_max": nir}}}


class ThresholdConfigTests(unittest.TestCase):
    def test_reads_the_configured_values(self):
        thresholds = thresholds_from_config(_cfg(0.0, 0.15))
        self.assertEqual(thresholds["ndwi_threshold"], 0.0)
        self.assertEqual(thresholds["nir_threshold"], 0.15)

    def test_values_match_the_project_configuration(self):
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        thresholds = thresholds_from_config(cfg)
        self.assertEqual(thresholds["ndwi_threshold"],
                         float(cfg["preprocess"]["thresholds"]["water_ndwi"]))
        self.assertEqual(thresholds["nir_threshold"],
                         float(cfg["preprocess"]["thresholds"]["water_nir_max"]))

    def test_out_of_range_ndwi_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "water_ndwi"):
            thresholds_from_config(_cfg(ndwi=1.5))

    def test_non_positive_nir_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "water_nir_max"):
            thresholds_from_config(_cfg(nir=0.0))

    def test_missing_key_is_reported_clearly(self):
        with self.assertRaisesRegex(ValueError, "missing a water threshold"):
            thresholds_from_config({"preprocess": {"thresholds": {}}})


class ClassifyTests(unittest.TestCase):
    def _water(self, shape=(4, 4)):
        ndwi1 = np.full(shape, -0.5, dtype=np.float32)
        ndwi2 = np.full(shape, -0.5, dtype=np.float32)
        ndwi1[0, 0] = 0.5      # water on both dates -> stable
        ndwi2[0, 0] = 0.5
        ndwi1[1, 0] = 0.5      # water on date 1 only -> dynamic
        ndwi1[2, 0] = np.nan   # invalid observation
        return water_masks(ndwi1, ndwi2)

    def test_classes_are_encoded_as_documented(self):
        water = self._water()
        classes = classify(water)
        self.assertEqual(int(classes[0, 0]), 2)   # stable water
        self.assertEqual(int(classes[1, 0]), 3)   # dynamic shoreline
        self.assertEqual(int(classes[2, 0]), 0)   # invalid
        self.assertEqual(int(classes[3, 3]), 1)   # land
        self.assertEqual(set(np.unique(classes).tolist()) <= {0, 1, 2, 3}, True)

    def test_observation_validity_forces_invalid_class(self):
        water = self._water()
        valid = np.ones(water["land"].shape, dtype=bool)
        valid[3, 3] = False
        classes = classify(water, observation_valid=valid)
        self.assertEqual(int(classes[3, 3]), 0)
        self.assertEqual(int(classes[0, 0]), 2)

    def test_shape_mismatch_is_rejected(self):
        water = self._water()
        with self.assertRaisesRegex(ValueError, "must match"):
            classify(water, observation_valid=np.ones((2, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
