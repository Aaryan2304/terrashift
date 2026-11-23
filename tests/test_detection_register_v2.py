"""Validation against detection-register-v2.

v2 keeps the reviewer's vocabulary and identifies rows by location_id, so the v1
rules cannot be reused unchanged. These tests pin the v2 contract.
"""

import unittest
from pathlib import Path

from scripts.registers.build_detection_register import validate_row, validate_row_v2
from terrashift.data.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = None


def setUpModule():
    """The manifest expands SENTINEL_DATA_ROOT, so these tests need it set."""
    global MANIFEST
    try:
        MANIFEST = load_manifest(ROOT / "configs" / "data_manifest.yaml")
    except ValueError as exc:
        raise unittest.SkipTest("data manifest unavailable: %s" % exc)


def _row(**overrides):
    row = {
        "detection_id": "TS-A_L01", "register_id": "A-0001", "location_id": "A_L01",
        "pair_id": "A", "tile": "T43QBB",
        "source_candidate_ids": "TA_005_c016;TA_050_c016",
        "source_thresholds": "0.05;0.50",
        "acquisition_date_before": "2022-12-23", "acquisition_date_after": "2023-05-12",
        "change_type": "building_construction", "decision_status": "accepted_man_made",
        "confidence": "high", "crs": "EPSG:32643",
        "geometry_wkt": "POLYGON ((303310 2114800, 303340 2114800, 303340 2114770, 303310 2114770, 303310 2114800))",
        "longitude": "303320.0", "latitude": "2114785.0",
        "area_pixels": "600", "area_m2": "60000.0", "area_ha": "6.0",
        "evidence_before": "artifacts/evidence/chips/A_L01_before.png",
        "evidence_after": "artifacts/evidence/chips/A_L01_after.png",
        "overlay_path": "artifacts/evidence/overlays/A_L01_panel.png",
        "decision_reason": "A new rectangular roof footprint appears in the later image.",
        "feature_geometry_class": "unclassified", "detection_source": "model",
        "reviewer": "Aaryan", "reviewed_at": "2026-11-22",
    }
    row.update(overrides)
    return row


class V2RowTests(unittest.TestCase):
    def test_accepts_a_reviewer_vocabulary_row(self):
        geometry = validate_row(_row(), MANIFEST, set())
        self.assertAlmostEqual(geometry.area, 900.0)

    def test_dispatched_from_validate_row_by_pair_id(self):
        self.assertIsNotNone(validate_row(_row(), MANIFEST, set()))

    def test_rejects_an_unknown_decision(self):
        with self.assertRaises(ValueError):
            validate_row_v2(_row(decision_status="accepted"), MANIFEST, set())

    def test_rejects_missing_required_column(self):
        row = _row()
        row.pop("decision_reason")
        with self.assertRaises(ValueError):
            validate_row_v2(row, MANIFEST, set())

    def test_rejects_duplicate_location(self):
        with self.assertRaises(ValueError):
            validate_row_v2(_row(), MANIFEST, {"A_L01"})

    def test_rejects_wrong_tile(self):
        with self.assertRaises(ValueError):
            validate_row_v2(_row(tile="T43RGM"), MANIFEST, set())


class ShippedRegisterTests(unittest.TestCase):
    def test_the_published_register_validates(self):
        register = ROOT / "artifacts" / "register" / "detections.csv"
        if not register.is_file():
            self.skipTest("published register is not present")
        import csv
        rows = list(csv.DictReader(register.open(newline="", encoding="utf-8")))
        seen = set()
        for row in rows:
            validate_row(row, MANIFEST, seen)
        self.assertEqual(len(seen), len(rows))


if __name__ == "__main__":
    unittest.main()
