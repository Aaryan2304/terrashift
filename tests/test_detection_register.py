"""Tests for the detection register validator.

Every rule in ``configs/detection_register_schema.yaml`` gets a positive and a
negative case, using synthetic rows only. No real detection is created.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.registers.build_detection_register import validate_row

MANIFEST = {
    "products": [
        {"id": "p1", "path": "/nonexistent/p1.SAFE"},
        {"id": "p2", "path": "/nonexistent/p2.SAFE"},
    ],
    "pairs": {"A": {"tile": "T43QBB", "crs": "EPSG:32643",
                    "date1": "p1", "date2": "p2"}},
}

GEOMETRY = ("POLYGON ((500000 3000000, 500100 3000000, 500100 2999900, "
            "500000 2999900, 500000 3000000))")


def _row(**overrides):
    row = {
        "register_id": "A-0001", "pair": "A", "tile": "T43QBB",
        "date1": "2022-12-23", "date2": "2023-05-12",
        "geometry_wkt": GEOMETRY, "crs": "EPSG:32643",
        "longitude": "73.85", "latitude": "19.02",
        "area_m2": "10000", "area_ha": "1.0",
        "change_type": "construction", "confidence": "high",
        "status": "confirmed", "detection_source": "manual_only",
        "evidence": "new building visible in the later image",
        "before_image": "before.png", "after_image": "after.png",
        "reviewer": "reviewer-1", "reviewed_at": "2026-11-15",
    }
    row.update(overrides)
    return row


class RegisterValidationTests(unittest.TestCase):
    def test_a_valid_confirmed_row_passes(self):
        geometry = validate_row(_row(), MANIFEST, set())
        self.assertEqual(geometry.geom_type, "Polygon")

    def test_missing_column_is_rejected(self):
        row = _row()
        del row["evidence"]
        with self.assertRaisesRegex(ValueError, "missing required column"):
            validate_row(row, MANIFEST, set())

    def test_duplicate_identifier_is_rejected(self):
        seen = {"A-0001"}
        with self.assertRaisesRegex(ValueError, "duplicate register_id"):
            validate_row(_row(), MANIFEST, seen)

    def test_wrong_tile_for_the_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "names tile"):
            validate_row(_row(tile="T43RGM"), MANIFEST, set())

    def test_wrong_crs_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "has CRS"):
            validate_row(_row(crs="EPSG:32644"), MANIFEST, set())

    def test_unknown_change_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "change_type"):
            validate_row(_row(change_type="roadworks"), MANIFEST, set())

    def test_confirmed_without_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "evidence note"):
            validate_row(_row(evidence="short"), MANIFEST, set())

    def test_candidate_must_not_carry_a_change_type(self):
        with self.assertRaisesRegex(ValueError, "must not carry change_type"):
            validate_row(_row(status="candidate", change_type="construction"),
                         MANIFEST, set())

    def test_candidate_with_unknown_type_is_allowed(self):
        validate_row(_row(status="candidate", change_type="unknown",
                          evidence="systematic shoreline sweep, not yet reviewed"),
                     MANIFEST, set())

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "geometry is"):
            validate_row(_row(geometry_wkt="LINESTRING (0 0, 1 1)"), MANIFEST, set())

    def test_invalid_date_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reviewed_at"):
            validate_row(_row(reviewed_at="last tuesday"), MANIFEST, set())

    def test_non_numeric_area_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-numeric area_m2"):
            validate_row(_row(area_m2="about a hectare"), MANIFEST, set())


if __name__ == "__main__":
    unittest.main()
