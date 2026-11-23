"""Tests for the GeoPackage -> GeoJSON export used by the rasteriser.

The rasteriser only accepts RFC 7946 GeoJSON in WGS84. GDAL writes a legacy
``crs`` member into GeoJSON output, which is not RFC 7946; the exporter removes
it. These tests pin that and the schema property order.
"""

import unittest

from scripts.review.export_manual_labels import PROPERTY_ORDER, to_payload

GEOMETRY = {"type": "Polygon", "coordinates": [[[72.8, 19.0], [72.9, 19.0],
                                                [72.9, 19.1], [72.8, 19.0]]]}


def props(**overrides):
    base = {"record_id": "A_1", "pair": "A", "label_class": "change",
            "reviewer": "Aaryan", "reviewed_at": "2026-11-17T08:00:00+05:30",
            "evidence_source": "before_after_visual", "evidence": "note",
            "confidence": "high"}
    base.update(overrides)
    return base


class PayloadTests(unittest.TestCase):
    def test_no_legacy_crs_member(self):
        payload = to_payload([(props(), GEOMETRY)])
        self.assertNotIn("crs", payload)
        self.assertEqual(payload["type"], "FeatureCollection")

    def test_properties_follow_the_schema_order(self):
        payload = to_payload([(props(), GEOMETRY)])
        self.assertEqual(list(payload["features"][0]["properties"]), PROPERTY_ORDER)

    def test_unexpected_properties_are_kept_but_last(self):
        payload = to_payload([(props(extra="kept"), GEOMETRY)])
        keys = list(payload["features"][0]["properties"])
        self.assertIn("extra", keys)
        self.assertEqual(keys[-1], "extra")

    def test_geometry_passes_through_unchanged(self):
        payload = to_payload([(props(), GEOMETRY)])
        self.assertEqual(payload["features"][0]["geometry"], GEOMETRY)

    def test_feature_count_and_name(self):
        payload = to_payload([(props(), GEOMETRY), (props(record_id="A_2"), GEOMETRY)],
                             name="manual_labels_A")
        self.assertEqual(len(payload["features"]), 2)
        self.assertEqual(payload["name"], "manual_labels_A")

    def test_empty_input_is_allowed(self):
        payload = to_payload([])
        self.assertEqual(payload["features"], [])


if __name__ == "__main__":
    unittest.main()
