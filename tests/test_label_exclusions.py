"""Tests for the exclusion manifest and its application.

The manifest exists so three guard-band records can stay in the master
GeoPackage while being kept out of every export. The dangerous failure is a
silent no-op: a typo in a record ID that removes nothing and reports success.
"""

import json
import os
import tempfile
import unittest

from scripts.apply_label_exclusions import apply_exclusions, load_manifest


def manifest(ids, reasons=None):
    return {"version": 1, "pair": "A", "exclusions": [
        {"record_id": record_id,
         "reason": (reasons or {}).get(record_id, "guard_band_overlap"),
         "detail": "test fixture"} for record_id in ids]}


def feature(record_id, pair="A", label="change"):
    return {"type": "Feature",
            "properties": {"record_id": record_id, "pair": pair, "label_class": label},
            "geometry": {"type": "Polygon", "coordinates": []}}


class LoadManifestTests(unittest.TestCase):
    def write(self, payload):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_valid_manifest_loads(self):
        path = self.write(manifest(["A_46"]))
        loaded = load_manifest(path)
        self.assertEqual(len(loaded["exclusions"]), 1)

    def test_missing_list_is_rejected(self):
        path = self.write({"version": 1})
        with self.assertRaisesRegex(ValueError, "no 'exclusions' list"):
            load_manifest(path)

    def test_duplicate_ids_are_rejected(self):
        path = self.write(manifest(["A_46", "A_46"]))
        with self.assertRaisesRegex(ValueError, "more than once"):
            load_manifest(path)

    def test_missing_reason_is_rejected(self):
        path = self.write({"version": 1,
                           "exclusions": [{"record_id": "A_46"}]})
        with self.assertRaisesRegex(ValueError, "no reason"):
            load_manifest(path)


class ApplyExclusionsTests(unittest.TestCase):
    def test_named_records_are_removed_and_the_rest_kept(self):
        features = [feature("A_1"), feature("A_46"), feature("A_2")]
        kept, excluded, unknown = apply_exclusions(features, manifest(["A_46"]))
        self.assertEqual([f["properties"]["record_id"] for f in kept], ["A_1", "A_2"])
        self.assertEqual([e["record_id"] for e in excluded], ["A_46"])
        self.assertEqual(unknown, [])

    def test_reason_and_detail_travel_with_the_record(self):
        features = [feature("A_58")]
        _, excluded, _ = apply_exclusions(features, manifest(["A_58"]))
        self.assertEqual(excluded[0]["reason"], "guard_band_overlap")
        self.assertIn("detail", excluded[0])

    def test_label_class_is_reported_from_the_source_not_the_manifest(self):
        features = [feature("A_51", label="no_change")]
        _, excluded, _ = apply_exclusions(features, manifest(["A_51"]))
        self.assertEqual(excluded[0]["label_class"], "no_change")

    def test_absent_id_is_reported_as_unknown(self):
        features = [feature("A_1")]
        kept, excluded, unknown = apply_exclusions(features, manifest(["A_999"]))
        self.assertEqual(len(kept), 1)
        self.assertEqual(excluded, [])
        self.assertEqual(unknown, ["A_999"])

    def test_pair_mismatch_raises(self):
        features = [feature("A_1", pair="B")]
        with self.assertRaisesRegex(ValueError, "belongs to pair B"):
            apply_exclusions(features, manifest(["A_2"]), pair="A")

    def test_empty_manifest_keeps_everything(self):
        features = [feature("A_1"), feature("A_2")]
        kept, excluded, unknown = apply_exclusions(features, manifest([]))
        self.assertEqual(len(kept), 2)
        self.assertEqual(excluded, [])
        self.assertEqual(unknown, [])

    def test_geometry_is_not_altered(self):
        features = [feature("A_1")]
        features[0]["geometry"]["coordinates"] = [[[0, 0], [1, 0], [1, 1], [0, 0]]]
        kept, _, _ = apply_exclusions(features, manifest(["A_2"]))
        self.assertEqual(kept[0]["geometry"]["coordinates"], [[[0, 0], [1, 0], [1, 1], [0, 0]]])


if __name__ == "__main__":
    unittest.main()
