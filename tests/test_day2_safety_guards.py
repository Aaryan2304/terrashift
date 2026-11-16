"""Regression tests for Day 2 data-split and output safety guards."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
import torch

from data.local_patches import write_local_patch_index
from data.manifest import load_manifest
from data.oscd_original import validate_region_splits
from scripts.export_review_candidates import export_candidates
from scripts.materialize_local_patches import materialize
from scripts.prepare_local_partitions import build_pair
from scripts.rasterize_manual_labels import rasterize_annotations
from train import main as train_main
from train_local_manual import run_training as train_local
from utils.inference import sliding_window_predict


class Day2SafetyGuardTests(unittest.TestCase):
    def test_manifest_expands_sentinel_data_root_and_rejects_unset_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                "products:\n  - id: demo\n    path: ${SENTINEL_DATA_ROOT}/demo.SAFE\n"
                "pairs: {}\n",
                encoding="utf-8")
            with mock.patch.dict(os.environ, {"SENTINEL_DATA_ROOT": str(root)}):
                manifest = load_manifest(manifest_path)
            self.assertEqual(manifest["products"][0]["path"], str(root / "demo.SAFE"))
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "SENTINEL_DATA_ROOT"):
                    load_manifest(manifest_path)

    def test_inference_includes_probabilities_equal_to_threshold(self):
        class ZeroLogitModel(torch.nn.Module):
            def forward(self, a, b):
                return torch.zeros((a.shape[0], 1, a.shape[2], a.shape[3]), device=a.device)

        image = torch.zeros(1, 8, 8)
        probabilities, mask = sliding_window_predict(
            ZeroLogitModel(), image, image.clone(), patch_size=8, stride=8,
            device=torch.device("cpu"), threshold=0.5)
        self.assertTrue((probabilities == 0.5).all())
        self.assertIsNotNone(mask)
        self.assertTrue(mask.all())

    def test_exporter_emits_rfc7946_wgs84_without_legacy_crs_member(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "candidates.csv"
            csv_path.write_text(
                "pair,component_id,priority_percentile,score_threshold,area_px,"
                "bbox_min_easting,bbox_min_northing,bbox_max_easting,"
                "bbox_max_northing,mean_score,max_score\n"
                "A,1,99.9,4.0,1,500000,0,500020,20,3.0,4.0\n",
                encoding="utf-8")
            output = root / "candidates.geojson"
            self.assertEqual(export_candidates(
                csv_path, output, "EPSG:32643", expected_pair="A"), 1)
            collection = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotIn("crs", collection)
            self.assertEqual(len(collection["features"]), 1)
            feature = collection["features"][0]
            self.assertEqual(
                feature["properties"]["review_status"],
                "unreviewed_candidate_not_ground_truth")
            ring = feature["geometry"]["coordinates"][0]
            self.assertTrue(all(-180 <= point[0] <= 180 and -90 <= point[1] <= 90
                                for point in ring))

    def test_exporter_preserves_existing_candidate_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "candidates.csv"
            csv_path.write_text(
                "pair,component_id,priority_percentile,score_threshold,area_px,"
                "bbox_min_easting,bbox_min_northing,bbox_max_easting,"
                "bbox_max_northing,mean_score,max_score\n"
                "A,1,99.9,4.0,1,500000,0,500020,20,3.0,4.0\n",
                encoding="utf-8")
            output = root / "candidates.geojson"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                export_candidates(csv_path, output, "EPSG:32643", expected_pair="A")
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_local_index_rejects_test_rows_without_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "index.csv"
            with self.assertRaisesRegex(ValueError, "include_test"):
                write_local_patch_index(
                    [{"split": "test", "partition_code": 3}],
                    output, 32, 32, 0.05, include_test=False)
            self.assertFalse(output.exists())

    def test_local_index_rejects_split_code_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "partition code"):
                write_local_patch_index(
                    [{"split": "train", "partition_code": 3}],
                    Path(temp) / "index.csv", 32, 32, 0.05)

    def test_local_index_writer_preserves_existing_csv_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "index.csv"
            metadata = output.with_suffix(".json")
            output.write_text("keep csv", encoding="utf-8")
            metadata.write_text("keep json", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_local_patch_index([], output, 32, 16, 0.05)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep csv")
            self.assertEqual(metadata.read_text(encoding="utf-8"), "keep json")

    def test_partition_writer_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            target = output / "pair_A_partition_20m.tif"
            target.write_bytes(b"keep")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                build_pair("A", {}, output / "missing-score.tif", output, 500, 0.7, 0.15)
            self.assertEqual(target.read_bytes(), b"keep")

    def test_materializer_refuses_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "patches"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                materialize(Path("missing.csv"), Path("missing.yaml"),
                            Path("missing.json"), output)
            self.assertEqual(marker.read_text(), "keep")

    def test_rasterizer_refuses_to_overwrite_manual_labels(self):
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [75.0, 19.0], [75.001, 19.0], [75.001, 19.001],
                [75.0, 19.001], [75.0, 19.0]]]},
            "properties": {
                "record_id": "r1", "pair": "A", "label_class": "change",
                "reviewer": "test", "reviewed_at": "2026-11-14",
                "evidence_source": "before_after_visual",
                "evidence": "synthetic review fixture", "confidence": "high"}
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "labels.geojson"
            labels.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
            output = root / "outputs"
            output.mkdir()
            target = output / "manual_labels_A.tif"
            target.write_bytes(b"keep")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                rasterize_annotations("A", labels, output, root / "missing-parts", root / "missing-scores")
            self.assertEqual(target.read_bytes(), b"keep")

    def test_oscd_split_validation_rejects_duplicate_regions(self):
        config = {
            "official_train": ["train_a", "train_b"],
            "official_test": ["test_a"],
            "train": ["train_a", "train_a"],
            "validation": ["train_b"],
            "test": ["test_a"],
            "excluded": {},
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "splits.yaml"
            config_path.write_text(yaml.safe_dump(config))
            city_lists = [["train_a", "train_b"], ["test_a"], ["train_a", "train_b", "test_a"]]
            with mock.patch("data.oscd_original._city_list", side_effect=city_lists):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    validate_region_splits(root, config_path)

    def test_local_trainer_rejects_an_index_that_read_test_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "index.csv"
            index.write_text("")
            index.with_suffix(".json").write_text(json.dumps({
                "test_index_included": True,
                "test_label_pixels_read": True,
                "uses_human_labels_only": True,
                "uses_osm_labels": False,
                "test_labels_used_for_training": False,
                "buffer_or_cross_split_windows_included": False,
                "patch_size": 256, "stride": 128,
            }))
            with self.assertRaisesRegex(ValueError, "without reading test labels"):
                train_local(
                    {"data": {"local_patch_index": str(index)}},
                    SimpleNamespace(init_checkpoint=None), temp, 1, 1, 4, 1e-3,
                    0, 0.5, [42], None)

    def test_local_trainer_rejects_unsafe_index_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "index.csv"
            index.write_text("")
            index.with_suffix(".json").write_text(json.dumps({
                "test_index_included": False,
                "test_label_pixels_read": False,
                "uses_human_labels_only": True,
                "uses_osm_labels": True,
                "test_labels_used_for_training": False,
                "buffer_or_cross_split_windows_included": False,
                "patch_size": 256, "stride": 128,
            }))
            with self.assertRaisesRegex(ValueError, "human-labeled"):
                train_local(
                    {"data": {"local_patch_index": str(index)}},
                    SimpleNamespace(init_checkpoint=None), temp, 1, 1, 4, 1e-3,
                    0, 0.5, [42], None)

    def test_local_trainer_rejects_patch_size_not_divisible_by_eight(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "index.csv"
            index.write_text("")
            index.with_suffix(".json").write_text(json.dumps({
                "test_index_included": False,
                "test_label_pixels_read": False,
                "uses_human_labels_only": True,
                "uses_osm_labels": False,
                "test_labels_used_for_training": False,
                "buffer_or_cross_split_windows_included": False,
                "patch_size": 250, "stride": 128,
            }))
            with self.assertRaisesRegex(ValueError, "divisible by 8"):
                train_local(
                    {"data": {"local_patch_index": str(index)}},
                    SimpleNamespace(init_checkpoint=None), temp, 1, 1, 4, 1e-3,
                    0, 0.5, [42], None)

    def test_training_refuses_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {
                "data": {"dataset": "oscd_original", "oscd_original_root": str(root)},
                "training": {"epochs": 1, "batch_size": 1, "lr": 0.001},
                "experiment": {"name": "test", "output_root": str(root), "seed": 1, "num_workers": 0},
                "model": {"base_filters": 4},
                "evaluation": {"threshold": 0.5},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep")
            with mock.patch.object(sys, "argv", [
                    "train.py", "--config", str(config_path), "--dataset", "oscd_original",
                    "--output-dir", str(output)]):
                with self.assertRaises(SystemExit) as raised:
                    train_main()
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
