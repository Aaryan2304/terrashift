"""Tests for the OSCD training instrumentation and index safeguards.

Covers the additions made for the GPU training run: per-region dataset
filtering, the nvidia-smi sampler, and the idempotent patch-index writer.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from terrashift.data.oscd_original import (DEFAULT_BANDS, OSCDOriginalPatchDataset,
                                write_patch_index)
from train_oscd_original import GpuSampler, epoch_memory

FIELDS = ["city", "split", "row_off", "col_off", "height", "width",
          "image1_dir", "image2_dir", "label_path", "label_encoding"]


def _row(city, split, row_off, col_off=0):
    return {"city": city, "split": split, "row_off": row_off, "col_off": col_off,
            "height": 128, "width": 128, "image1_dir": "/a", "image2_dir": "/b",
            "label_path": "/c", "label_encoding": "observed_1_no_change_2_change"}


def _write_index(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class RegionFilterTests(unittest.TestCase):
    def test_region_filter_selects_only_named_cities(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "index.csv"
            _write_index(index, [
                _row("bordeaux", "validation", 0),
                _row("bordeaux", "validation", 128),
                _row("pisa", "validation", 0),
                _row("paris", "train", 0),
            ])
            dataset = OSCDOriginalPatchDataset(
                index, "validation", DEFAULT_BANDS,
                mean=[0.1] * len(DEFAULT_BANDS), std=[0.1] * len(DEFAULT_BANDS),
                regions=["bordeaux"])
            self.assertEqual(len(dataset), 2)
            self.assertEqual({row["city"] for row in dataset.rows}, {"bordeaux"})

    def test_region_filter_rejects_a_city_outside_the_split(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "index.csv"
            _write_index(index, [_row("bordeaux", "validation", 0),
                                 _row("paris", "train", 0)])
            with self.assertRaisesRegex(ValueError, "absent from the validation split"):
                OSCDOriginalPatchDataset(
                    index, "validation", DEFAULT_BANDS,
                    mean=[0.1] * len(DEFAULT_BANDS), std=[0.1] * len(DEFAULT_BANDS),
                    regions=["paris"])

    def test_no_filter_keeps_every_split_row(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "index.csv"
            _write_index(index, [_row("bordeaux", "validation", 0),
                                 _row("pisa", "validation", 0),
                                 _row("paris", "train", 0)])
            dataset = OSCDOriginalPatchDataset(
                index, "validation", DEFAULT_BANDS,
                mean=[0.1] * len(DEFAULT_BANDS), std=[0.1] * len(DEFAULT_BANDS))
            self.assertEqual(len(dataset), 2)


class GpuSamplerTests(unittest.TestCase):
    def test_missing_nvidia_smi_does_not_raise(self):
        with mock.patch("subprocess.check_output", side_effect=OSError("no nvidia-smi")):
            with GpuSampler(interval_s=0.05) as sampler:
                time.sleep(0.2)
        summary = sampler.summary()
        self.assertEqual(summary["gpu_samples"], 0)
        self.assertIsNone(summary["gpu_util_mean_pct"])
        self.assertIsNone(summary["gpu_memory_used_max_mib"])

    def test_samples_are_parsed_and_summarised(self):
        output = "37, 1234\n"
        with mock.patch("subprocess.check_output", return_value=output):
            with GpuSampler(interval_s=0.05) as sampler:
                time.sleep(0.25)
        summary = sampler.summary()
        self.assertGreater(summary["gpu_samples"], 0)
        self.assertAlmostEqual(summary["gpu_util_mean_pct"], 37.0)
        self.assertAlmostEqual(summary["gpu_util_max_pct"], 37.0)
        self.assertAlmostEqual(summary["gpu_memory_used_max_mib"], 1234.0)

    def test_epoch_memory_reports_both_keys(self):
        memory = epoch_memory()
        self.assertIn("peak_allocated_gib", memory)
        self.assertIn("peak_reserved_gib", memory)
        for value in memory.values():
            self.assertTrue(value is None or isinstance(value, float))


class PatchIndexWriteTests(unittest.TestCase):
    def test_second_write_with_identical_rows_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.csv"
            rows = [_row("paris", "train", 0), _row("paris", "train", 128)]
            self.assertEqual(write_patch_index(rows, path, 128, 128), "written")
            before = path.read_bytes()
            before_meta = path.with_suffix(".json").read_bytes()
            self.assertEqual(write_patch_index(rows, path, 128, 128), "unchanged")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.with_suffix(".json").read_bytes(), before_meta)

    def test_different_rows_are_refused_without_touching_the_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.csv"
            write_patch_index([_row("paris", "train", 0)], path, 128, 128)
            before = path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_patch_index([_row("paris", "train", 256)], path, 128, 128)
            self.assertEqual(path.read_bytes(), before)

    def test_missing_sidecar_is_rewritten_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.csv"
            rows = [_row("paris", "train", 0)]
            write_patch_index(rows, path, 128, 128)
            path.with_suffix(".json").unlink()
            self.assertEqual(write_patch_index(rows, path, 128, 128),
                             "metadata_rewritten")
            self.assertTrue(path.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
