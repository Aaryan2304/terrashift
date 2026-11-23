"""Tests for the local-dataset quantisation adapter.

Nothing here exports ONNX, calibrates, quantises or benchmarks. The heavy operations
are replaced by fixtures and fakes so the contract can be checked cheaply.
"""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

import quantize
from terrashift.preprocessing.normalisation import load_statistics, pair_statistics
from terrashift.data.validation_windows import IGNORE_VALUE, ValidationLatticeDataset

ROOT = Path(__file__).resolve().parents[1]
HAVE_STATISTICS = (ROOT / "outputs" / "dataset" / "v2" / "normalisation.json").is_file()
from terrashift.evaluation.lattice import LatticeAccumulator

ROOT = Path(__file__).resolve().parents[1]
NORMALISATION = ROOT / "outputs" / "dataset" / "v2" / "normalisation.json"
LABELS = {"A": ROOT / "outputs" / "labels" / "pair_a_v2" / "manual_labels_A.tif",
          "B": ROOT / "outputs" / "labels" / "pair_b_v2" / "manual_labels_B.tif"}
PARTITIONS = {"A": ROOT / "outputs" / "partitions_v2_observation_valid" / "pair_A_partition_10m.tif",
              "B": ROOT / "outputs" / "partitions_v2_observation_valid" / "pair_B_partition_10m.tif"}


def synthetic_rows(counts):
    rows = []
    for pair, total in counts.items():
        rows.extend([{"pair": pair, "row_off": index, "col_off": 0,
                      "height": 256, "width": 256, "split": "train"}
                     for index in range(total)])
    return rows


class CalibrationPlanTests(unittest.TestCase):
    """Selection is train-only, both pairs, exact size and deterministic."""

    def test_selects_exact_batch_times_batches(self):
        rows = synthetic_rows({"A": 122, "B": 220})
        indices, counts = quantize.calibration_plan(rows, 8, 20, seed=43)
        self.assertEqual(len(indices), 160)
        self.assertEqual(sum(counts.values()), 160)
        self.assertEqual(len(set(indices)), 160, "a row was selected twice")

    def test_both_pairs_are_represented(self):
        rows = synthetic_rows({"A": 122, "B": 220})
        _, counts = quantize.calibration_plan(rows, 8, 20, seed=43)
        self.assertEqual(set(counts), {"A", "B"})
        self.assertGreater(counts["A"], 0)
        self.assertGreater(counts["B"], 0)
        # proportional: A is 122/342 of the train split
        self.assertAlmostEqual(counts["A"] / 160, 122 / 342, delta=0.05)

    def test_is_deterministic_for_a_fixed_seed(self):
        rows = synthetic_rows({"A": 122, "B": 220})
        first, _ = quantize.calibration_plan(rows, 8, 20, seed=43)
        second, _ = quantize.calibration_plan(rows, 8, 20, seed=43)
        self.assertEqual(first, second)

    def test_a_different_seed_changes_the_selection(self):
        rows = synthetic_rows({"A": 122, "B": 220})
        first, _ = quantize.calibration_plan(rows, 8, 20, seed=43)
        other, _ = quantize.calibration_plan(rows, 8, 20, seed=7)
        self.assertNotEqual(first, other)

    def test_refuses_to_ask_for_more_rows_than_exist(self):
        rows = synthetic_rows({"A": 10, "B": 10})
        with self.assertRaises(ValueError):
            quantize.calibration_plan(rows, 8, 20)

    def test_single_pair_allocates_everything_to_it(self):
        rows = synthetic_rows({"B": 300})
        indices, counts = quantize.calibration_plan(rows, 8, 4, seed=43)
        self.assertEqual(counts, {"B": 32})
        self.assertEqual(len(indices), 32)


class _FakeDataset:
    """Stands in for LocalManualPatchDataset; records how it was constructed."""

    captured = {}

    def __init__(self, index_path, split, manifest_path, registration_path,
                 normalisation=None, bands=None, mean=None, std=None):
        type(self).captured = {"split": split, "normalisation": normalisation,
                               "mean": mean, "std": std, "bands": bands,
                               "index_path": str(index_path)}
        self.rows = synthetic_rows({"A": 122, "B": 220})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"image1": np.zeros((4, 256, 256), dtype=np.float32),
                "image2": np.zeros((4, 256, 256), dtype=np.float32)}


@unittest.skipUnless(HAVE_STATISTICS, "per-pair statistics are not in the repository")
class LocalCalibrationAdapterTests(unittest.TestCase):
    """The adapter must use the train split and the pair statistics, once each."""

    def setUp(self):
        self._original = quantize.LocalManualPatchDataset
        quantize.LocalManualPatchDataset = _FakeDataset
        self.addCleanup(setattr, quantize, "LocalManualPatchDataset", self._original)
        self.payload = load_statistics(NORMALISATION)
        self.cfg = {"data": {"local_patch_index": "./outputs/dataset/v2/train_index.csv"}}
        self.args = types.SimpleNamespace(manifest="configs/data_manifest.yaml",
                                          registration="unused.json",
                                          batch_size=8, calibration_batches=20,
                                          calibration_seed=43)
        self.checkpoint = {"bands": ["B02", "B03", "B04", "B08"]}

    def test_uses_the_train_split_only(self):
        quantize.make_local_calibration_loader(self.cfg, self.checkpoint, self.args,
                                               self.payload)
        self.assertEqual(_FakeDataset.captured["split"], "train")
        self.assertIn("train_index.csv", _FakeDataset.captured["index_path"])

    def test_uses_pair_statistics_and_not_a_single_mean_std(self):
        quantize.make_local_calibration_loader(self.cfg, self.checkpoint, self.args,
                                               self.payload)
        captured = _FakeDataset.captured
        self.assertIs(captured["normalisation"], self.payload)
        self.assertIsNone(captured["mean"], "checkpoint mean must not also be applied")
        self.assertIsNone(captured["std"], "checkpoint std must not also be applied")

    def test_reports_the_pair_distribution(self):
        _, plan = quantize.make_local_calibration_loader(self.cfg, self.checkpoint,
                                                         self.args, self.payload)
        self.assertEqual(plan["split"], "train")
        self.assertEqual(plan["rows_selected"], 160)
        self.assertEqual(set(plan["per_pair"]), {"A", "B"})
        self.assertEqual(sum(plan["per_pair"].values()), 160)

    def test_pairs_have_different_statistics(self):
        mean_a, std_a = pair_statistics(self.payload, "A")
        mean_b, std_b = pair_statistics(self.payload, "B")
        self.assertFalse(np.allclose(mean_a, mean_b),
                         "the two pairs would be normalised identically")
        self.assertFalse(np.allclose(std_a, std_b))

    def test_normalising_twice_would_change_the_values(self):
        """Documents why the adapter must not pass mean/std as well."""
        from terrashift.preprocessing.normalisation import normalise_reflectance
        mean, std = pair_statistics(self.payload, "A")
        raw = np.full((4, 8, 8), 0.15, dtype=np.float32)
        once = normalise_reflectance(raw, mean, std)
        twice = normalise_reflectance(once, mean, std)
        self.assertFalse(np.allclose(once, twice))


class ValidationWindowTests(unittest.TestCase):
    """The canonical lattice, and one contribution per pixel."""

    def test_lattice_masks_exclude_unreviewed_pixels(self):
        target = np.full((512, 64), IGNORE_VALUE, dtype=np.uint8)
        target[100:110, 10:15] = 1        # 50 reviewed change pixels
        target[300:305, 5:9] = 0          # 20 reviewed no-change pixels, later window

        def loader(row_off, col_off, size):
            return (np.zeros((4, size, size), dtype=np.float32),
                    np.zeros((4, size, size), dtype=np.float32))

        dataset = ValidationLatticeDataset(["A"], target, col_offset=0,
                                           tile_width=64, patch=256, stride=128,
                                           image_loader=loader)
        sample = dataset[0]           # window rows 0..255, so only the first block
        self.assertTrue(torch.is_tensor(sample["valid_mask"]))
        valid = sample["valid_mask"].numpy()
        self.assertEqual(int(valid.sum()), 50)
        self.assertEqual(int((sample["mask"].numpy() == 1).sum()), 50)
        self.assertTrue(np.all(sample["mask"].numpy()[~valid] == 0),
                        "unreviewed pixels must not become negatives")
        later = dataset[2]            # window rows 256..511, holding the no-change block
        self.assertEqual(int(later["valid_mask"].numpy().sum()), 20)
        self.assertEqual(int((later["mask"].numpy() == 1).sum()), 0)

    def test_real_window_counts_match_the_established_geometry(self):
        if not all(path.is_file() for path in list(LABELS.values()) + list(PARTITIONS.values())):
            self.skipTest("label rasters are not present")
        from terrashift.data.validation_windows import read_validation_target, select_windows
        totals = {}
        for pair in ("A", "B"):
            target, col_offset, tile_width = read_validation_target(
                LABELS[pair], PARTITIONS[pair])
            windows = select_windows(target, quantize.PATCH_SIZE, quantize.STRIDE,
                                     block=512, halo=256, tile_width=tile_width,
                                     col_offset=col_offset)
            totals[pair] = len(windows)
            for window in windows:
                self.assertGreaterEqual(window["col0"], 0)
                self.assertLess(window["col0"], tile_width)
        self.assertEqual(totals, {"A": 90, "B": 69})
        self.assertEqual(sum(totals.values()), 159)

    def test_overlapping_windows_count_each_pixel_once(self):
        accumulator = LatticeAccumulator((32, 16), col_offset=0)
        first = np.full((16, 16), 0.2, dtype=np.float32)
        valid = np.ones((16, 16), dtype=bool)
        accumulator.add(0, 0, first, valid)
        accumulator.add(8, 0, np.full((16, 16), 0.4, dtype=np.float32), valid)
        # row 12 is covered by both windows, row 0 only by the first
        self.assertEqual(int(accumulator.count[12, 0]), 2)
        self.assertEqual(int(accumulator.count[0, 0]), 1)
        self.assertAlmostEqual(float(accumulator.total[12, 0] / accumulator.count[12, 0]),
                               0.3, places=6)


class ThresholdAndSessionTests(unittest.TestCase):
    """Fixed thresholds, and no second sigmoid."""

    def test_metrics_use_exactly_the_requested_thresholds(self):
        scores = np.array([0.2, 0.04, 0.05, 0.5, 0.9], dtype=np.float32)
        labels = np.array([1, 1, 0, 0, 1], dtype=np.int64)
        result = quantize.metrics_at_thresholds(scores, labels, [0.05, 0.5])
        self.assertEqual(sorted(result["thresholds"]), ["0.05", "0.50"])
        self.assertEqual(result["positives"], 3)
        # >= 0.05 keeps 4 of 5 pixels, two of them positive
        self.assertEqual(result["thresholds"]["0.05"]["tp"], 2)
        self.assertEqual(result["thresholds"]["0.05"]["fp"], 2)

    def test_no_threshold_is_selected(self):
        scores = np.array([0.1, 0.6], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        result = quantize.metrics_at_thresholds(scores, labels, [0.05])
        self.assertNotIn("best_threshold", result)
        self.assertEqual(sorted(result["thresholds"]), ["0.05"])

    def test_session_output_is_used_as_a_probability(self):
        class FakeSession:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)

            def run(self, names, feed):
                return [self.value]

        probabilities = np.array([[[[0.1, 0.2]]]], dtype=np.float32)
        session = FakeSession(probabilities)
        result = quantize.session_probabilities(session, ["image1", "image2"],
                                                np.zeros((1, 4, 2, 2), np.float32),
                                                np.zeros((1, 4, 2, 2), np.float32))
        np.testing.assert_allclose(result, probabilities[:, 0])
        self.assertTrue(quantize.ONNX_OUTPUT_IS_PROBABILITY)

    def test_wrapper_output_is_a_probability_in_range(self):
        model = quantize.Wrapper(quantize.SiameseUNet(in_channels=4, base=4))
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 4, 64, 64), torch.zeros(1, 4, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)


class ExportContractTests(unittest.TestCase):
    """The export must exercise the deployed 256 patch, not the old 128 dummy."""

    def test_dummy_is_the_deployed_patch_size(self):
        first, second = quantize.export_dummy(["B02", "B03", "B04", "B08"])
        self.assertEqual(tuple(first.shape), (1, 4, 256, 256))
        self.assertEqual(tuple(second.shape), (1, 4, 256, 256))

    def test_patch_and_stride_match_the_training_configuration(self):
        self.assertEqual(quantize.PATCH_SIZE, 256)
        self.assertEqual(quantize.STRIDE, 128)

    def test_benchmark_shapes_cover_both_batch_sizes_at_256(self):
        shapes = quantize.bench_shapes(["B02", "B03", "B04", "B08"])
        self.assertEqual(tuple(shapes["256x1"]["image1"]), (1, 4, 256, 256))
        self.assertEqual(tuple(shapes["256x8"]["image1"]), (8, 4, 256, 256))
        self.assertIn("128x1_diagnostic", shapes)


class OutputSafetyTests(unittest.TestCase):
    """A run must never clobber an existing artefact."""

    def test_refuses_when_a_target_file_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "onboard_int8.onnx"
            target.write_bytes(b"x")
            with self.assertRaises(SystemExit):
                quantize.ensure_new_outputs([target], Path(directory))

    def test_refuses_a_non_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "leftover.partial").write_text("x")
            with self.assertRaises(SystemExit):
                quantize.ensure_new_outputs([], Path(directory))

    def test_allows_a_fresh_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            quantize.ensure_new_outputs([Path(directory) / "new.onnx"], Path(directory))

    def test_overwrite_requires_the_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "onboard_fp32.onnx"
            target.write_bytes(b"x")
            quantize.ensure_new_outputs([target], Path(directory), allow_overwrite=True)

    def test_report_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quantization_report.json"
            quantize.write_json_atomic(path, {"status": "complete"})
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text())["status"], "complete")
            self.assertFalse(path.with_suffix(".json.partial").exists())


class ComponentSensitivityTests(unittest.TestCase):
    """Component recall must come from the 2-D mask, aligned with the score vector."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compare_quantized_validation",
            ROOT / "scripts" / "evaluation" / "compare_quantized_validation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module

    def _fixture(self):
        target = np.zeros((20, 20), dtype=np.uint8)
        target[2:5, 2:6] = 1          # 12 px component
        target[12:13, 10:12] = 1      # 2 px component
        covered = target != IGNORE_VALUE
        rows, cols = np.nonzero(covered)
        labels = target[covered].astype(np.int64)
        scores = np.zeros(labels.size, dtype=np.float32)
        from scipy import ndimage as ndi
        raster, count = ndi.label(covered & (target == 1), structure=np.ones((3, 3)))
        component_of_pixel = raster[rows, cols]
        scores[(component_of_pixel == 1)] = 0.9
        scores[(component_of_pixel == 2)] = 0.2
        return scores, labels, target

    def test_recall_is_per_component_not_per_pixel(self):
        scores, labels, target = self._fixture()
        result = self.module.component_recall(scores, labels, target, 0.5)
        self.assertEqual(result["components"], 2)
        self.assertEqual([row["pixels"] for row in result["rows"]], [12, 2])
        self.assertAlmostEqual(result["rows"][0]["recall"], 1.0, places=6)
        self.assertAlmostEqual(result["rows"][1]["recall"], 0.0, places=6)

    def test_size_groups_are_reported_separately(self):
        scores, labels, target = self._fixture()
        result = self.module.component_recall(scores, labels, target, 0.5)
        self.assertIn("small_lt_200px", result["groups"])
        self.assertEqual(result["groups"]["small_lt_200px"]["components"], 2)
        # pixel-weighted over both components: 12 of 14 px recovered
        self.assertAlmostEqual(result["groups"]["small_lt_200px"]["pixel_weighted_recall"],
                               12 / 14, places=6)

    def test_ordering_mismatch_is_refused(self):
        scores, labels, target = self._fixture()
        with self.assertRaises(SystemExit):
            self.module.component_recall(scores, labels[::-1].copy(), target, 0.5)


if __name__ == "__main__":
    unittest.main()
