"""Tests for the quantisation hardening pass.

Four corrections: macro AP/AUC, component identity by area rather than rank, dynamic
axes on the output, and a parity gate before quantisation. Nothing here exports,
calibrates or quantises.
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
import torch

import quantize
from terrashift.data.validation_windows import IGNORE_VALUE

ROOT = Path(__file__).resolve().parents[1]


def _load_comparison_module():
    spec = importlib.util.spec_from_file_location(
        "compare_quantized_validation",
        ROOT / "scripts" / "evaluation" / "compare_quantized_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _MiniDataset:
    """One window over a small synthetic target."""

    def __init__(self, target):
        self.target = np.asarray(target)
        self.valid = self.target != IGNORE_VALUE

    def __len__(self):
        return 1

    def __getitem__(self, index):
        height, width = self.target.shape
        return {"image1": torch.zeros(4, height, width),
                "image2": torch.zeros(4, height, width),
                "valid_mask": torch.from_numpy(self.valid),
                "row_off": 0, "col_off": 0,
                "core_offset": 0, "core_count": height}


class _ConstantSession:
    """Returns one constant probability for every pixel."""

    def __init__(self, value):
        self.value = float(value)

    def run(self, names, feed):
        # session_probabilities passes names=None, so key off the feed
        key = "image1" if "image1" in feed else next(iter(feed))
        batch, _, height, width = feed[key].shape
        return [np.full((batch, 1, height, width), self.value, dtype=np.float32)]


def _target(shapes):
    target = np.full((12, 12), IGNORE_VALUE, dtype=np.uint8)
    for rows, cols, value in shapes:
        target[rows, cols] = value
    return target


class MacroAggregationTests(unittest.TestCase):
    """Macro AP/AUC are unweighted pair means; pooled micro is untouched."""

    def _evaluate(self):
        # Pair A: 10 reviewed pixels, 5 positive.  Pair B: 30 reviewed pixels, 6 positive.
        target_a = _target([(slice(0, 1), slice(0, 5), 1), (slice(1, 2), slice(0, 5), 0)])
        target_b = _target([(slice(0, 1), slice(0, 5), 1), (slice(1, 2), slice(0, 1), 1),
                            (slice(1, 2), slice(1, 5), 0), (slice(2, 6), slice(0, 5), 0)])
        entries = {
            "A": {"target": target_a, "col_offset": 0, "dataset": _MiniDataset(target_a)},
            "B": {"target": target_b, "col_offset": 0, "dataset": _MiniDataset(target_b)},
        }
        session = _ConstantSession(0.9)
        return quantize.evaluate_sessions({"fp32": (session, ["image1", "image2"])},
                                          entries, [0.5])["fp32"]

    def test_macro_ap_is_the_mean_of_the_pair_aps(self):
        result = self._evaluate()
        expected = np.mean([result["pairs"]["A"]["ap"], result["pairs"]["B"]["ap"]])
        self.assertAlmostEqual(result["macro"]["ap"], float(expected), places=6)

    def test_macro_auc_is_the_mean_of_the_pair_aucs(self):
        result = self._evaluate()
        expected = np.mean([result["pairs"]["A"]["auc"], result["pairs"]["B"]["auc"]])
        self.assertAlmostEqual(result["macro"]["auc"], float(expected), places=6)

    def test_macro_records_how_many_pairs_were_averaged(self):
        self.assertEqual(self._evaluate()["macro"]["pairs_averaged"], 2)

    def test_pooled_micro_is_unchanged_and_differs_from_macro(self):
        result = self._evaluate()
        self.assertEqual(result["pooled"]["pixels"], 40)
        # different pixel counts per pair, so micro and macro must not coincide
        self.assertNotAlmostEqual(result["pooled"]["ap"], result["macro"]["ap"], places=4)

    def test_threshold_metrics_moved_under_macro_thresholds(self):
        result = self._evaluate()
        self.assertIn("0.50", result["macro"]["thresholds"])
        self.assertIn("recall", result["macro"]["thresholds"]["0.50"])

    def _summary(self):
        """Minimal payload with the shape the real comparison produces."""
        def metrics(ap, auc):
            return {"ap": ap, "auc": auc, "pixels": 10, "positives": 4,
                    "thresholds": {"0.50": {"tp": 1, "fp": 1, "fn": 3, "tn": 5,
                                            "precision": 0.5, "recall": 0.25, "f1": 0.333,
                                            "iou": 0.2,
                                            "predicted_positive_fraction": 0.2}}}
        def component(recall, status="verified"):
            return {"rows": [{"component": 8, "pixels": 10749, "recall": recall,
                              "rank_by_area": 1}],
                    "groups": {"small_lt_200px": {"components": 1, "pixels": 10,
                                                  "pixel_weighted_recall": 0.4}},
                    "known_component": {"status": status, "pixels": 10749,
                                        "recall": recall, "component_id": 8}}
        detection = {}
        for model in ("fp32", "int8"):
            detection[model] = {
                "pairs": {"A": metrics(0.9, 0.95), "B": metrics(0.7, 0.75)},
                "pooled": {"ap": 0.8, "auc": 0.85, "pixels": 20},
                "macro": {"ap": 0.8, "auc": 0.85},
            }
        differences = {pair: {"pixels": 10, "mean_abs_probability_difference": 0.01,
                              "max_abs_probability_difference": 0.2,
                              "p95_abs_difference": 0.05, "p99_abs_difference": 0.1,
                              "pearson_r": 1.0,
                              "decision_change_fraction": {"0.50": 0.1},
                              "decision_changes": {"0.50": 1}}
                       for pair in ("A", "B")}
        components = {pair: {"0.50": {"fp32": component(0.5), "int8": component(0.45)}}
                      for pair in ("A", "B")}
        return {
            "models": {"fp32": "/q/onboard_fp32.onnx", "int8": "/q/onboard_int8.onnx"},
            "provider": "CPUExecutionProvider", "optimisation": "ORT_ENABLE_BASIC",
            "pairs": ["A", "B"], "thresholds": [0.5],
            "threshold_status": "fixed comparison points; not validated",
            "patch_size": 256, "stride": 128,
            "normalisation": "/n/normalisation.json", "normalisation_sha256": "abc",
            "detection": detection,
            "ap_delta_pooled": -0.01, "auc_delta_pooled": -0.02,
            "differences": differences, "components": components,
            "sizes": {"fp32_onnx_bytes": 100, "int8_onnx_bytes": 28,
                      "reduction_bytes": 72, "reduction_percent": 72.0,
                      "int8_as_percent_of_fp32": 28.0,
                      "fp32_sha256": "f", "int8_sha256": "i",
                      "fp32_matches_quant_report": True,
                      "int8_matches_quant_report": True,
                      "checkpoint_bytes": 200, "checkpoint_sha256": "c"},
            "latency": {"256x1": {"cpu_fp32": {"median_ms": 10.0, "min_ms": 9.0, "runs": 20},
                                  "cpu_int8": {"median_ms": 12.0, "min_ms": 11.0, "runs": 20},
                                  "int8_over_fp32_median": 1.2},
                        "256x8": {"cpu_fp32": {"median_ms": 70.0, "min_ms": 65.0, "runs": 20},
                                  "cpu_int8": {"median_ms": 90.0, "min_ms": 85.0, "runs": 20},
                                  "int8_over_fp32_median": 1.29}},
            "peak_rss_mib": 1000.0,
        }

    def test_markdown_shows_macro_and_deltas(self):
        module = _load_comparison_module()
        text = module.render_markdown(self._summary())
        self.assertIn("AP macro", text)
        self.assertIn("AUC macro", text)
        self.assertIn("FP32 remains the preferred model", text)
        self.assertIn("No CPU speedup is claimed", text)

    def test_markdown_flags_an_identity_mismatch(self):
        module = _load_comparison_module()
        summary = self._summary()
        summary["components"]["B"]["0.50"]["int8"]["known_component"]["status"] = "id_mismatch"
        text = module.render_markdown(summary)
        self.assertIn("Identity mismatch", text)


class DynamicAxisTests(unittest.TestCase):
    """Batch, height and width must be dynamic on the output as well."""

    def test_every_tensor_declares_the_three_axes(self):
        axes = quantize.dynamic_axes_for(["image1", "image2"])
        self.assertEqual(set(axes), {"image1", "image2", "probability"})
        for mapping in axes.values():
            self.assertEqual(mapping, {0: "batch", 2: "height", 3: "width"})

    def test_axis_names_are_consistent_across_tensors(self):
        axes = quantize.dynamic_axes_for(["image1", "image2"])
        self.assertEqual(axes["image1"], axes["probability"])
        self.assertEqual(axes["image2"], axes["probability"])

    def test_output_name_matches_the_export(self):
        self.assertEqual(quantize.OUTPUT_NAME, "probability")


class ExportContractVerificationTests(unittest.TestCase):
    """The post-export runtime check, exercised against fakes."""

    class _Session:
        def __init__(self, value, shape_override=None):
            self.value = value
            self.shape_override = shape_override

        def get_inputs(self):
            return [type("I", (), {"name": "image1"})(),
                    type("I", (), {"name": "image2"})(),]

        def get_outputs(self):
            return [type("O", (), {"name": "probability"})()]

        def run(self, names, feed):
            key = "image1" if "image1" in feed else next(iter(feed))
            batch, _, height, width = feed[key].shape
            shape = self.shape_override or (batch, 1, height, width)
            return [np.full(shape, self.value, dtype=np.float32)]

    def test_accepts_a_correct_graph(self):
        result = quantize.verify_export_contract(self._Session(0.5),
                                                 ["image1", "image2"],
                                                 ["B02", "B03", "B04", "B08"])
        self.assertTrue(result["passed"])
        self.assertTrue(result["batches"]["batch_1"]["shape_ok"])
        self.assertTrue(result["batches"]["batch_8"]["shape_ok"])
        self.assertTrue(result["batches"]["batch_8"]["in_probability_range"])

    def test_rejects_a_wrong_output_shape(self):
        result = quantize.verify_export_contract(self._Session(0.5, (1, 1, 128, 128)),
                                                 ["image1", "image2"],
                                                 ["B02", "B03", "B04", "B08"])
        self.assertFalse(result["passed"])

    def test_rejects_probabilities_outside_the_unit_interval(self):
        result = quantize.verify_export_contract(self._Session(2.5),
                                                 ["image1", "image2"],
                                                 ["B02", "B03", "B04", "B08"])
        self.assertFalse(result["passed"])


class ParityGateTests(unittest.TestCase):
    """Quantisation must not start on an unproven conversion."""

    def _report(self, **parity):
        return {"stages": {"parity": parity}}

    def test_blocks_when_parity_was_skipped(self):
        with self.assertRaises(SystemExit):
            quantize.check_parity_gate({"stages": {}})

    def test_blocks_when_the_difference_exceeds_tolerance(self):
        with self.assertRaises(SystemExit):
            quantize.check_parity_gate(self._report(max_abs_difference=2e-5,
                                                    non_finite_torch=0,
                                                    non_finite_onnx=0,
                                                    scores_match=True))

    def test_blocks_on_non_finite_predictions(self):
        with self.assertRaises(SystemExit):
            quantize.check_parity_gate(self._report(max_abs_difference=1e-9,
                                                    non_finite_torch=3,
                                                    non_finite_onnx=0,
                                                    scores_match=True))

    def test_blocks_when_the_score_ordering_differs(self):
        with self.assertRaises(SystemExit):
            quantize.check_parity_gate(self._report(max_abs_difference=1e-9,
                                                    non_finite_torch=0,
                                                    non_finite_onnx=0,
                                                    scores_match=False))

    def test_blocks_on_a_non_finite_difference(self):
        with self.assertRaises(SystemExit):
            quantize.check_parity_gate(self._report(max_abs_difference=float("nan"),
                                                    non_finite_torch=0,
                                                    non_finite_onnx=0,
                                                    scores_match=True))

    def test_passes_and_records_the_measurement(self):
        result = quantize.check_parity_gate(self._report(max_abs_difference=3e-7,
                                                         non_finite_torch=0,
                                                         non_finite_onnx=0,
                                                         scores_match=True))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["tolerance"], 1e-5)
        self.assertAlmostEqual(result["max_abs_difference"], 3e-7)

    def test_tolerance_is_exactly_one_e_minus_five(self):
        self.assertEqual(quantize.PARITY_TOLERANCE, 1e-5)
        self.assertTrue(math.isfinite(quantize.PARITY_TOLERANCE))

    def test_boundary_value_is_accepted(self):
        result = quantize.check_parity_gate(self._report(max_abs_difference=1e-5,
                                                         non_finite_torch=0,
                                                         non_finite_onnx=0,
                                                         scores_match=True))
        self.assertEqual(result["status"], "passed")


class ComponentIdentityTests(unittest.TestCase):
    """Component 8 is found by area, not by rank."""

    def setUp(self):
        self.module = _load_comparison_module()

    def _rows(self, component, pixels):
        return [{"component": component, "pixels": pixels, "recall": 0.75,
                 "rank_by_area": 1, "mean_score": 0.5, "median_score": 0.5}]

    def test_verifies_the_audited_identity(self):
        result = self.module.identify_known_component(self._rows(8, 10749), "B")
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["component_id"], 8)
        self.assertEqual(result["pixels"], 10749)

    def test_reports_an_id_mismatch_instead_of_relabelling(self):
        result = self.module.identify_known_component(self._rows(3, 10749), "B")
        self.assertEqual(result["status"], "id_mismatch")
        self.assertEqual(result["found_id"], 3)

    def test_reports_when_the_size_is_absent(self):
        result = self.module.identify_known_component(self._rows(8, 9000), "B")
        self.assertEqual(result["status"], "not_identified")

    def test_pair_a_has_no_audited_identity(self):
        result = self.module.identify_known_component(self._rows(8, 10749), "A")
        self.assertEqual(result["status"], "not_applicable")

    def test_known_component_constant_matches_the_audit(self):
        self.assertEqual(quantize.KNOWN_COMPONENT,
                         {"pair": "B", "component_id": 8, "pixels": 10749})


if __name__ == "__main__":
    unittest.main()
