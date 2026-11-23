"""Tests for the precision policy and the class-balance measurement.

bf16 only, no GradScaler: fp16 autocast produced non-finite gradients on every step
for this model at every loss scale the scaler tried.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from train import resolve_amp_dtype
from train_local_manual import instance_class_counts


class PrecisionTests(unittest.TestCase):
    def test_cuda_with_amp_returns_bfloat16(self):
        with mock.patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertIs(resolve_amp_dtype(torch.device("cuda"), True), torch.bfloat16)

    def test_amp_disabled_returns_none(self):
        self.assertIsNone(resolve_amp_dtype(torch.device("cuda"), False))

    def test_cpu_returns_none(self):
        self.assertIsNone(resolve_amp_dtype(torch.device("cpu"), True))

    def test_unsupported_device_raises_with_a_clear_message(self):
        with mock.patch("torch.cuda.is_bf16_supported", return_value=False):
            with mock.patch("torch.cuda.get_device_name", return_value="Fake GPU"):
                with self.assertRaisesRegex(RuntimeError, "bfloat16"):
                    resolve_amp_dtype(torch.device("cuda"), True)

    def test_never_returns_float16(self):
        with mock.patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertIsNot(resolve_amp_dtype(torch.device("cuda"), True), torch.float16)

    def test_run_epoch_passes_the_dtype_and_has_no_scaling_for_bf16(self):
        import inspect
        import train

        source = inspect.getsource(train.run_epoch)
        self.assertIn("dtype=amp_dtype", source)
        self.assertIn("amp_dtype != torch.bfloat16", source)
        self.assertNotIn("torch.float16", source)


class _FakeDataset:
    """Minimal stand-in returning masks with known positive/negative counts."""

    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        positive, negative, ignored = self.patches[index]
        labels = np.concatenate([np.ones(positive, np.int64),
                                 np.zeros(negative, np.int64)])
        valid = np.concatenate([np.ones(positive + negative, bool),
                                np.zeros(ignored, bool)])
        padded_labels = np.concatenate([labels, np.zeros(ignored, np.int64)])
        return {"mask": torch.from_numpy(padded_labels),
                "valid_mask": torch.from_numpy(valid)}


class ClassCountTests(unittest.TestCase):
    def test_counts_and_rates(self):
        dataset = _FakeDataset([(10, 30, 100), (0, 40, 100), (25, 5, 100)])
        counts = instance_class_counts(dataset)
        self.assertEqual(counts["positive"], 35)
        self.assertEqual(counts["negative"], 75)
        self.assertEqual(counts["ignored"], 300)
        self.assertEqual(counts["all_negative_patches"], 1)
        self.assertEqual(counts["all_positive_patches"], 0)
        self.assertAlmostEqual(counts["pooled_rate"], 35 / 110, places=9)
        # Equal weight per sample: (0.25 + 0 + 0.8333) / 3
        expected = (10 / 40 + 0.0 + 25 / 30) / 3
        self.assertAlmostEqual(counts["mean_sample_rate"], expected, places=9)

    def test_pooled_and_sample_rates_differ_when_patches_differ(self):
        dataset = _FakeDataset([(100, 0, 0), (0, 100, 0)])
        counts = instance_class_counts(dataset)
        self.assertAlmostEqual(counts["pooled_rate"], 0.5, places=9)
        self.assertAlmostEqual(counts["mean_sample_rate"], 0.5, places=9)

        # One all-positive patch and one mostly-negative patch: the pooled rate and
        # the equal-weight-per-sample rate are not the same number.
        dataset = _FakeDataset([(100, 0, 0), (1, 99, 0), (0, 100, 0)])
        counts = instance_class_counts(dataset)
        self.assertAlmostEqual(counts["pooled_rate"], 101 / 300, places=9)
        self.assertAlmostEqual(counts["mean_sample_rate"], (1.0 + 0.01 + 0.0) / 3, places=9)

    def test_ignored_pixels_are_not_counted_as_negative(self):
        dataset = _FakeDataset([(5, 5, 10_000)])
        counts = instance_class_counts(dataset)
        self.assertEqual(counts["positive"], 5)
        self.assertEqual(counts["negative"], 5)
        self.assertEqual(counts["ignored"], 10_000)


class PosWeightTests(unittest.TestCase):
    """The BCE weight follows the reduction, not the pooled pixel counts."""

    def test_configured_value_wins(self):
        from train import resolve_pos_weight

        self.assertAlmostEqual(resolve_pos_weight(0.834, 0.2973), 0.834)

    def test_derived_value_uses_the_pooled_fraction(self):
        from train import resolve_pos_weight

        self.assertAlmostEqual(resolve_pos_weight(None, 0.2973),
                               (1 - 0.2973) / 0.2973, places=9)

    def test_per_sample_and_pooled_rates_give_different_weights(self):
        from train import resolve_pos_weight

        pooled = resolve_pos_weight(None, 0.2973)
        balanced = resolve_pos_weight(None, 0.5453)
        self.assertAlmostEqual(balanced, 0.83385, places=5)
        self.assertGreater(pooled, balanced * 2.5)

    def test_config_carries_the_balanced_weight(self):
        import yaml

        with open("config.yaml", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        self.assertAlmostEqual(cfg["training"]["pos_weight"], 0.834, places=6)

    def test_trainer_reads_the_config_value_and_records_the_derivation(self):
        import inspect
        import train_local_manual

        source = inspect.getsource(train_local_manual)
        self.assertIn('cfg["training"].get("pos_weight")', source)
        self.assertIn("mean_per_sample_positive_fraction", source)
        self.assertIn("pooled_patch_instance_positive_fraction", source)

    def test_log_label_names_the_pooled_quantity(self):
        import inspect
        import train_local_manual

        source = inspect.getsource(train_local_manual)
        self.assertIn("pooled patch-instance positive fraction", source)
        self.assertNotIn("train-only labeled-pixel fraction", source)


class MetadataNamingTests(unittest.TestCase):
    """Metadata must not call patch-instance counts "unique"."""

    def test_old_misleading_keys_are_gone(self):
        import inspect
        import train_local_manual

        source = inspect.getsource(train_local_manual)
        self.assertNotIn("train_unique_labelled_pixels", source)
        self.assertNotIn("train_class_counts_unique", source)
        self.assertNotIn("train_patch_weighted_pixels", source)

    def test_occurrence_keys_are_present(self):
        import inspect
        import train_local_manual

        source = inspect.getsource(train_local_manual)
        for key in ("train_reviewed_pixel_occurrences",
                    "train_patch_pixel_occurrences",
                    "train_class_counts_patch_instances",
                    "train_patch_instance_counts",
                    "count_basis"):
            self.assertIn(key, source, "missing %s" % key)

    def test_cli_exposes_normalisation(self):
        import inspect
        import train

        source = inspect.getsource(train.main)
        self.assertIn('"--normalisation"', source)
        self.assertIn("only valid with --dataset local_manual", source)


if __name__ == "__main__":
    unittest.main()
