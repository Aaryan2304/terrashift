"""Checkpoint retention: last.pt every epoch, best.pt only on improvement.

Simulates epochs directly through the saving function the trainer calls, with tiny
state dicts, so the policy is tested without loading data or a model.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch

from train_local_manual import (per_pair_log_values, save_epoch_checkpoints,
                               write_checkpoint_atomic)


def payload(epoch, value):
    return {"epoch": epoch, "model_state_dict": {"w": torch.full((2,), float(value))},
            "optimizer_state_dict": {"step": epoch}, "bands": ["B02", "B03", "B04", "B08"],
            "mean": [0.0] * 4, "std": [1.0] * 4, "base": 16, "seed": 42,
            "pos_weight": 0.834, "precision": "bf16", "val_metrics": {"ap": value}}


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ckpt_"))

    def files(self):
        return sorted(p.name for p in self.root.glob("*.pt"))

    def test_last_tracks_the_most_recent_epoch(self):
        best_ap = -1.0
        for epoch, ap in ((1, 0.40), (2, 0.60), (3, 0.50)):
            best_ap = save_epoch_checkpoints(self.root, payload(epoch, ap), ap, best_ap)[0]
            last = torch.load(self.root / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(last["epoch"], epoch)
            self.assertAlmostEqual(last["val_metrics"]["ap"], ap)

    def test_best_tracks_the_highest_fitness(self):
        best_ap = -1.0
        for epoch, ap in ((1, 0.40), (2, 0.60), (3, 0.50)):
            best_ap = save_epoch_checkpoints(self.root, payload(epoch, ap), ap, best_ap)[0]
        best = torch.load(self.root / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(best["epoch"], 2)
        self.assertAlmostEqual(best["val_metrics"]["ap"], 0.60)

    def test_non_improving_epoch_does_not_touch_best(self):
        best_ap, _, _, improved = save_epoch_checkpoints(self.root, payload(1, 0.60), 0.60, -1.0)
        self.assertTrue(improved)
        best_before = (self.root / "best.pt").stat().st_mtime_ns
        _, _, _, improved = save_epoch_checkpoints(self.root, payload(2, 0.50), 0.50, best_ap)
        self.assertFalse(improved)
        self.assertEqual((self.root / "best.pt").stat().st_mtime_ns, best_before)
        best = torch.load(self.root / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(best["epoch"], 1)

    def test_equal_fitness_is_not_an_improvement(self):
        best_ap = save_epoch_checkpoints(self.root, payload(1, 0.70), 0.70, -1.0)[0]
        _, _, _, improved = save_epoch_checkpoints(self.root, payload(2, 0.70), 0.70, best_ap)
        self.assertFalse(improved)
        self.assertEqual(torch.load(self.root / "best.pt", weights_only=True)["epoch"], 1)

    def test_non_finite_fitness_never_becomes_best(self):
        best_ap, _, _, improved = save_epoch_checkpoints(self.root, payload(1, float("nan")),
                                                   float("nan"), -1.0)
        self.assertFalse(improved)
        self.assertEqual(best_ap, -1.0)
        self.assertFalse((self.root / "best.pt").exists())
        self.assertTrue((self.root / "last.pt").exists())

    def test_only_two_checkpoint_files_exist(self):
        best_ap = -1.0
        for epoch in range(1, 7):
            best_ap = save_epoch_checkpoints(self.root, payload(epoch, epoch / 10),
                                                epoch / 10, best_ap)[0]
        self.assertEqual(self.files(), ["best.pt", "last.pt"])

    def test_both_checkpoints_load_with_the_full_schema(self):
        best_ap = save_epoch_checkpoints(self.root, payload(1, 0.5), 0.5, -1.0)[0]
        for name in ("last.pt", "best.pt"):
            # last.pt holds RNG state and needs the full unpickler; best.pt is
            # weights_only-compatible because inference loads it that way.
            state = torch.load(self.root / name, map_location="cpu",
                               weights_only=(name == "best.pt"))
            for key in ("epoch", "model_state_dict", "optimizer_state_dict", "bands",
                        "mean", "std", "base", "seed", "pos_weight", "precision",
                        "val_metrics"):
                self.assertIn(key, state, "%s missing %s" % (name, key))

    def test_atomic_write_leaves_no_temporary_file(self):
        target = self.root / "x.pt"
        write_checkpoint_atomic(payload(1, 0.1), target)
        self.assertTrue(target.is_file())
        self.assertFalse(any(p.name.endswith(".tmp") for p in self.root.iterdir()))

    def test_atomic_write_replaces_an_existing_file(self):
        target = self.root / "y.pt"
        write_checkpoint_atomic(payload(1, 0.1), target)
        write_checkpoint_atomic(payload(2, 0.2), target)
        self.assertEqual(torch.load(target, weights_only=True)["epoch"], 2)

    def test_original_oscd_checkpoint_is_never_targeted(self):
        import inspect

        import train_local_manual
        from terrashift.training import checkpointing

        # The checkpoint policy moved into the package in Task 6, so the literals are checked
        # where they now live; the entry point must still never name the OSCD checkpoint.
        source = inspect.getsource(checkpointing)
        self.assertIn('"last.pt"', source)
        self.assertIn('"best.pt"', source)
        self.assertNotIn("oscd_gpu_v1", source)
        self.assertNotIn("oscd_gpu_v1", inspect.getsource(train_local_manual))


class PerPairLogTests(unittest.TestCase):
    def test_values_are_lifted_from_per_pair(self):
        metrics = {"ap": 0.7, "per_pair": {
            "A": {"ap": 0.55, "reviewed_pixels": 2759},
            "B": {"ap": 0.76, "reviewed_pixels": 22650}}}
        values = per_pair_log_values(metrics)
        self.assertAlmostEqual(values["val_ap_A"], 0.55)
        self.assertAlmostEqual(values["val_ap_B"], 0.76)
        self.assertEqual(values["val_reviewed_A"], 2759)
        self.assertEqual(values["val_reviewed_B"], 22650)

    def test_missing_pairs_give_blanks_not_errors(self):
        values = per_pair_log_values({"ap": 0.7})
        self.assertEqual(set(values), {"val_ap_A", "val_ap_B",
                                       "val_reviewed_A", "val_reviewed_B"})
        self.assertTrue(all(v is None for v in values.values()))

    def test_one_missing_pair_is_tolerated(self):
        values = per_pair_log_values({"per_pair": {"A": {"ap": 0.5, "reviewed_pixels": 10}}})
        self.assertAlmostEqual(values["val_ap_A"], 0.5)
        self.assertIsNone(values["val_ap_B"])

    def test_log_fields_include_the_pair_columns_and_keep_aggregates(self):
        import inspect
        import train_local_manual

        source = inspect.getsource(train_local_manual)
        for column in ("val_ap_A", "val_ap_B", "val_reviewed_A", "val_reviewed_B",
                       "val_ap", "val_loss", "train_loss", "epoch"):
            self.assertIn('"%s"' % column, source, "missing column %s" % column)

    def test_config_points_at_the_v2_index(self):
        import yaml

        with open("config.yaml", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        self.assertEqual(cfg["data"]["local_patch_index"],
                         "./outputs/dataset/v2/train_index.csv")


if __name__ == "__main__":
    unittest.main()
