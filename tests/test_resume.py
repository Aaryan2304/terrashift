"""Epoch-boundary resume: interruption, restart, and the things that must not change.

The mini trainer here mirrors ``train_local_manual``'s structure, a CSV row and a
checkpoint per epoch, RNG-sensitive shuffling, without touching the dataset.
"""

from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train_local_manual import save_epoch_checkpoints
from terrashift.training.resume import (capture_rng_state, check_compatibility,
                          repair_best_checkpoint, restore_rng_state,
                          resume_from_checkpoint, truncate_train_log)

FIELDS = ["epoch", "val_ap", "val_ap_A", "val_ap_B"]


def resolved_config(normalisation_sha="sha-aaa"):
    return {
        "patch_index": "/data/train_index.csv", "patch_size": 256,
        "patch_stride": 128, "normalisation_sha256": normalisation_sha,
        "initialization_checkpoint": "/ckpt/oscd_best.pt",
        "initialization_mode": "checkpoint", "base_filters": 16, "batch_size": 4,
        "lr": 1e-4, "pos_weight": 0.834, "bands": ["B02", "B03", "B04", "B08"],
        "loss": {"type": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
        "precision": {"autocast_dtype": "bf16"}, "augmentation": {"enabled": False},
    }


class MiniTrainer:
    """Two parameters, one AdamW step per epoch, RNG draw per epoch."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.log = self.run_dir / "train_log.csv"

    def epoch(self, number, val_ap, best_ap, best_epoch, best_metrics, rng_draw=None):
        # A real step so optimizer state is non-trivial.
        self.optimizer.zero_grad()
        loss = (self.model(torch.ones(4, 2)) ** 2).mean()
        loss.backward()
        self.optimizer.step()
        mode = "a" if number > 1 else "w"
        with self.log.open(mode, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if mode == "w":
                writer.writeheader()
            writer.writerow({"epoch": number, "val_ap": val_ap,
                             "val_ap_A": val_ap, "val_ap_B": val_ap})
        payload = {"epoch": number, "model_state_dict": self.model.state_dict(),
                   "optimizer_state_dict": self.optimizer.state_dict(),
                   "val_metrics": {"ap": val_ap}, "bands": ["B02", "B03", "B04", "B08"],
                   "run_config": resolved_config()}
        return save_epoch_checkpoints(self.run_dir, payload, val_ap, best_ap,
                                      best_epoch, best_metrics)

    def resume(self):
        state = resume_from_checkpoint(self.run_dir, device="cpu")
        self.model.load_state_dict(state["checkpoint"]["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["checkpoint"]["optimizer_state_dict"])
        truncate_train_log(self.log, state["start_epoch"] - 1)
        restore_rng_state(state["rng_state"])
        return state


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        self.run_dir = Path(tempfile.mkdtemp(prefix="resume_")) / "seed_42"

    def test_full_interruption_and_resume(self):
        trainer = MiniTrainer(self.run_dir)
        best_ap, best_epoch, best_metrics = -1.0, None, None
        for epoch, ap in ((1, 0.40), (2, 0.60)):
            best_ap, best_epoch, best_metrics, _ = trainer.epoch(
                epoch, ap, best_ap, best_epoch, best_metrics)
        # The checkpoint captured the RNG state at the end of epoch 2, so the first
        # draw after a resume must be the one this draw produces.
        reference_draw = torch.rand(3)

        # Simulate a crash: a log row written for epoch 3 with no checkpoint.
        with trainer.log.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerow(
                {"epoch": 3, "val_ap": 0.99, "val_ap_A": 0.99, "val_ap_B": 0.99})

        model_before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        opt_before = trainer.optimizer.state_dict()["state"]

        # The process restarts. A fresh trainer stands in for it.
        restarted = MiniTrainer(self.run_dir)
        state = restarted.resume()

        self.assertEqual(state["start_epoch"], 3, "training must continue at epoch 3")
        self.assertAlmostEqual(state["best_ap"], 0.60)
        self.assertEqual(state["best_epoch"], 2)
        self.assertAlmostEqual(state["best_metrics"]["ap"], 0.60)

        # Optimizer state restored, not reinitialised.
        self.assertTrue(opt_before, "optimizer had no state to compare")
        restored = restarted.optimizer.state_dict()["state"]
        self.assertEqual(set(restored), set(opt_before))
        for key in opt_before:
            for field, value in opt_before[key].items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, restored[key][field]),
                                    "optimizer field %s not restored" % field)

        # Model weights restored.
        for name, value in model_before.items():
            self.assertTrue(torch.equal(value, restarted.model.state_dict()[name]))

        # RNG restored: the first draw after resuming must match the reference.
        self.assertTrue(torch.equal(torch.rand(3), reference_draw),
                        "torch RNG stream was not restored")

        # The uncommitted epoch 3 row is gone before the loop appends.
        with trainer.log.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(r["epoch"]) for r in rows], [1, 2])
        self.assertNotIn("0.99", [r["val_ap"] for r in rows])

        # Continue to epoch 4.
        best_ap, best_epoch, best_metrics = (state["best_ap"], state["best_epoch"],
                                             state["best_metrics"])
        for epoch, ap in ((3, 0.50), (4, 0.65)):
            best_ap, best_epoch, best_metrics, _ = restarted.epoch(
                epoch, ap, best_ap, best_epoch, best_metrics)

        self.assertEqual(best_epoch, 4, "a later improvement must take over best")
        self.assertAlmostEqual(best_ap, 0.65)

        with trainer.log.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(r["epoch"]) for r in rows], [1, 2, 3, 4])

        self.assertEqual(sorted(p.name for p in self.run_dir.glob("*.pt")),
                         ["best.pt", "last.pt"])

        last = torch.load(self.run_dir / "last.pt", weights_only=False)
        self.assertEqual(last["epoch"], 4)
        self.assertEqual(last["best_epoch"], 4)
        self.assertIn("rng_state", last)

    def test_non_improving_resume_after_crash_keeps_best(self):
        trainer = MiniTrainer(self.run_dir)
        best_ap, best_epoch, best_metrics = -1.0, None, None
        for epoch, ap in ((1, 0.30), (2, 0.70)):
            best_ap, best_epoch, best_metrics, _ = trainer.epoch(
                epoch, ap, best_ap, best_epoch, best_metrics)
        restarted = MiniTrainer(self.run_dir)
        state = restarted.resume()
        self.assertEqual(state["best_epoch"], 2)
        restarted.epoch(3, 0.40, state["best_ap"], state["best_epoch"],
                        state["best_metrics"])
        best = torch.load(self.run_dir / "best.pt", weights_only=True)
        self.assertEqual(best["epoch"], 2, "best.pt must not be replaced by a worse epoch")

    def test_changed_normalisation_is_rejected(self):
        trainer = MiniTrainer(self.run_dir)
        trainer.epoch(1, 0.4, -1.0, None, None)
        state = resume_from_checkpoint(self.run_dir)
        current = resolved_config(normalisation_sha="sha-different")
        with self.assertRaisesRegex(SystemExit, "refusing to resume"):
            check_compatibility(state["checkpoint"]["run_config"], current)

    def test_other_incompatible_settings_are_rejected(self):
        trainer = MiniTrainer(self.run_dir)
        trainer.epoch(1, 0.4, -1.0, None, None)
        saved = resume_from_checkpoint(self.run_dir)["checkpoint"]["run_config"]
        for key, value in (("batch_size", 8), ("lr", 1e-3), ("pos_weight", 2.36),
                           ("patch_size", 128), ("initialization_checkpoint", "/other.pt")):
            current = resolved_config()
            current[key] = value
            with self.subTest(changed=key):
                with self.assertRaises(SystemExit):
                    check_compatibility(saved, current)

    def test_matching_config_passes(self):
        trainer = MiniTrainer(self.run_dir)
        trainer.epoch(1, 0.4, -1.0, None, None)
        saved = resume_from_checkpoint(self.run_dir)["checkpoint"]["run_config"]
        self.assertTrue(check_compatibility(saved, resolved_config()))

    def test_precision_change_is_rejected(self):
        trainer = MiniTrainer(self.run_dir)
        trainer.epoch(1, 0.4, -1.0, None, None)
        saved = resume_from_checkpoint(self.run_dir)["checkpoint"]["run_config"]
        current = resolved_config()
        current["precision"] = {"autocast_dtype": "fp16"}
        with self.assertRaisesRegex(SystemExit, "precision_dtype"):
            check_compatibility(saved, current)

    def test_missing_last_checkpoint_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "does not exist"):
            resume_from_checkpoint(self.run_dir)

    def test_rng_capture_is_round_trippable(self):
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        state = capture_rng_state()
        torch_first, py_first, np_first = torch.rand(5), random.random(), np.random.rand()
        restore_rng_state(state)
        self.assertTrue(torch.equal(torch.rand(5), torch_first))
        self.assertEqual(random.random(), py_first)
        self.assertEqual(np.random.rand(), np_first)
        self.assertIsNone(state["dataloader_generator"])

    def test_crash_between_writes_is_repaired_on_resume(self):
        """last.pt replaced, best.pt not: the recorded best epoch must be recoverable."""
        trainer = MiniTrainer(self.run_dir)
        best_ap, best_epoch, best_metrics = -1.0, None, None
        # Epoch 1 completes normally: both files written, best at epoch 1.
        best_ap, best_epoch, best_metrics, _ = trainer.epoch(
            1, 0.40, best_ap, best_epoch, best_metrics)
        self.assertEqual(best_epoch, 1)

        # Epoch 2 improves. The process dies after last.pt is replaced and before
        # best.pt is: build that intermediate state directly, since the normal save
        # path writes both files.
        stale_best = torch.load(self.run_dir / "best.pt", weights_only=True)
        self.assertEqual(stale_best["epoch"], 1)
        import train_local_manual
        from terrashift.training.resume import capture_rng_state
        train_local_manual.write_checkpoint_atomic(
            {"epoch": 2, "model_state_dict": trainer.model.state_dict(),
             "optimizer_state_dict": trainer.optimizer.state_dict(),
             "val_metrics": {"ap": 0.80}, "best_ap": 0.80, "best_epoch": 2,
             "best_metrics": {"ap": 0.80}, "rng_state": capture_rng_state(),
             "bands": ["B02", "B03", "B04", "B08"], "run_config": resolved_config()},
            self.run_dir / "last.pt")
        # best.pt still holds epoch 1: the crash window.
        self.assertEqual(torch.load(self.run_dir / "best.pt",
                                    weights_only=True)["epoch"], 1)

        state = resume_from_checkpoint(self.run_dir)
        self.assertTrue(state["best_repaired"]["repaired"])
        self.assertEqual(state["best_repaired"]["best_epoch"], 2)
        self.assertEqual(state["start_epoch"], 3)

        repaired = torch.load(self.run_dir / "best.pt", weights_only=True)
        self.assertEqual(repaired["epoch"], 2, "best.pt must be restored to epoch 2")
        self.assertAlmostEqual(repaired["val_metrics"]["ap"], 0.80)
        self.assertNotIn("rng_state", repaired,
                         "best.pt must stay weights_only-loadable")

        # A later non-improving epoch must not undo the repair.
        restarted = MiniTrainer(self.run_dir)
        restarted.model.load_state_dict(state["checkpoint"]["model_state_dict"])
        restarted.optimizer.load_state_dict(state["checkpoint"]["optimizer_state_dict"])
        restarted.epoch(3, 0.10, state["best_ap"], state["best_epoch"],
                        state["best_metrics"])
        still = torch.load(self.run_dir / "best.pt", weights_only=True)
        self.assertEqual(still["epoch"], 2)

    def test_repair_is_a_no_op_when_best_epoch_is_earlier(self):
        trainer = MiniTrainer(self.run_dir)
        best_ap, best_epoch, best_metrics = -1.0, None, None
        for epoch, ap in ((1, 0.70), (2, 0.40)):
            best_ap, best_epoch, best_metrics, _ = trainer.epoch(
                epoch, ap, best_ap, best_epoch, best_metrics)
        state = resume_from_checkpoint(self.run_dir)
        self.assertFalse(state["best_repaired"]["repaired"])
        self.assertEqual(torch.load(self.run_dir / "best.pt", weights_only=True)["epoch"], 1)

    def test_repair_does_not_touch_a_current_best(self):
        trainer = MiniTrainer(self.run_dir)
        best_ap, best_epoch, best_metrics, _ = trainer.epoch(1, 0.60, -1.0, None, None)
        before = (self.run_dir / "best.pt").stat().st_mtime_ns
        result = repair_best_checkpoint(
            self.run_dir,
            torch.load(self.run_dir / "last.pt", weights_only=False))
        self.assertFalse(result["repaired"])
        self.assertEqual((self.run_dir / "best.pt").stat().st_mtime_ns, before)

    def test_cli_allows_resume_into_an_existing_run(self):
        import inspect
        import train

        source = inspect.getsource(train.main)
        # The non-empty-output guard must not block a resume, and must still
        # block a collision.
        self.assertIn("cannot resume: no last.pt for seed", source)
        self.assertIn("if not getattr(args, \"resume\", False):", source)

    def test_cli_exposes_resume(self):
        import inspect
        import train

        source = inspect.getsource(train.main)
        self.assertIn('"--resume"', source)
        self.assertIn("only supported for --dataset local_manual", source)


if __name__ == "__main__":
    unittest.main()
