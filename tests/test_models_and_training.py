"""Contract tests for the migrated models and training helpers.

Synthetic inputs only: no SAFE products, no training run. The checkpoint test loads the
selected artefact read-only and skips when it is absent.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import torch

import terrashift.training.checkpointing as checkpointing
import terrashift.training.config as training_config
import terrashift.training.seeding as seeding
from terrashift.models.siamese_unet import DiceBCELoss, FocalLoss, SiameseUNet

REPO = pathlib.Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"
BANDS = ("B02", "B03", "B04", "B08")


class ModelConstructionTests(unittest.TestCase):
    def test_parameter_count_is_unchanged(self):
        model = SiameseUNet(in_channels=4, base=16)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 482881)

    def test_forward_shape_and_dtype(self):
        model = SiameseUNet(in_channels=4, base=16).eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 4, 64, 64), torch.zeros(2, 4, 64, 64))
        self.assertEqual(tuple(out.shape), (2, 1, 64, 64))
        self.assertEqual(out.dtype, torch.float32)

    def test_forward_is_deterministic_in_eval_mode(self):
        model = SiameseUNet(in_channels=4, base=16).eval()
        a, b = torch.randn(1, 4, 32, 32), torch.randn(1, 4, 32, 32)
        with torch.no_grad():
            first, second = model(a, b), model(a, b)
        self.assertTrue(torch.equal(first, second))

    def test_state_dict_round_trip(self):
        source = SiameseUNet(in_channels=4, base=16).eval()
        target = SiameseUNet(in_channels=4, base=16).eval()
        target.load_state_dict(source.state_dict(), strict=True)
        x = torch.randn(1, 4, 32, 32)
        with torch.no_grad():
            self.assertTrue(torch.equal(source(x, x), target(x, x)))


class SelectedCheckpointTests(unittest.TestCase):
    """The selected checkpoint must load strictly with the migrated model."""

    def setUp(self):
        if not CHECKPOINT.is_file():
            self.skipTest("selected checkpoint not present")

    def test_strict_load_and_forward(self):
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model = SiameseUNet(in_channels=len(payload["bands"]), base=int(payload["base"]))
        result = model.load_state_dict(payload["model_state_dict"], strict=True)
        self.assertEqual(str(result).lower().count("missing"), 0)
        self.assertEqual(str(result).lower().count("unexpected"), 0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 482881)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 4, 64, 64), torch.zeros(1, 4, 64, 64))
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(out).all())


class LossTests(unittest.TestCase):
    def test_bce_dice_is_finite_and_differentiable(self):
        criterion = DiceBCELoss(pos_weight=None)
        logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, :4, :4] = 1.0
        loss = criterion(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_focal_loss_is_finite(self):
        criterion = FocalLoss()
        logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
        loss = criterion(logits, torch.zeros(1, 1, 8, 8))
        self.assertTrue(torch.isfinite(loss))


class TrainingStepTests(unittest.TestCase):
    """One synthetic optimisation step on CPU, small enough to run in a unit test."""

    def test_gradients_flow_and_loss_decreases(self):
        seeding.seed_everything(43)
        model = SiameseUNet(in_channels=4, base=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        criterion = DiceBCELoss()
        x1 = torch.randn(2, 4, 32, 32)
        x2 = x1.clone()
        x2[:, :, 4:12, 4:12] += 1.0
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 4:12, 4:12] = 1.0

        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            loss = criterion(model(x1, x2), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        self.assertTrue(all(isinstance(v, float) for v in losses))
        self.assertLess(losses[-1], losses[0], "three steps should reduce the loss")


class SeedingTests(unittest.TestCase):
    def test_seed_reproduces_tensor_draws(self):
        seeding.seed_everything(7)
        a = torch.randn(3)
        seeding.seed_everything(7)
        b = torch.randn(3)
        self.assertTrue(torch.equal(a, b))


class ConfigTests(unittest.TestCase):
    def test_pos_weight_override_is_used_verbatim(self):
        self.assertEqual(training_config.resolve_pos_weight(0.834, 0.2973), 0.834)

    def test_pos_weight_is_derived_when_absent(self):
        derived = training_config.resolve_pos_weight(None, 0.25)
        self.assertAlmostEqual(derived, 3.0, places=6)

    def test_seed_list_parsing(self):
        self.assertEqual(training_config.parse_seeds("42,43", 42), [42, 43])
        self.assertEqual(training_config.parse_seeds(None, 42), [42])

    def test_load_config_reads_the_repository_config(self):
        config = training_config.load_config(str(REPO / "config.yaml"))
        self.assertIn("training", config)
        self.assertIn("data", config)


class CheckpointRoundTripTests(unittest.TestCase):
    def test_write_and_read_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "last.pt"
            payload = {"epoch": 3, "model_state_dict": {"w": torch.zeros(2)},
                       "rng_state": None}
            checkpointing.write_checkpoint_atomic(payload, path)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(loaded["epoch"], 3)


if __name__ == "__main__":
    unittest.main()
