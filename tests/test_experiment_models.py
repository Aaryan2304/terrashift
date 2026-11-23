"""Tests for the experiment-only model additions.

The baseline model and its loss are unchanged by these additions; what is tested
here is the new focal term and the ResNet-18 variant used in the comparison.
"""

import unittest

import torch

from terrashift.models.siamese_resnet18 import SiameseResNet18
from terrashift.models.siamese_unet import DiceBCELoss, FocalLoss, SiameseUNet


class FocalLossTests(unittest.TestCase):
    def test_gamma_zero_equals_weighted_bce(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 1, 8, 8)
        target = (torch.rand(2, 1, 8, 8) > 0.7).float()
        focal = FocalLoss(gamma=0.0)(logits, target)
        bce = DiceBCELoss(bce_weight=1.0, dice_weight=0.0)(logits, target)
        self.assertAlmostEqual(float(focal), float(bce), places=5)

    def test_positive_weight_is_applied(self):
        logits = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        plain = FocalLoss(gamma=2.0)(logits, target)
        weighted = FocalLoss(gamma=2.0, pos_weight=10.0)(logits, target)
        self.assertGreater(float(weighted), float(plain))

    def test_easy_examples_are_down_weighted(self):
        # A confident correct prediction should cost less than an uncertain one.
        easy_logits = torch.full((1, 1, 4, 4), 6.0)
        hard_logits = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        loss = FocalLoss(gamma=2.0)
        self.assertLess(float(loss(easy_logits, target)), float(loss(hard_logits, target)))

    def test_valid_mask_excludes_pixels(self):
        logits = torch.zeros(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        target[..., 2:] = 1.0
        mask = torch.zeros(1, 1, 4, 4)
        mask[..., :2] = 1.0            # only the zeros are valid
        loss_masked = float(FocalLoss(gamma=0.0)(logits, target, valid_mask=mask))
        only_zeros = float(FocalLoss(gamma=0.0)(logits, torch.zeros_like(target)))
        self.assertAlmostEqual(loss_masked, only_zeros, places=5)

    def test_all_invalid_returns_zero_gradient_value(self):
        logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
        target = torch.zeros(1, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        loss = FocalLoss()(logits, target, valid_mask=mask)
        self.assertEqual(float(loss), 0.0)


class ResNetSiameseTests(unittest.TestCase):
    def test_forward_shape_matches_input(self):
        model = SiameseResNet18(in_channels=4, pretrained=False).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 4, 128, 128), torch.randn(1, 4, 128, 128))
        self.assertEqual(tuple(out.shape), (1, 1, 128, 128))

    def test_odd_input_size_still_matches(self):
        model = SiameseResNet18(in_channels=4, pretrained=False).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 4, 96, 96), torch.randn(1, 4, 96, 96))
        self.assertEqual(tuple(out.shape), (1, 1, 96, 96))

    def test_stem_accepts_four_bands(self):
        model = SiameseResNet18(in_channels=4, pretrained=False)
        self.assertEqual(model.stem[0].in_channels, 4)

    def test_identical_inputs_give_near_zero_difference(self):
        # Absolute-difference fusion: equal dates must produce the same feature
        # stack, so the head sees zeros everywhere.
        model = SiameseResNet18(in_channels=4, pretrained=False).eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            a1, a2, a3, a4 = model._encode(x)
            b1, b2, b3, b4 = model._encode(x)
        for left, right in ((a1, b1), (a2, b2), (a3, b3), (a4, b4)):
            self.assertLess(float((left - right).abs().max()), 1e-5)

    def test_parameter_budget(self):
        resnet = SiameseResNet18(in_channels=4, pretrained=False).parameter_count()
        baseline = SiameseUNet(in_channels=4, base=16).parameter_count()
        self.assertGreater(resnet, baseline)
        self.assertLess(resnet / baseline, 40,
                        "the alternative must stay in the same order of magnitude")


if __name__ == "__main__":
    unittest.main()
