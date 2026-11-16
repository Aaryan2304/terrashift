"""FC-Siam-diff-inspired Siamese U-Net for change detection.

Architecture
------------
Two encoders with shared weights process the two dates. At every level the
absolute difference of the feature maps is taken. A U-Net decoder works on
the differences, with skip connections from the same level.

This is the FC-Siam-diff design from Daudt et al., "Fully Convolutional
Siamese Networks for Change Detection", IGARSS 2018,
https://arxiv.org/abs/1810.08462

It is not a reproduction of the original implementation. The original uses a
pretrained backbone and a specific channel layout. This is a smaller model
chosen to fit 4 GB of VRAM.

The output is a single channel of raw logits. Apply sigmoid for a probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SiameseUNet(nn.Module):
    """Siamese U-Net with absolute feature differences.

    Parameters
    ----------
    in_channels : int
        Number of spectral bands per date.
    base : int
        Channels at the first encoder level. The deeper levels use base*2,
        base*4 and base*8.
    """

    def __init__(self, in_channels=13, base=32):
        super().__init__()
        self.enc1 = _block(in_channels, base)
        self.enc2 = _block(base, base * 2)
        self.enc3 = _block(base * 2, base * 4)
        self.enc4 = _block(base * 4, base * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = _block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _block(base * 2, base)

        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def forward(self, x1, x2):
        e1_1, e2_1, e3_1, e4_1 = self._encode(x1)
        e1_2, e2_2, e3_2, e4_2 = self._encode(x2)

        d1 = torch.abs(e1_1 - e1_2)
        d2 = torch.abs(e2_1 - e2_2)
        d3 = torch.abs(e3_1 - e3_2)
        d4 = torch.abs(e4_1 - e4_2)

        u3 = self.up3(d4)
        c3 = self.dec3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(c3)
        c2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, d1], dim=1))
        return self.head(c1)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiceBCELoss(nn.Module):
    """Binary cross-entropy plus soft Dice.

    The Dice term handles the class imbalance: changed pixels are a small
    fraction of the image.

    ``pos_weight`` is the PyTorch ``BCEWithLogitsLoss`` positive-class weight,
    ``N_negative / N_positive``. For illustration, at a 4.85% change rate the
    value is about 19.6; the actual value is computed from the training masks.
    A value greater than 1 increases recall; less than 1 increases precision.

    Know the side effect before you read the probabilities: a large
    ``pos_weight`` inflates them, so 0.5 is no longer a meaningful decision
    threshold. Select the epoch and the threshold on a ranking metric such as
    average precision, not on the probability value.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0,
                 pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        # A registered buffer, so `.to(device)` moves it with the model. A plain
        # CPU tensor would raise a device mismatch on CUDA.
        weight = None if pos_weight is None else torch.tensor([float(pos_weight)])
        self.register_buffer("pos_weight", weight)

    def forward(self, logits, target, valid_mask=None):
        target_f = target.float()
        if valid_mask is None:
            valid_f = torch.ones_like(target_f)
        else:
            valid_f = valid_mask.to(device=logits.device, dtype=target_f.dtype)
            if valid_f.shape != target_f.shape:
                raise ValueError("valid_mask must match target shape")
        dims = tuple(range(1, logits.dim()))
        valid_per_sample = valid_f.sum(dims)
        keep = valid_per_sample > 0
        if not torch.any(keep):
            return logits.sum() * 0.0
        bce_map = F.binary_cross_entropy_with_logits(
            logits, target_f, pos_weight=self.pos_weight, reduction="none")
        bce_per_sample = (bce_map * valid_f).sum(dims) / valid_per_sample.clamp_min(1.0)
        prob = torch.sigmoid(logits)
        inter = (prob * target_f * valid_f).sum(dims)
        denom = ((prob + target_f) * valid_f).sum(dims)
        dice_per_sample = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        return (self.bce_weight * bce_per_sample[keep].mean()
                + self.dice_weight * dice_per_sample[keep].mean())


class FocalLoss(nn.Module):
    """Focal loss over logits, with the same masking contract as DiceBCELoss.

    ``(1 - p_t) ** gamma`` down-weights the easy majority pixels, which is the
    same imbalance problem the Dice term addresses from the other direction.
    ``pos_weight`` is kept so the comparison against BCE is not confounded by
    dropping the class weight.
    """

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = float(gamma)
        weight = None if pos_weight is None else torch.tensor([float(pos_weight)])
        self.register_buffer("pos_weight", weight)

    def forward(self, logits, target, valid_mask=None):
        target_f = target.float()
        valid_f = (torch.ones_like(target_f) if valid_mask is None
                   else valid_mask.to(device=logits.device, dtype=target_f.dtype))
        dims = tuple(range(1, logits.dim()))
        valid_per_sample = valid_f.sum(dims)
        keep = valid_per_sample > 0
        if not torch.any(keep):
            return logits.sum() * 0.0
        bce = F.binary_cross_entropy_with_logits(
            logits, target_f, pos_weight=self.pos_weight, reduction="none")
        probability = torch.sigmoid(logits)
        p_t = probability * target_f + (1.0 - probability) * (1.0 - target_f)
        modulated = (1.0 - p_t).clamp_min(0.0) ** self.gamma * bce
        per_sample = (modulated * valid_f).sum(dims) / valid_per_sample.clamp_min(1.0)
        return per_sample[keep].mean()


if __name__ == "__main__":
    model = SiameseUNet(in_channels=13, base=32)
    print("parameters: %.2f M" % (model.parameter_count() / 1e6))
    x1 = torch.randn(2, 13, 256, 256)
    x2 = torch.randn(2, 13, 256, 256)
    with torch.no_grad():
        out = model(x1, x2)
    print("output shape:", tuple(out.shape))
    n = sum(p.numel() for p in model.parameters())
    print("VRAM estimate for parameters (fp32): %.1f MB" % (n * 4 / 1e6))
