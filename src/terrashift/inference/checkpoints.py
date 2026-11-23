"""Checkpoint loading for inference, including the sigmoid wrapper."""
from __future__ import annotations

import numpy as np
import torch

from terrashift.data.validity import DEFAULT_BANDS
from terrashift.models.siamese_unet import DiceBCELoss, SiameseUNet


class Wrapper(torch.nn.Module):
    """Two named inputs, so the ONNX graph matches the training call.

    The sigmoid lives here. That is why the ONNX output is a probability, and why
    nothing downstream may apply a second one.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image1, image2):
        return torch.sigmoid(self.model(image1, image2))

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    bands = tuple(checkpoint.get("bands", DEFAULT_BANDS))
    model = SiameseUNet(in_channels=len(bands), base=int(checkpoint["base"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return Wrapper(model).to(device), bands, checkpoint


def enumerate_support(size):
    """Theoretical input-dependency interval for every output position.

    Same propagation as ``theoretical_receptive_field`` but it keeps every
    position instead of only the centre. Pooling makes the support depend on the
    output coordinate modulo the total stride (2**3 = 8), so the centre pixel is
    not representative of the whole output.
    """
    def conv3(sets):
        n = len(sets)
        out = []
        for i in range(n):
            support = set()
            for j in (i - 1, i, i + 1):
                if 0 <= j < n:
                    support |= sets[j]
            out.append(support)
        return out

    def pool2(sets):
        return [sets[2 * i] | sets[2 * i + 1] for i in range(len(sets) // 2)]

    def conv_transpose2(sets):
        out = []
        for support in sets:
            out.append(support)
            out.append(support)
        return out

    def block(sets):
        return conv3(conv3(sets))

    def skip(a, b):
        return [left | right for left, right in zip(a, b)]

    x = [{i} for i in range(size)]
    e1 = block(x)
    e2 = block(pool2(e1))
    e3 = block(pool2(e2))
    e4 = block(pool2(e3))
    c3 = block(skip(conv_transpose2(e4), e3))
    c2 = block(skip(conv_transpose2(c3), e2))
    c1 = block(skip(conv_transpose2(c2), e1))
    return [(min(s), max(s), len(s)) for s in c1]
