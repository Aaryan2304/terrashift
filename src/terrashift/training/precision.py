"""Numerical-precision and count helpers for training."""

import numpy as np
import torch


def resolve_amp_dtype(device, requested):
    """Return the autocast dtype, or None when AMP is off or unsupported.

    bfloat16 only. fp16 autocast produced non-finite gradients for this model at
    every loss scale the GradScaler tried (measured: scales 65536 down to 4096, all
    steps overflowed; fp32 and bf16 are both finite). bf16 shares fp32's exponent
    range, so no gradient scaling is needed or wanted.
    """
    import torch

    if not requested or device.type != "cuda":
        return None
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "AMP requires bfloat16, which %s does not support; run with amp=false "
            "or use a device with bf16 support" % torch.cuda.get_device_name(0))
    return torch.bfloat16

def counts_from_probs(probs, target, threshold=0.5):
    """Confusion counts from probabilities and a 0/1 target."""
    pred = probs >= threshold
    truth = target >= 0.5
    return (int((pred & truth).sum()), int((pred & ~truth).sum()),
            int((~pred & truth).sum()), int((~pred & ~truth).sum()))
