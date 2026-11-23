"""FP32 ONNX export helpers."""

import torch

PATCH_SIZE = 256  # the patch the exported graph is built and validated at


def export_dummy(bands, size=PATCH_SIZE):
    """The export dummy at the deployed patch size, not the old 128 square."""
    return (torch.zeros(1, len(bands), size, size),
            torch.zeros(1, len(bands), size, size))
