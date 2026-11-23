"""PyTorch execution session used by the inference and evaluation scripts.

Extracted from ``scripts/evaluation/gpu_inference_evaluation.py``. This is the PyTorch counterpart
of :mod:`terrashift.inference.onnx_runtime` and is deliberately kept separate from it:
one wraps a ``torch.nn.Module``, the other an ONNX Runtime session. Importing this
module does not touch CUDA, a checkpoint or a dataset.
"""

from __future__ import annotations

import torch

__all__ = ["TorchSession"]

class TorchSession:
    """Expose the trained torch model through the session.run contract."""

    def __init__(self, wrapper, device, half=False):
        self.wrapper = wrapper
        self.device = device
        self.half = half

    def run(self, names, feed):
        x1 = torch.from_numpy(feed["image1"]).to(self.device, non_blocking=True)
        x2 = torch.from_numpy(feed["image2"]).to(self.device, non_blocking=True)
        if self.half:
            x1 = x1.half()
            x2 = x2.half()
        with torch.no_grad():
            out = self.wrapper(x1, x2)
        return [out.float().cpu().numpy()]
