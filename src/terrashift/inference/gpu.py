"""GPU pipeline helpers: device-memory sampling and evaluation entry construction.

Extracted from ``scripts/evaluation/gpu_inference_evaluation.py``. Importing this module does not
initialise CUDA, so it works on a CPU-only machine.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import torch


class GpuSampler(threading.Thread):
    """Sample device-wide VRAM use; the only way to see ORT allocations."""

    def __init__(self, interval=0.02):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0.0
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                self.peak = max(self.peak, float(out.stdout.strip().splitlines()[0]))
            except (OSError, subprocess.SubprocessError, IndexError, ValueError):
                return
            self.stop_event.wait(self.interval)

def build_entries(cfg):
    stub = argparse.Namespace(
        pairs=["A", "B"],
        manifest=cfg["data"]["manifest"],
        registration=cfg["data"]["registration_summary"],
        labels_dir=cfg["data"]["manual_label_dir"],
        partition_dir=cfg["data"]["local_partition_dir"])
    _, bands, checkpoint = quantize.load_model(str(CHECKPOINT), "cpu")
    normalisation = quantize.require_statistics(str(NORMALISATION), bands, ("A", "B"))
    entries = quantize.build_validation_entries(cfg, stub, checkpoint, normalisation)
    return entries, bands, checkpoint, normalisation
