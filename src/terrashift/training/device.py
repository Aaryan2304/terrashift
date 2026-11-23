"""Device sampling and memory instrumentation for training runs."""

import subprocess
import threading

import numpy as np
import torch


class GpuSampler:
    """Poll nvidia-smi in a thread and report mean and peak utilisation.

    ``torch.cuda.utilization`` needs the ``nvidia-ml-py`` package, which is not
    installed. ``nvidia-smi`` is already present on this machine, so the same
    numbers are read from it instead of adding a dependency. If nvidia-smi is
    missing or the query fails, the sampler stays silent and reports None; a
    monitoring gap must not abort a training run.
    """

    def __init__(self, interval_s=0.5):
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread = None
        self.samples = []

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5, stderr=subprocess.DEVNULL)
                first = out.strip().splitlines()[0]
                util, used = (piece.strip() for piece in first.split(","))
                self.samples.append((float(util), float(used)))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self):
        if not self.samples:
            return {"gpu_util_mean_pct": None, "gpu_util_max_pct": None,
                    "gpu_memory_used_max_mib": None, "gpu_samples": 0}
        utils = [s[0] for s in self.samples]
        memory = [s[1] for s in self.samples]
        return {"gpu_util_mean_pct": round(float(np.mean(utils)), 1),
                "gpu_util_max_pct": round(float(np.max(utils)), 1),
                "gpu_memory_used_max_mib": round(float(np.max(memory)), 0),
                "gpu_samples": len(self.samples)}

def epoch_memory():
    """Peak allocated and reserved GiB since the last reset."""
    if not torch.cuda.is_available():
        return {"peak_allocated_gib": None, "peak_reserved_gib": None}
    return {"peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
