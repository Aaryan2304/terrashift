"""Atomic checkpoint writing and retention for training runs."""

import os
from pathlib import Path

import numpy as np
import torch

from terrashift.training.resume import capture_rng_state


def write_checkpoint_atomic(payload, path):
    """torch.save to a sibling temporary file, then os.replace.

    A crash mid-write leaves the previous checkpoint intact rather than a truncated
    file that fails to load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path

def save_epoch_checkpoints(run_dir, payload, val_ap, best_ap, best_epoch=None,
                           best_metrics=None):
    """Write ``last.pt`` every epoch; write ``best.pt`` only on improvement.

    The selection metric is validation average precision, higher is better. Returns
    ``(best_ap, best_epoch, best_metrics, improved)`` so the caller carries the best
    state forward and a resumed run can restore it.

    No per-epoch checkpoint files are written: only ``last.pt`` and ``best.pt``.
    """
    run_dir = Path(run_dir)
    improved = bool(np.isfinite(val_ap) and val_ap > best_ap)
    if improved:
        best_ap = float(val_ap)
        best_epoch = payload.get("epoch")
        best_metrics = payload.get("val_metrics")
    shared = {"best_ap": best_ap, "best_epoch": best_epoch,
              "best_metrics": best_metrics}
    last_payload = dict(payload)
    last_payload.update(shared)
    # RNG state lives in last.pt only. It contains numpy arrays and tuples, which
    # torch.load(weights_only=True) rejects; best.pt is what inference loads and
    # must stay weights_only-compatible.
    last_payload["rng_state"] = capture_rng_state()
    write_checkpoint_atomic(last_payload, run_dir / "last.pt")
    if improved:
        best_payload = dict(payload)
        best_payload.update(shared)
        write_checkpoint_atomic(best_payload, run_dir / "best.pt")
    return best_ap, best_epoch, best_metrics, improved
