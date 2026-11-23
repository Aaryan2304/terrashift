"""Epoch-boundary resume: RNG capture, log reconciliation, compatibility checks.

Resume is deliberately coarse. It restarts at the first epoch that has no completed
checkpoint, and it refuses to run at all if the configuration differs from the one
that produced the checkpoint. There is no mid-epoch recovery, no LR scheduler state
(there is no scheduler) and no GradScaler state (bf16 needs none).
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

__all__ = ["capture_rng_state", "restore_rng_state", "check_compatibility",
           "truncate_train_log", "resume_from_checkpoint", "repair_best_checkpoint",
           "RESUME_KEYS"]

# Everything that must match for a resume to be honest. Changing any of these makes
# the resumed run a different experiment, not a continuation of the old one.
RESUME_KEYS = (
    "patch_index", "patch_size", "patch_stride", "normalisation_sha256",
    "initialization_checkpoint", "initialization_mode", "base_filters",
    "batch_size", "lr", "pos_weight", "bands", "loss", "precision_dtype",
    "augmentation",
)


def capture_rng_state():
    """All RNG state a resumed run needs to reproduce the stream.

    The DataLoader uses no dedicated generator: it shuffles through the global torch
    CPU RNG, which the ``torch_cpu`` entry covers. ``dataloader_generator`` is
    recorded as None so the absence is explicit rather than assumed.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "dataloader_generator": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        raise ValueError("checkpoint has no RNG state to restore")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint holds CUDA RNG state but CUDA is not available")
        torch.cuda.set_rng_state_all(state["cuda"])


def _flatten(config):
    """Compare on value, with the precision block reduced to its dtype string."""
    flat = dict(config or {})
    precision = flat.get("precision")
    if isinstance(precision, dict):
        flat["precision_dtype"] = precision.get("autocast_dtype")
    return flat


def check_compatibility(saved, current):
    """Raise if the run that produced ``saved`` is not the run being resumed."""
    saved_flat = _flatten(saved)
    current_flat = _flatten(current)
    mismatches = []
    for key in RESUME_KEYS:
        if key not in saved_flat:
            continue
        if saved_flat[key] != current_flat.get(key):
            mismatches.append("%s: checkpoint %r, current %r"
                              % (key, saved_flat[key], current_flat.get(key)))
    if mismatches:
        raise SystemExit(
            "refusing to resume: the run configuration changed since this checkpoint\n  "
            + "\n  ".join(mismatches))
    return True


def truncate_train_log(path, last_committed_epoch):
    """Drop log rows beyond the last committed checkpoint epoch.

    A crash can leave a row for an epoch whose checkpoint was never written. Keeping
    it would report an epoch that does not exist; deleting only those rows keeps one
    record per completed epoch.
    """
    path = Path(path)
    if not path.is_file():
        return {"kept": 0, "dropped": 0}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    kept = [row for row in rows if int(row["epoch"]) <= int(last_committed_epoch)]
    dropped = len(rows) - len(kept)
    if dropped:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        os.replace(temporary, path)
    return {"kept": len(kept), "dropped": dropped}


def _atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def repair_best_checkpoint(run_dir, last_state):
    """Reconstruct ``best.pt`` when a crash landed between the two writes.

    ``save_epoch_checkpoints`` writes ``last.pt`` then ``best.pt``. If the process dies
    between them after an improving epoch, ``last.pt`` records the new best epoch while
    ``best.pt`` still holds the previous one, and because no later epoch improves on
    the recorded best, ``best.pt`` would stay stale for the rest of the run.

    Recovery is narrow on purpose: it acts only when the last completed epoch *is* the
    recorded best epoch and ``best.pt`` does not already carry that epoch. RNG state is
    stripped, so ``best.pt`` stays loadable with ``weights_only=True``.
    """
    run_dir = Path(run_dir)
    best_epoch = last_state.get("best_epoch")
    last_epoch = last_state.get("epoch")
    if best_epoch is None or last_epoch is None or int(best_epoch) != int(last_epoch):
        return {"repaired": False, "reason": "last completed epoch is not the best epoch"}

    best_path = run_dir / "best.pt"
    if best_path.is_file():
        try:
            # best.pt never holds RNG state, so it must stay weights_only-loadable.
            current = torch.load(best_path, map_location="cpu", weights_only=True)
        except Exception:                       # noqa: BLE001 - corrupt file is the case
            current = None
        if current is not None and int(current.get("epoch", -1)) == int(best_epoch):
            return {"repaired": False, "reason": "best.pt already holds the best epoch"}

    payload = dict(last_state)
    payload.pop("rng_state", None)
    _atomic_save(payload, best_path)
    return {"repaired": True, "best_epoch": int(best_epoch),
            "reason": "best.pt restored from last.pt after an interrupted save"}


def resume_from_checkpoint(run_dir, device="cpu"):
    """Load ``last.pt`` and return everything the training loop needs to continue."""
    run_dir = Path(run_dir)
    last_path = run_dir / "last.pt"
    if not last_path.is_file():
        raise SystemExit("refusing to resume: %s does not exist" % last_path)
    # weights_only=False: the payload carries RNG state and plain metadata, and the
    # file is a local artefact of this project's own training loop.
    state = torch.load(last_path, map_location=device, weights_only=False)
    for key in ("epoch", "model_state_dict", "optimizer_state_dict", "run_config"):
        if key not in state:
            raise SystemExit("refusing to resume: %s has no %s" % (last_path, key))
    best_metrics_path = run_dir / "best_metrics.json"
    best_metrics = state.get("best_metrics")
    if best_metrics is None and best_metrics_path.is_file():
        best_metrics = json.loads(best_metrics_path.read_text())
    # A crash between the last.pt and best.pt writes leaves a stale best.pt whose
    # epoch disagrees with the best epoch last.pt recorded. Repair it here, before the
    # run continues, so a later non-improving epoch cannot leave it stale forever.
    repair = repair_best_checkpoint(run_dir, state)
    return {
        "checkpoint": state,
        "best_repaired": repair,
        "start_epoch": int(state["epoch"]) + 1,
        "last_epoch": int(state["epoch"]),
        "best_ap": float(state.get("best_ap", -1.0)),
        "best_epoch": state.get("best_epoch"),
        "best_metrics": best_metrics,
        "rng_state": state.get("rng_state"),
    }
