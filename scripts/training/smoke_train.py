"""Five-step real-data training smoke test. Not a training run.

Uses the real training patches, the real masks, the real per-pair normalisation and
the OSCD-pretrained weights, and executes exactly five optimizer steps. Nothing is
written to the production checkpoint path; the smoke weights, if saved at all, go to
an explicitly named smoke directory.

    python scripts/training/smoke_train.py --output outputs/smoke/v1
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.local_patches import BANDS, LocalManualPatchDataset, pair_image_loader
from terrashift.preprocessing.normalisation import (normalise_reflectance, pair_statistics,
                                require_statistics)
from terrashift.models.siamese_unet import DiceBCELoss, SiameseUNet

STEPS = 5
BATCH = 4
LR = 1e-4


class Args:
    """Minimal CLI stand-in for train_local_manual.run_training."""

    def __init__(self, normalisation, init_checkpoint):
        self.normalisation = normalisation
        self.init_checkpoint = init_checkpoint
        self.max_train_batches = None
        self.max_val_batches = None
        self.pos_weight = None


def startup_without_normalisation(index_path, manifest, registration):
    """Call the trainer's own startup with no --normalisation and catch the guard."""
    from train_local_manual import run_training

    cfg = {"data": {"local_patch_index": str(index_path),
                    "manual_label_dir": "outputs/labels",
                    "local_partition_dir": "outputs/partitions_v2_observation_valid",
                    "manifest": manifest, "registration_summary": registration,
                    "bands": list(BANDS)},
           "training": {"bce_weight": 1.0, "dice_weight": 1.0, "weight_decay": 1e-4,
                        "amp": True},
           "augmentation": {"enabled": False}}
    started = time.perf_counter()
    try:
        run_training(cfg, Args(None, None), str(Path("/tmp/never")), 1, BATCH, 16,
                     LR, 0, 0.5, [42], torch.device("cpu"))
    except SystemExit as error:
        return {"raised": True, "message": str(error),
                "seconds": time.perf_counter() - started}
    except Exception as error:                      # noqa: BLE001 - reported, not hidden
        return {"raised": False, "error": type(error).__name__ + ": " + str(error),
                "seconds": time.perf_counter() - started}
    return {"raised": False, "error": "no exception", "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/dataset/v2")
    parser.add_argument("--checkpoint",
                        default="outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16",
                        help="fp16 with GradScaler is numerically unstable for this "
                             "model: every gradient was non-finite at scales "
                             "65536-16384. bf16 needs no scaler and is stable.")
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    indexPath = Path(args.root) / "train_index.csv"
    manifest = "configs/data_manifest.yaml"
    registration = "outputs/audit/registration_v2_observation_valid/registration.json"
    bands = BANDS

    report = {"steps_requested": STEPS, "batch_size": BATCH, "lr": LR,
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    # ---- 1. startup enforcement ------------------------------------------------
    report["startup_without_normalisation"] = startup_without_normalisation(
        indexPath, manifest, registration)
    print("startup without --normalisation:", report["startup_without_normalisation"])

    normalisation = require_statistics(Path(args.root) / "normalisation.json", bands,
                                       ("A", "B"))
    report["normalisation_file"] = normalisation["_file"]
    report["normalisation_sha256"] = normalisation["_sha256"]
    print("startup with normalisation: ok, sha256", normalisation["_sha256"][:16])

    # ---- dataset ---------------------------------------------------------------
    dataset = LocalManualPatchDataset(indexPath, "train", Path(manifest),
                                      Path(registration), normalisation=normalisation,
                                      bands=bands)
    pairs = [dataset.rows[i]["pair"] for i in range(len(dataset))]
    report["dataset"] = {"patches": len(dataset),
                         "pair_counts": {p: pairs.count(p) for p in set(pairs)}}
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)

    # ---- normalisation correctness, spot-checked on real windows ---------------
    loader_for_pair = pair_image_loader(manifest, registration, bands=bands)
    checks = {}
    for pair in ("A", "B"):
        row = next(r for r in dataset.rows if r["pair"] == pair)
        r0, c0 = int(row["row_off"]), int(row["col_off"])
        raw = loader_for_pair(pair, r0, c0, int(row["height"]))[0]
        mean, std = pair_statistics(normalisation, pair)
        expected = normalise_reflectance(raw, mean, std)
        index = dataset.rows.index(row)
        sample = dataset[index]["image1"].numpy()
        checks[pair] = {"max_abs_difference": float(np.abs(expected - sample).max()),
                        "mean": float(sample.mean()), "std": float(sample.std()),
                        "statistics_mean": [float(v) for v in mean.ravel()],
                        "statistics_std": [float(v) for v in std.ravel()]}
        print("pair %s spot check: max |expected - dataset| = %.3e"
              % (pair, checks[pair]["max_abs_difference"]))
    report["normalisation_spot_check"] = checks

    # ---- model -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(checkpoint["bands"]) != tuple(bands):
        raise SystemExit("checkpoint bands do not match")
    model = SiameseUNet(in_channels=len(bands), base=int(checkpoint["base"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    criterion = DiceBCELoss(bce_weight=1.0, dice_weight=1.0,
                            pos_weight=float(checkpoint["pos_weight"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    use_amp = device.type == "cuda" and amp_dtype is not None
    # GradScaler applies to fp16 only; bf16 shares fp32's exponent range.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    torch.cuda.reset_peak_memory_stats()
    before = {name: tensor.detach().clone()
              for name, tensor in list(model.named_parameters())[:3]}

    steps = []
    iterator = iter(loader)
    for step in range(1, STEPS + 1):
        wait_start = time.perf_counter()
        batch = next(iterator)
        data_time = time.perf_counter() - wait_start
        compute_start = time.perf_counter()

        x1 = batch["image1"].to(device, non_blocking=True)
        x2 = batch["image2"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True).float().unsqueeze(1)
        valid = batch["valid_mask"].to(device, non_blocking=True).bool().unsqueeze(1)
        target = torch.where(valid, target, torch.zeros_like(target))

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x1, x2)
            loss = criterion(logits, target, valid)
        # fp16 needs scaling; bf16 and fp32 do not. Unscale before clipping, or the
        # clipped norm is a multiple of the loss scale and the scaler's overflow
        # detection is defeated - which silently skipped three steps in an earlier
        # version of this harness.
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grad_finite = all(bool(torch.isfinite(g).all()) for g in grads)
        grad_norm = float(grad_norm_value)
        step_skipped = not any(torch.isfinite(g).all() for g in grads)
        label_values = sorted(set(int(v) for v in torch.unique(batch["mask"]).tolist()))
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - compute_start

        changed = {name: float((tensor - before[name]).abs().max())
                   for name, tensor in model.named_parameters() if name in before}
        steps.append({
            "step": step, "loss": float(loss), "loss_finite": bool(torch.isfinite(loss)),
            "input_shapes": [list(x1.shape), list(x2.shape)],
            "input_finite": bool(torch.isfinite(x1).all() and torch.isfinite(x2).all()),
            "logit_finite": bool(torch.isfinite(logits).all()),
            "grad_finite": grad_finite, "grad_norm": grad_norm,
            "scaler_scale": float(scaler.get_scale()) if scaler.is_enabled() else 1.0,
            "grads_present": len(grads), "parameters": len(list(model.parameters())),
            "label_values": label_values,
            "valid_pixels": int(valid.sum()), "ignored_pixels": int((~valid).sum()),
            "data_seconds": data_time, "compute_seconds": elapsed,
            "parameter_changes": changed,
        })
        print("step %d | loss %.5f | grad norm %.4f | data %.3fs | compute %.3fs | "
              "labels %s | changed %s"
              % (step, float(loss), grad_norm, data_time, elapsed, label_values,
                 {k: round(v, 8) for k, v in changed.items()}))

    peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3 if device.type == "cuda" else 0.0
    free, total = torch.cuda.mem_get_info() if device.type == "cuda" else (0, 0)

    report["steps"] = steps
    report["timing"] = {
        "median_compute_s": statistics.median(s["compute_seconds"] for s in steps),
        "p95_compute_s": float(np.percentile([s["compute_seconds"] for s in steps], 95)),
        "median_data_s": statistics.median(s["data_seconds"] for s in steps),
        "p95_data_s": float(np.percentile([s["data_seconds"] for s in steps], 95)),
        "loader_share_of_step": statistics.median(
            s["data_seconds"] / max(s["data_seconds"] + s["compute_seconds"], 1e-9)
            for s in steps),
    }
    report["memory"] = {"peak_allocated_gib": peak_alloc,
                        "peak_reserved_gib": peak_reserved,
                        "free_gib": free / 1024 ** 3, "total_gib": total / 1024 ** 3}
    report["checks"] = {
        "all_losses_finite": all(s["loss_finite"] for s in steps),
        "all_inputs_finite": all(s["input_finite"] for s in steps),
        "all_logits_finite": all(s["logit_finite"] for s in steps),
        "all_grads_finite": all(s["grad_finite"] for s in steps),
        "all_grads_present": all(s["grads_present"] > 0 for s in steps),
        # The last step must have moved parameters; earlier steps may be skipped by
        # the AMP scaler while it backs off from an overflowing loss scale.
        "parameters_changed_on_last_step": any(
            v > 0 for v in steps[-1]["parameter_changes"].values()),
        "steps_with_parameter_change": sum(
            1 for s in steps if any(v > 0 for v in s["parameter_changes"].values())),
        "steps_with_finite_gradients": sum(1 for s in steps if s["grad_finite"]),
        "label_values": sorted({v for s in steps for v in s["label_values"]}),
        "steps_completed": len(steps),
    }
    report["amp"] = args.amp
    report["note"] = ("smoke test only: five optimizer steps, weights not a trained "
                      "model, production checkpoint untouched")

    out = Path(args.output) / "smoke_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)
    print("checks:", json.dumps(report["checks"]))
    print("timing:", json.dumps(report["timing"]))
    print("memory:", json.dumps(report["memory"]))
    if not all([report["checks"]["all_losses_finite"], report["checks"]["all_inputs_finite"],
                report["checks"]["all_logits_finite"], report["checks"]["all_grads_finite"],
                report["checks"]["parameters_changed_on_last_step"]]):
        raise SystemExit("smoke test checks failed")


if __name__ == "__main__":
    main()
