"""Measure peak VRAM for one training step at several batch sizes.

This runs the real training path: the same dataset, model, loss, AMP context
and optimiser step that ``train.py`` uses. It then reports peak allocated and
reserved memory for each batch size, so the batch size for a full run is chosen
from a measurement instead of a guess.

The measurement is a few steps on one cached batch, so it does not include the
data loader's host memory or the cost of re-reading patches. That is
intentional: the question is whether the model step fits, not how fast the disk
is.

Usage:
    python scripts/measure_training_memory.py --sizes 2,4,8,16 \
        --out outputs/audit/gpu/memory_test_oscd.json

Nothing is overwritten. Results are printed and, with ``--out``, written to a
new JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.oscd_original import DEFAULT_BANDS, OSCDOriginalPatchDataset
from models.siamese_unet import DiceBCELoss, SiameseUNet
from train import run_epoch, seed_everything


def build_dataset(cfg, patch_size):
    data_cfg = cfg["data"]
    dataset = OSCDOriginalPatchDataset(
        Path(data_cfg["oscd_patch_index"]), "train", bands=DEFAULT_BANDS,
        mean=[0.1] * len(DEFAULT_BANDS), std=[0.1] * len(DEFAULT_BANDS))
    first = dataset[0]
    height, width = first["image1"].shape[-2:]
    if (height, width) != (patch_size, patch_size):
        raise SystemExit("index holds %dx%d patches, expected %dx%d"
                         % (height, width, patch_size, patch_size))
    return dataset


def measure(dataset, batch_size, device, base, steps, cfg):
    """Run ``steps`` real training steps and return the memory statistics."""
    seed_everything(42)
    model = SiameseUNet(in_channels=len(DEFAULT_BANDS), base=base).to(device)
    criterion = DiceBCELoss(
        bce_weight=cfg["training"]["bce_weight"],
        dice_weight=cfg["training"]["dice_weight"],
        pos_weight=4.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    batch = [dataset[i] for i in range(batch_size)]
    collated = torch.utils.data.default_collate(batch)
    del batch

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    timings = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        run_epoch(model, [collated], criterion, device, optimizer=optimizer,
                  scaler=scaler, amp=cfg["training"]["amp"], collect=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.time() - start)

    result = {
        "batch_size": batch_size,
        "seconds_per_step": float(np.median(timings)),
        "steps": steps,
    }
    if device.type == "cuda":
        result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        result["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    del model, criterion, optimizer, scaler, collated
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--sizes", default="2,4,8,16",
                        help="comma-separated batch sizes, ascending")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default=None,
                        help="cuda or cpu; default is cuda when available")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--headroom-gib", type=float, default=0.75,
                        help="reserved memory that must remain free")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    if not sizes or any(s < 1 for s in sizes):
        raise SystemExit("sizes must be positive integers")
    if sizes != sorted(sizes):
        raise SystemExit("sizes must be ascending so the run can stop at the first failure")
    if args.steps < 1:
        raise SystemExit("steps must be positive")
    if args.out is not None and args.out.exists():
        raise SystemExit("refusing to overwrite existing measurement: %s" % args.out)

    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(cfg, args.patch_size)

    total_gib = None
    if device.type == "cuda":
        total_gib = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        print("device %s, total VRAM %.2f GiB, patch %d, base %d, amp %s"
              % (torch.cuda.get_device_name(0), total_gib, args.patch_size,
                 args.base, cfg["training"]["amp"]))
    else:
        print("device cpu, patch %d, base %d" % (args.patch_size, args.base))

    results = []
    selected = None
    for size in sizes:
        try:
            row = measure(dataset, size, device, args.base, args.steps, cfg)
        except RuntimeError as exc:
            message = str(exc).splitlines()[0]
            print("batch %2d: FAILED (%s)" % (size, message[:110]))
            results.append({"batch_size": size, "failed": True, "error": message[:300]})
            break
        results.append(row)
        if device.type == "cuda":
            free_gib = total_gib - row["peak_reserved_gib"]
            print("batch %2d: allocated %.3f GiB, reserved %.3f GiB, free %.3f GiB, "
                  "%.2f s/step" % (size, row["peak_allocated_gib"],
                                   row["peak_reserved_gib"], free_gib,
                                   row["seconds_per_step"]))
            if free_gib >= args.headroom_gib:
                selected = size
        else:
            print("batch %2d: %.2f s/step" % (size, row["seconds_per_step"]))

    payload = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "total_vram_gib": total_gib,
        "patch_size": args.patch_size,
        "base_filters": args.base,
        "amp": bool(cfg["training"]["amp"]),
        "steps_per_size": args.steps,
        "headroom_required_gib": args.headroom_gib,
        "results": results,
        "largest_batch_with_headroom": selected,
    }
    print("\nlargest batch size with at least %.2f GiB free: %s"
          % (args.headroom_gib, selected))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
