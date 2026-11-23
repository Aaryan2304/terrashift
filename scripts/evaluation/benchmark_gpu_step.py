"""Benchmark the real training step: forward, loss, backward, AdamW, AMP.

Synthetic inputs of the correct shape only. No dataset, no target-domain training,
no checkpoint written.

Reports peak **allocated** and peak **reserved** VRAM separately, because reserved is
what actually caps the card: PyTorch keeps freed blocks in its cache, so a run that
reports 0.66 GiB allocated can still reserve 2.0 GiB. Reporting only the incremental
reserved figure would understate total usage; this reports the absolute peak of both
against the device total.

    python scripts/evaluation/benchmark_gpu_step.py --output outputs/audit/gpu/step_benchmark.json
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

from terrashift.models.siamese_unet import DiceBCELoss, SiameseUNet

CONFIGS = [(size, batch) for size in (128, 256) for batch in (1, 2, 4, 8)]
WARMUP = 3
TIMED = 15


def process_rss_gib():
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        return float("nan")
    return float("nan")


def one_config(size, batch, device, base=16, channels=4, amp=True):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = SiameseUNet(in_channels=channels, base=base).to(device)
    criterion = DiceBCELoss(pos_weight=torch.tensor([40.8])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    x1 = torch.rand(batch, channels, size, size, device=device)
    x2 = torch.rand(batch, channels, size, size, device=device)
    target = (torch.rand(batch, 1, size, size, device=device) > 0.9).float()
    valid = torch.ones_like(target, dtype=torch.bool)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x1, x2)
            loss = criterion(logits, target, valid)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    loss_value = float("nan")
    try:
        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()
        times = []
        for _ in range(TIMED):
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss_value = step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        result = {
            "patch": size, "batch": batch, "status": "completed",
            "loss_last": loss_value,
            "step_median_ms": 1000 * statistics.median(times),
            "step_p95_ms": 1000 * float(np.percentile(times, 95)),
            "step_min_ms": 1000 * min(times),
            "peak_allocated_gib": peak_alloc, "peak_reserved_gib": peak_reserved,
            "device_total_gib": total,
            "headroom_allocated_gib": total - peak_alloc,
            "headroom_reserved_gib": total - peak_reserved,
        }
    except torch.cuda.OutOfMemoryError as error:
        result = {"patch": size, "batch": batch, "status": "OOM",
                  "error": str(error).splitlines()[0][:200]}
    finally:
        del model, criterion, optimizer, scaler, x1, x2, target, valid
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    baseline = process_rss_gib()
    report = {
        "device": properties.name,
        "total_vram_gib": properties.total_memory / 1024 ** 3,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "amp": not args.no_amp,
        "base_filters": args.base,
        "warmup_steps": WARMUP, "timed_steps": TIMED,
        "baseline_process_rss_gib": baseline,
        "note": ("synthetic inputs; peak allocated and peak reserved are absolute "
                 "peaks against total VRAM, not incremental figures"),
        "results": [],
    }
    print("device %s | %.2f GiB | torch %s cuda %s" % (
        report["device"], report["total_vram_gib"], report["torch"], report["cuda"]))
    for size, batch in CONFIGS:
        result = one_config(size, batch, device, base=args.base, amp=not args.no_amp)
        report["results"].append(result)
        if result["status"] == "completed":
            print("  %3d px b%d: %.1f ms median, %.1f ms p95 | alloc %.3f GiB | "
                  "reserved %.3f GiB | headroom %.2f GiB"
                  % (size, batch, result["step_median_ms"], result["step_p95_ms"],
                     result["peak_allocated_gib"], result["peak_reserved_gib"],
                     result["headroom_reserved_gib"]))
        else:
            print("  %3d px b%d: OOM" % (size, batch))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
