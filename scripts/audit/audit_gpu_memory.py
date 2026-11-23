"""Device-wide VRAM check around a short 256 px training step.

PyTorch's counters describe only this process's allocator. On a laptop GPU the
desktop compositor, other browsers and Windows itself hold VRAM that PyTorch will
never see, so "4.00 GiB total minus 0.66 GiB reserved" is not the headroom the
training run actually has. This script reports both views, and the driver's own
per-process accounting.

    python scripts/audit/audit_gpu_memory.py --output outputs/audit/gpu/device_memory.json
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[2]

from terrashift.models.siamese_unet import DiceBCELoss, SiameseUNet

GIB = 1024 ** 3


def nvidia_smi(query):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.SubprocessError) as error:
        return "unavailable (%s)" % type(error).__name__


def compute_processes():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        return result.stdout.strip() or "none"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def run_step(patch, batch, steps=5):
    device = torch.device("cuda")
    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = SiameseUNet(in_channels=4, base=16).to(device)
    criterion = DiceBCELoss(pos_weight=torch.tensor([40.8])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    x1 = torch.rand(batch, 4, patch, patch, device=device)
    x2 = torch.rand(batch, 4, patch, patch, device=device)
    target = (torch.rand(batch, 1, patch, patch, device=device) > 0.9).float()
    valid = torch.ones_like(target, dtype=torch.bool)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            loss = criterion(model(x1, x2), target, valid)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / GIB
    peak_reserved = torch.cuda.max_memory_reserved() / GIB
    free_after, _ = torch.cuda.mem_get_info()
    del model, criterion, optimizer, scaler, x1, x2, target, valid
    torch.cuda.empty_cache()
    return {
        "patch": patch, "batch": batch, "steps": steps,
        "free_before_gib": free_before / GIB, "total_gib": total / GIB,
        "free_after_gib": free_after / GIB,
        "peak_allocated_gib": peak_alloc, "peak_reserved_gib": peak_reserved,
        "unavailable_to_pytorch_gib": (total - free_before) / GIB,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    free, total = torch.cuda.mem_get_info()
    report = {
        "device": torch.cuda.get_device_properties(0).name,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "mem_get_info_before": {"free_gib": free / GIB, "total_gib": total / GIB},
        "nvidia_smi_memory": nvidia_smi("memory.total,memory.used,memory.free"),
        "nvidia_smi_utilisation": nvidia_smi("utilization.gpu"),
        "compute_processes": compute_processes(),
        "note": ("mem_get_info reports what the driver can hand to this process now, "
                 "which already excludes VRAM held by the Windows desktop and other "
                 "applications. It is the only figure that bounds a training run."),
        "runs": [],
    }
    print("device %s | mem_get_info free %.2f of %.2f GiB"
          % (report["device"], free / GIB, total / GIB))
    print("nvidia-smi memory (total, used, free):", report["nvidia_smi_memory"])
    print("compute processes:", report["compute_processes"])
    for batch in (4, 8):
        run = run_step(256, batch)
        # Re-read the driver view while the run's allocations are still cached.
        run["nvidia_smi_during"] = nvidia_smi("memory.used,memory.free")
        report["runs"].append(run)
        print("256 px b%d: peak allocated %.3f GiB, reserved %.3f GiB | "
              "driver free before %.2f, after %.2f GiB | not available to torch %.2f GiB"
              % (batch, run["peak_allocated_gib"], run["peak_reserved_gib"],
                 run["free_before_gib"], run["free_after_gib"],
                 run["unavailable_to_pytorch_gib"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
