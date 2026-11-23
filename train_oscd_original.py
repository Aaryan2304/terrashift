"""Training entry point for original OSCD region-level patches.

The function is called by ``train.py`` when config.yaml selects
``terrashift.data.dataset: oscd_original``. The official test regions are indexed for
later evaluation, but this training path never constructs a test dataset.
"""

from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from terrashift.data.augment import PairedRandomGeometry
from terrashift.data.oscd_original import (DEFAULT_BANDS, OSCDOriginalPatchDataset,
                                build_patch_index, compute_training_statistics,
                                validate_region_splits, write_patch_index)
from terrashift.models.siamese_unet import DiceBCELoss, FocalLoss, SiameseUNet
from terrashift.training.device import GpuSampler, epoch_memory







def run_training(cfg, args, root, output_dir, epochs, batch_size, base, lr,
                 workers, threshold, seeds, device):
    from train import resolve_pos_weight, run_epoch, seed_everything

    data_cfg = cfg["data"]
    root = Path(root)
    split_config = Path(data_cfg["oscd_region_splits"])
    index_path = Path(data_cfg["oscd_patch_index"])
    patch_size = int(data_cfg["oscd_patch_size"])
    stride = int(data_cfg["oscd_patch_stride"])
    bands = tuple(data_cfg.get("bands", DEFAULT_BANDS))
    if patch_size % 8:
        raise ValueError("OSCD patch size must be a multiple of 8")
    if stride <= 0 or stride > patch_size:
        raise ValueError("OSCD patch stride must be positive and no larger than patch size")

    split_info = validate_region_splits(root, split_config)
    rows = build_patch_index(root, split_config, patch_size, stride)
    index_status = write_patch_index(rows, index_path, patch_size, stride)
    print("patch index:", index_status, index_path)
    mean, std = compute_training_statistics(rows, bands)
    train_set = OSCDOriginalPatchDataset(index_path, "train", bands, mean, std,
                                         transform=(PairedRandomGeometry()
                                                    if cfg.get("augmentation", {}).get("enabled", False)
                                                    else None))
    val_set = OSCDOriginalPatchDataset(index_path, "validation", bands, mean, std)
    train_fraction = train_set.change_fraction()
    pos_weight = resolve_pos_weight(args.pos_weight, train_fraction)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=(device.type == "cuda"), drop_last=False)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=(device.type == "cuda"), drop_last=False)
    val_region_sets = {
        city: OSCDOriginalPatchDataset(index_path, "validation", bands, mean, std,
                                       regions=[city])
        for city in split_info["validation"]}
    val_region_loaders = {
        city: DataLoader(subset, batch_size=batch_size, shuffle=False,
                         num_workers=workers,
                         pin_memory=(device.type == "cuda"), drop_last=False)
        for city, subset in val_region_sets.items()}

    print("dataset: original OSCD; device:", device, "bands:", list(bands))
    print("train patches:", len(train_set), "regions:", len(split_info["train"]))
    print("validation patches:", len(val_set), "regions:", len(split_info["validation"]))
    print("official test regions:", len(split_info["test"]), "(not opened)")
    print("excluded regions:", split_info["excluded"])
    print("train-only mean:", mean.tolist())
    print("train-only std:", std.tolist())
    print("train change fraction %.6f -> pos_weight %s" % (
        train_fraction, "none" if pos_weight is None else "%.4f" % pos_weight))

    summaries = []
    for seed in seeds:
        seed_everything(seed)
        run_dir = Path(output_dir) / ("seed_%d" % seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Experiment switches live in train.py's parser but this module owns the
        # OSCD training loop, so they are honoured here too. The default path is
        # unchanged: no --loss means the config value, no --arch means the
        # baseline SiameseUNet.
        arch = getattr(args, "arch", None) or "siamese_unet"
        if arch == "siamese_resnet18":
            from terrashift.models.siamese_resnet18 import SiameseResNet18
            model = SiameseResNet18(
                in_channels=len(bands),
                pretrained=(getattr(args, "arch_pretrained", "yes") == "yes")).to(device)
        else:
            model = SiameseUNet(in_channels=len(bands), base=base).to(device)

        loss_name = getattr(args, "loss", None) or cfg["training"].get("loss", "bce_dice")
        if loss_name == "bce":
            criterion = DiceBCELoss(bce_weight=1.0, dice_weight=0.0,
                                    pos_weight=pos_weight).to(device)
        elif loss_name == "focal":
            criterion = FocalLoss(
                gamma=float(cfg["training"].get("focal_gamma", 2.0)),
                pos_weight=pos_weight).to(device)
        else:
            criterion = DiceBCELoss(
                bce_weight=cfg["training"]["bce_weight"],
                dice_weight=cfg["training"]["dice_weight"],
                pos_weight=pos_weight).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=cfg["training"]["weight_decay"])
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(cfg["training"]["amp"] and device.type == "cuda"))
        resolved = {
            "dataset": "original_oscd",
            "loss": loss_name,
            "architecture": arch,
            "data_root": str(root.resolve()),
            "split_config": str(split_config.resolve()),
            "patch_index": str(index_path.resolve()),
            "patch_size": patch_size,
            "patch_stride": stride,
            "train_regions": split_info["train"],
            "validation_regions": split_info["validation"],
            "official_test_regions_locked": split_info["test"],
            "excluded_regions": split_info["excluded"],
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "base_filters": base,
            "lr": lr,
            "workers": workers,
            "bands": list(bands),
            "mean_train_only": mean.tolist(),
            "std_train_only": std.tolist(),
            "train_change_fraction": train_fraction,
            "pos_weight": pos_weight,
            "threshold": threshold,
            "selection_metric": "average_precision",
            "augmentation": cfg.get("augmentation", {"enabled": False}),
            "device": str(device),
            "cli_overrides": {key: value for key, value in vars(args).items()
                               if value is not None},
            "official_test_labels_read": False,
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
        log_path = run_dir / "train_log.csv"
        checkpoint_path = run_dir / "best.pt"
        fields = ["epoch", "train_loss", "train_ap", "train_auc", "train_f1",
                  "val_loss", "val_ap", "val_auc", "val_f1", "val_iou",
                  "train_seconds", "val_seconds", "gpu_util_mean_pct",
                  "gpu_util_max_pct", "gpu_memory_used_max_mib", "gpu_samples",
                  "peak_allocated_gib", "peak_reserved_gib"]
        region_fields = ["epoch", "region", "patches", "loss", "ap", "auc",
                         "f1", "iou"]
        best_ap = -1.0
        best_values = {"ap": float("nan"), "auc": float("nan"),
                       "f1": float("nan"), "iou": float("nan")}
        with log_path.open("w", newline="", encoding="utf-8") as fh, \
                (run_dir / "validation_by_region.csv").open(
                    "w", newline="", encoding="utf-8") as region_fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            region_writer = csv.DictWriter(region_fh, fieldnames=region_fields)
            region_writer.writeheader()
            for epoch in range(1, epochs + 1):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                with GpuSampler() as sampler:
                    train_start = time.time()
                    tr = run_epoch(
                        model, train_loader, criterion, device, optimizer=optimizer,
                        scaler=scaler, amp=cfg["training"]["amp"],
                        max_batches=args.max_train_batches, collect=True,
                        threshold=threshold)
                    train_seconds = time.time() - train_start
                    val_start = time.time()
                    with torch.no_grad():
                        va = run_epoch(
                            model, val_loader, criterion, device, amp=False,
                            max_batches=args.max_val_batches, collect=True,
                            threshold=threshold)
                    val_seconds = time.time() - val_start
                memory = epoch_memory()
                gpu = sampler.summary()
                writer.writerow({
                    "epoch": epoch, "train_loss": tr["loss"],
                    "train_ap": tr["ap"], "train_auc": tr["auc"],
                    "train_f1": tr["f1"], "val_loss": va["loss"],
                    "val_ap": va["ap"], "val_auc": va["auc"],
                    "val_f1": va["f1"], "val_iou": va["iou"],
                    "train_seconds": round(train_seconds, 1),
                    "val_seconds": round(val_seconds, 1),
                    **gpu, **memory,
                })
                fh.flush()
                for city, loader in val_region_loaders.items():
                    with torch.no_grad():
                        region_metrics = run_epoch(
                            model, loader, criterion, device, amp=False,
                            collect=True, threshold=threshold)
                    region_writer.writerow({
                        "epoch": epoch, "region": city,
                        "patches": len(val_region_sets[city]),
                        "loss": region_metrics["loss"], "ap": region_metrics["ap"],
                        "auc": region_metrics["auc"], "f1": region_metrics["f1"],
                        "iou": region_metrics["iou"],
                    })
                region_fh.flush()
                print("seed %d epoch %d/%d train loss %.5f AP %.4f | "
                      "val loss %.5f AP %.4f F1 %.4f IoU %.4f | %.1fs train, "
                      "%.1fs val, GPU %.0f%%, peak %.2f GiB" % (
                          seed, epoch, epochs, tr["loss"], tr["ap"],
                          va["loss"], va["ap"], va["f1"], va["iou"],
                          train_seconds, val_seconds,
                          gpu["gpu_util_mean_pct"] or 0.0,
                          memory["peak_reserved_gib"] or 0.0))
                if np.isfinite(va["ap"]) and va["ap"] > best_ap:
                    best_ap = va["ap"]
                    best_values = {key: va[key] for key in ("ap", "auc", "f1", "iou")}
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "bands": list(bands),
                        "mean": mean.tolist(), "std": std.tolist(),
                        "base": base, "seed": seed, "pos_weight": pos_weight,
                        "val_metrics": va, "run_config": resolved,
                    }, checkpoint_path)
        summaries.append({
            "seed": seed, "best_ap": best_values["ap"],
            "best_auc": best_values["auc"], "best_f1": best_values["f1"],
            "best_iou": best_values["iou"], "pos_weight": pos_weight,
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "multi_seed_summary.csv"
    metrics = ("best_ap", "best_auc", "best_f1", "best_iou")
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", *metrics, "pos_weight"])
        writer.writeheader()
        writer.writerows(summaries)
        # Population standard deviation (ddof=0); the row label says so
        # because with three seeds it differs from the sample value by 1.22x.
        for label, reducer in (("mean", np.nanmean),
                               ("std_population", np.nanstd)):
            writer.writerow({"seed": label, **{
                metric: float(reducer([row[metric] for row in summaries]))
                for metric in metrics}})
    print("wrote", summary_path)
    print("official test labels were not read")
    return summaries
