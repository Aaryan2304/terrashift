"""Fine-tune on human-reviewed local patches after labels and index exist."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.augment import PairedRandomGeometry
from data.local_patches import BANDS, LocalManualPatchDataset
from utils.unique_metrics import evaluate_unique, weighted_training_statistics
from models.siamese_unet import DiceBCELoss, SiameseUNet


def run_training(cfg, args, output_dir, epochs, batch_size, base, lr,
                 workers, threshold, seeds, device):
    from train import resolve_pos_weight, run_epoch, seed_everything

    data_cfg = cfg["data"]
    index_path = Path(data_cfg["local_patch_index"])
    index_meta_path = index_path.with_suffix(".json")
    if not index_meta_path.is_file():
        raise FileNotFoundError("local patch index metadata is required: %s" % index_meta_path)
    index_meta = json.loads(index_meta_path.read_text(encoding="utf-8"))
    test_meta_keys = ("test_index_included", "test_label_pixels_read")
    if any(key not in index_meta or index_meta[key] is not False for key in test_meta_keys):
        raise ValueError("training requires an index built without reading test labels")
    required_meta = {
        "uses_human_labels_only": True,
        "uses_osm_labels": False,
        "test_labels_used_for_training": False,
        "buffer_or_cross_split_windows_included": False,
    }
    if any(index_meta.get(key) is not expected for key, expected in required_meta.items()):
        raise ValueError("training requires a human-labeled, split-safe local index")
    patch_size = int(index_meta["patch_size"])
    stride = int(index_meta["stride"])
    if patch_size <= 0 or patch_size % 8:
        raise ValueError("local patch size must be positive and divisible by 8")
    if stride <= 0 or stride > patch_size:
        raise ValueError("local patch stride must be positive and no larger than patch size")
    manifest_path = Path(data_cfg["manifest"])
    registration_path = Path(data_cfg["registration_summary"])
    bands = tuple(data_cfg.get("bands", BANDS))
    if args.init_checkpoint is not None and not Path(args.init_checkpoint).is_file():
        raise FileNotFoundError("initial checkpoint not found: %s" % args.init_checkpoint)
    initial_checkpoint = None if args.init_checkpoint is None else Path(args.init_checkpoint)
    train_raw = LocalManualPatchDataset(
        index_path, "train", manifest_path, registration_path, bands=bands)
    statistics = weighted_training_statistics(train_raw)
    mean, std = statistics["mean"], statistics["std"]
    train_set = LocalManualPatchDataset(
        index_path, "train", manifest_path, registration_path,
        mean=mean, std=std,
        transform=(PairedRandomGeometry() if cfg.get("augmentation", {}).get("enabled", False)
                   else None), bands=bands)
    val_set = LocalManualPatchDataset(
        index_path, "validation", manifest_path, registration_path,
        mean=mean, std=std, bands=bands)
    reviewed_train = statistics["positive"] + statistics["negative"]
    if reviewed_train <= 0:
        raise ValueError("training annotations contain no reviewed pixels")
    fraction = statistics["positive"] / reviewed_train
    pos_weight = resolve_pos_weight(args.pos_weight, fraction)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=(device.type == "cuda"), drop_last=False)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=(device.type == "cuda"), drop_last=False)

    print("dataset: local human-reviewed patches; device:", device)
    print("train patches:", len(train_set), "validation patches:", len(val_set))
    print("official OSCD test labels are not used")
    print("train-only labeled-pixel fraction %.6f -> pos_weight %s" % (
        fraction, "none" if pos_weight is None else "%.4f" % pos_weight))

    summaries = []
    for seed in seeds:
        seed_everything(seed)
        run_dir = Path(output_dir) / ("seed_%d" % seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        model = SiameseUNet(in_channels=len(bands), base=base).to(device)
        if initial_checkpoint is not None:
            checkpoint = torch.load(initial_checkpoint, map_location=device, weights_only=True)
            if "model_state_dict" not in checkpoint:
                raise ValueError("initial checkpoint has no model_state_dict")
            if tuple(checkpoint.get("bands", ())) != bands:
                raise ValueError("initial checkpoint bands do not match the local band order")
            if int(checkpoint.get("base", -1)) != base:
                raise ValueError("initial checkpoint base width does not match the local model")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
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
            "dataset": "local_manual",
            "patch_index": str(index_path.resolve()),
            "manual_label_dir": str(Path(data_cfg["manual_label_dir"]).resolve()),
            "manifest": str(manifest_path.resolve()),
            "registration_summary": str(registration_path.resolve()),
            "initialization_checkpoint": (None if initial_checkpoint is None
                                          else str(initial_checkpoint.resolve())),
            "initialization_mode": "checkpoint" if initial_checkpoint else "random_initialization",
            "train_patch_count": len(train_set),
            "validation_patch_count": len(val_set),
            "train_unique_labelled_pixels": statistics["unique_pixels"],
            "train_patch_weighted_pixels": statistics["patch_pixels"],
            "train_class_counts_unique": {
                "positive": statistics["positive"],
                "negative": statistics["negative"]},
            "validation_metrics_basis": "unique_geographic_pixels",
            "validation_aggregation_rule": "mean of overlapping patch probabilities",
            "patch_size": patch_size,
            "patch_stride": stride,
            "min_labeled_fraction": index_meta["min_labeled_fraction"],
            "overlapping_patches_repeat_pixels_in_training": True,
            "test_dataset_opened": False,
            "test_index_included": bool(index_meta.get("test_index_included", False)),
            "bands": list(bands), "mean_train_only": mean.tolist(),
            "std_train_only": std.tolist(),
            "normalization": "per-band z-score from train labeled pixels; no inter-date correction",
            "train_change_fraction": fraction,
            "pos_weight": pos_weight, "seed": seed, "epochs": epochs,
            "batch_size": batch_size, "base_filters": base, "lr": lr,
            "workers": workers, "threshold": threshold,
            "selection_metric": "average_precision",
            "augmentation": cfg.get("augmentation", {"enabled": False}),
            "device": str(device),
            "cli_overrides": {key: (str(value) if isinstance(value, Path) else value)
                              for key, value in vars(args).items()
                              if value is not None},
            "ignored_label_value": 255,
            "official_test_labels_read": False,
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
        log_path = run_dir / "train_log.csv"
        checkpoint_path = run_dir / "best.pt"
        fields = ["epoch", "train_loss", "train_ap", "train_auc", "train_f1",
                  "val_loss", "val_ap", "val_auc", "val_f1", "val_iou"]
        best_ap = -1.0
        best = {key: float("nan") for key in ("ap", "auc", "f1", "iou")}
        with log_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                train_metrics = run_epoch(
                    model, train_loader, criterion, device, optimizer=optimizer,
                    scaler=scaler, amp=cfg["training"]["amp"],
                    max_batches=args.max_train_batches, collect=True,
                    threshold=threshold)
                with torch.no_grad():
                    val_metrics = evaluate_unique(
                        model, val_loader, device, threshold=threshold,
                        criterion=criterion, max_batches=args.max_val_batches)
                writer.writerow({
                    "epoch": epoch, "train_loss": train_metrics["loss"],
                    "train_ap": train_metrics["ap"], "train_auc": train_metrics["auc"],
                    "train_f1": train_metrics["f1"], "val_loss": val_metrics["loss"],
                    "val_ap": val_metrics["ap"], "val_auc": val_metrics["auc"],
                    "val_f1": val_metrics["f1"], "val_iou": val_metrics["iou"],
                })
                fh.flush()
                print("seed %d epoch %d/%d train AP %.4f | val AP %.4f F1 %.4f IoU %.4f" % (
                    seed, epoch, epochs, train_metrics["ap"], val_metrics["ap"],
                    val_metrics["f1"], val_metrics["iou"]))
                if np.isfinite(val_metrics["ap"]) and val_metrics["ap"] > best_ap:
                    best_ap = val_metrics["ap"]
                    best = {key: val_metrics[key] for key in best}
                    torch.save({
                        "epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "bands": list(bands), "mean": mean.tolist(),
                        "std": std.tolist(), "base": base, "seed": seed,
                        "pos_weight": pos_weight, "val_metrics": val_metrics,
                        "run_config": resolved,
                    }, checkpoint_path)
        summaries.append({"seed": seed, "best_ap": best["ap"],
                          "best_auc": best["auc"], "best_f1": best["f1"],
                          "best_iou": best["iou"], "pos_weight": pos_weight})

    summary_path = Path(output_dir) / "multi_seed_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
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
    return summaries
