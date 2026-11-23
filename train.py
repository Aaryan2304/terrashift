"""Train the Siamese U-Net on an OSCD scene split.

This script uses the official train and validation groups only. It does not
open the test labels. The model uses the four native 10 m Sentinel-2 bands
shared with the local products: B02, B03, B04 and B08.

Three methodological choices are deliberate:

- The best epoch is selected on **average precision**, not on F1 at probability
  0.5. A large ``pos_weight`` inflates the probabilities, so 0.5 stops being a
  meaningful threshold. Average precision is threshold-free.
- **AUC and AP** are reported alongside precision, recall, F1 and IoU. AUC and
  AP judge the ranking; F1 and IoU judge the ranking *and* the threshold.
- Runs are **repeated over several seeds** and the mean and spread are
  reported. A single run on 60 scenes is not a result.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from terrashift.data.dataset import OSCDDataset
from terrashift.data.augment import PairedRandomGeometry
from terrashift.models.siamese_unet import DiceBCELoss, SiameseUNet, FocalLoss
from terrashift.metrics import (average_precision, metrics_from_counts, roc_auc)
from terrashift.training.seeding import seed_everything
from terrashift.training.config import load_config, parse_seeds, resolve_pos_weight
from terrashift.training.precision import counts_from_probs, resolve_amp_dtype


BANDS = ["B02", "B03", "B04", "B08"]

CSV_FIELDS = ["epoch", "train_loss", "train_precision", "train_recall",
              "train_f1", "train_iou", "train_ap", "train_auc",
              "val_loss", "val_precision", "val_recall", "val_f1", "val_iou",
              "val_ap", "val_auc", "train_seconds", "val_seconds"]








def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              amp=False, max_batches=0, collect=False, threshold=0.5):
    """Run one epoch. When ``collect`` is set, also compute AUC and AP."""
    training = optimizer is not None
    model.train(training)
    amp_dtype = resolve_amp_dtype(device, amp)
    total_loss = 0.0
    seen = 0
    counts = np.zeros(4, dtype=np.int64)
    scores = []
    labels = []
    start = time.time()

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        x1 = batch["image1"].to(device, non_blocking=True)
        x2 = batch["image2"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True).float().unsqueeze(1)
        valid = batch.get("valid_mask")
        if valid is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            valid = valid.to(device, non_blocking=True).bool()
            if valid.ndim == 3:
                valid = valid.unsqueeze(1)
            if valid.shape != target.shape:
                raise ValueError("batch valid_mask must match the mask shape")

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                logits = model(x1, x2)
                loss = criterion(logits, target, valid)
            if training:
                # bf16 needs no gradient scaling. The scaler argument is kept so
                # existing callers do not break, but it is only honoured for a
                # non-bf16 autocast mode, which resolve_amp_dtype never returns.
                if scaler is not None and scaler.is_enabled() and amp_dtype != torch.bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        # Probabilities in float32, outside the autocast context, so the
        # ranking metrics are not computed on fp16 rounding.
        probs = torch.sigmoid(logits.detach().float())
        valid_probs = probs[valid]
        valid_target = target[valid]
        counts += np.asarray(counts_from_probs(valid_probs, valid_target, threshold),
                             dtype=np.int64)
        if collect and valid_probs.numel() > 0:
            scores.append(valid_probs.cpu().numpy().ravel())
            labels.append(valid_target.detach().cpu().numpy().ravel())

        n = x1.shape[0]
        total_loss += float(loss.detach()) * n
        seen += n

    result = metrics_from_counts(*counts.tolist())
    if collect:
        if scores:
            s = np.concatenate(scores)
            y = np.concatenate(labels)
            result["ap"] = average_precision(y, s)
            result["auc"] = roc_auc(y, s)
        else:
            result["ap"] = float("nan")
            result["auc"] = float("nan")
    result["loss"] = total_loss / max(seen, 1)
    result["samples"] = seen
    result["seconds"] = time.time() - start
    return result






def train_one(seed, cfg, args, root, output_dir, epochs, batch_size, base,
              lr, workers, pos_weight, device, mean, std, threshold):
    """Train one seed. Returns a summary dict."""
    seed_everything(seed)
    run_dir = os.path.join(output_dir, "seed_%d" % seed)
    os.makedirs(run_dir, exist_ok=True)

    train_set = OSCDDataset(root=root, split="train", bands=BANDS,
                            mean=mean, std=std,
                            transform=(PairedRandomGeometry()
                                       if cfg.get("augmentation", {}).get("enabled", False)
                                       else None))
    val_set = OSCDDataset(root=root, split="val", bands=BANDS,
                          mean=mean, std=std)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=workers,
                            pin_memory=(device.type == "cuda"))

    # Architecture selection. The default is the baseline model, unchanged; the
    # ResNet variant exists only for the comparison experiment.
    arch = args.arch or "siamese_unet"
    if arch == "siamese_resnet18":
        from terrashift.models.siamese_resnet18 import SiameseResNet18
        model = SiameseResNet18(in_channels=len(BANDS),
                                pretrained=(args.arch_pretrained == "yes")).to(device)
    else:
        model = SiameseUNet(in_channels=len(BANDS), base=base).to(device)
    # Loss selection. The default reads config.yaml and reproduces the baseline
    # exactly: bce_dice with bce_weight 1.0 and dice_weight 1.0. "--loss bce"
    # drops the Dice term; "--loss focal" swaps in a focal term with the same
    # pos_weight, so the comparison changes one thing at a time.
    loss_name = args.loss or cfg["training"].get("loss", "bce_dice")
    if loss_name == "bce":
        criterion = DiceBCELoss(bce_weight=1.0, dice_weight=0.0,
                                pos_weight=pos_weight)
    elif loss_name == "focal":
        criterion = FocalLoss(gamma=float(cfg["training"].get("focal_gamma", 2.0)),
                              pos_weight=pos_weight)
    else:
        criterion = DiceBCELoss(bce_weight=cfg["training"]["bce_weight"],
                                dice_weight=cfg["training"]["dice_weight"],
                                pos_weight=pos_weight)
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(cfg["training"]["amp"] and device.type == "cuda"))

    resolved = {
        "config_path": os.path.abspath(args.config),
        "dataset": (args.dataset or cfg["data"].get("dataset", "oscd100")),
        "root": os.path.abspath(root),
        "train_split": "train",
        "validation_split": "val",
        "test_dataset_opened": False,
        "output_dir": os.path.abspath(run_dir),
        "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "base_filters": base, "lr": lr, "workers": workers, "loss": loss_name,
        "architecture": arch,
        "optimizer": cfg["training"]["optimizer"],
        "loss": cfg["training"]["loss"],
        "bce_weight": cfg["training"]["bce_weight"],
        "dice_weight": cfg["training"]["dice_weight"],
        "pos_weight": pos_weight,
        "amp": cfg["training"]["amp"],
        "device": str(device), "bands": BANDS,
        "model_type": cfg["model"]["type"],
        "in_channels": len(BANDS),
        "out_channels": cfg["model"]["out_channels"],
        "augmentation": cfg.get("augmentation", {"enabled": False}),
        "threshold": threshold,
        "selection_metric": "average_precision",
        "cli_overrides": {k: v for k, v in vars(args).items() if v is not None},
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as fh:
        json.dump(resolved, fh, indent=2)

    log_path = os.path.join(run_dir, "train_log.csv")
    checkpoint_path = os.path.join(run_dir, "best.pt")
    best_ap = -1.0
    best_epoch = 0
    best_val = {"ap": float("nan"), "f1": float("nan"),
                "iou": float("nan"), "auc": float("nan")}

    print("=" * 78)
    print("seed %d | device %s | bands %s | base %d | batch %d | pos_weight %s"
          % (seed, device, BANDS, base, batch_size,
             "none" if pos_weight is None else "%.4f" % pos_weight))
    print("train %d samples | validation %d samples | parameters %d"
          % (len(train_set), len(val_set), model.parameter_count()))
    print("train change fraction %.5f" % train_set.change_fraction())

    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            tr = run_epoch(model, train_loader, criterion, device,
                           optimizer=optimizer, scaler=scaler,
                           amp=cfg["training"]["amp"],
                           max_batches=args.max_train_batches,
                           collect=True, threshold=threshold)
            with torch.no_grad():
                va = run_epoch(model, val_loader, criterion, device,
                               amp=False, max_batches=args.max_val_batches,
                               collect=True, threshold=threshold)
            row = {
                "epoch": epoch,
                "train_loss": tr["loss"], "train_precision": tr["precision"],
                "train_recall": tr["recall"], "train_f1": tr["f1"],
                "train_iou": tr["iou"], "train_ap": tr["ap"],
                "train_auc": tr["auc"],
                "val_loss": va["loss"], "val_precision": va["precision"],
                "val_recall": va["recall"], "val_f1": va["f1"],
                "val_iou": va["iou"], "val_ap": va["ap"], "val_auc": va["auc"],
                "train_seconds": tr["seconds"], "val_seconds": va["seconds"]}
            writer.writerow(row)
            log_file.flush()
            print("epoch %d/%d train loss %.5f AP %.4f | val loss %.5f "
                  "AP %.4f F1 %.4f IoU %.4f | %.1fs"
                  % (epoch, epochs, tr["loss"], tr["ap"], va["loss"],
                     va["ap"], va["f1"], va["iou"],
                     tr["seconds"] + va["seconds"]))

            if va["ap"] > best_ap:
                best_ap = va["ap"]
                best_epoch = epoch
                best_val = dict(va)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "bands": BANDS,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "base": base,
                    "seed": seed,
                    "pos_weight": pos_weight,
                    "val_metrics": va,
                    "run_config": resolved,
                }, checkpoint_path)

    print("seed %d best: epoch %d, val AP %.4f, val F1 %.4f, val IoU %.4f"
          % (seed, best_epoch, best_val["ap"], best_val["f1"],
             best_val["iou"]))
    return {"seed": seed, "best_epoch": best_epoch,
            "best_ap": best_val["ap"], "best_f1": best_val["f1"],
            "best_iou": best_val["iou"], "best_auc": best_val["auc"],
            "pos_weight": pos_weight}




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml",
                        help="YAML configuration file")
    parser.add_argument("--root", default=None,
                        help="selected dataset root (original OSCD data or OSCD100 extraction)")
    parser.add_argument("--dataset", choices=("oscd_original", "oscd100", "local_manual"), default=None,
                        help="dataset override; defaults to config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--base", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--arch", choices=["siamese_unet", "siamese_resnet18"],
                        default=None, help="override the model; default keeps baseline")
    parser.add_argument("--arch-pretrained", choices=["yes", "no"], default="yes",
                        help="ImageNet weights for siamese_resnet18")
    parser.add_argument("--loss", choices=["bce", "bce_dice", "focal"],
                        default=None,
                        help="override training.loss; default keeps config")
    parser.add_argument("--seed", type=int, default=None,
                        help="single seed, used when --seeds is absent")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="override the positive-class weight")
    parser.add_argument("--threshold", type=float, default=None,
                        help="probability threshold for the confusion counts")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="debug limit; 0 means use the full train split")
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="debug limit; 0 means use the full validation split")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="optional original-OSCD checkpoint for local fine-tuning")
    parser.add_argument("--normalisation", type=Path, default=None,
                        help="Pair-specific normalisation statistics JSON")
    parser.add_argument("--resume", action="store_true",
                        help="continue from last.pt in the run directory; epoch "
                             "boundary only, no mid-epoch recovery")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    exp_cfg = cfg["experiment"]

    dataset = args.dataset if args.dataset is not None else cfg["data"].get("dataset", "oscd100")
    if dataset not in ("oscd_original", "oscd100", "local_manual"):
        parser.error("unsupported dataset in config: %s" % dataset)
    if args.init_checkpoint is not None and dataset != "local_manual":
        parser.error("--init-checkpoint is only valid with --dataset local_manual")
    if args.normalisation is not None and dataset != "local_manual":
        parser.error("--normalisation is only valid with --dataset local_manual")
    if args.resume and dataset != "local_manual":
        parser.error("--resume is only supported for --dataset local_manual")
    default_root = (cfg["data"].get("oscd_original_root", "./data")
                    if dataset == "oscd_original" else cfg["data"]["oscd_root"])
    root = args.root if args.root is not None else default_root
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else train_cfg["batch_size"]
    base = args.base if args.base is not None else cfg["model"]["base_filters"]
    lr = args.lr if args.lr is not None else train_cfg["lr"]
    workers = args.workers if args.workers is not None else exp_cfg["num_workers"]
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        run_name = exp_cfg["name"]
        if dataset == "oscd100":
            run_name = "oscd100_smoke"
        elif dataset == "local_manual":
            run_name = "local_manual_v1"
        output_dir = os.path.join(exp_cfg["output_root"], "checkpoints", run_name)
    output_path = Path(output_dir)
    if output_path.exists() and (
            not output_path.is_dir() or any(output_path.iterdir())):
        if not getattr(args, "resume", False):
            parser.error(
                "output directory is not empty; choose a new --output-dir to "
                "preserve existing runs")
        # Resuming into an existing run is the whole point; require that the
        # checkpoints it needs are actually there. Seeds are parsed further down,
        # so derive them here.
        if args.seeds is not None:
            resume_seeds = [int(value) for value in str(args.seeds).split(",")
                            if value.strip()]
        elif args.seed is not None:
            resume_seeds = [int(args.seed)]
        else:
            configured = exp_cfg.get("seed", 42)
            resume_seeds = ([int(v) for v in configured]
                            if isinstance(configured, (list, tuple))
                            else [int(configured)])
        missing = [seed for seed in resume_seeds
                   if not (output_path / ("seed_%d" % seed) / "last.pt").is_file()]
        if missing:
            parser.error(
                "cannot resume: no last.pt for seed(s) %s in %s"
                % (", ".join(str(seed) for seed in missing), output_path))
    threshold = args.threshold if args.threshold is not None else cfg["evaluation"]["threshold"]

    if epochs < 1 or batch_size < 1 or base < 1:
        parser.error("epochs, batch-size and base must be positive")

    if args.seeds is not None and args.seed is not None:
        parser.error("use --seed or --seeds, not both")
    if args.seeds is not None:
        seeds = parse_seeds(args.seeds, exp_cfg.get("seeds", exp_cfg["seed"]))
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = parse_seeds(None, exp_cfg.get("seeds", exp_cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == "oscd_original":
        from train_oscd_original import run_training as run_original_training
        run_original_training(
            cfg, args, root, output_dir, epochs, batch_size, base, lr,
            workers, threshold, seeds, device)
        return
    if dataset == "local_manual":
        from train_local_manual import run_training as run_local_training
        run_local_training(
            cfg, args, output_dir, epochs, batch_size, base, lr,
            workers, threshold, seeds, device)
        return

    # Dataset statistics are computed once and reused by every seed.
    train_set = OSCDDataset(root=root, split="train", bands=BANDS)
    mean = train_set.mean.flatten().numpy()
    std = train_set.std.flatten().numpy()
    fraction = train_set.change_fraction()
    pos_weight = resolve_pos_weight(args.pos_weight, fraction)

    print("device:", device, "bands:", BANDS, "base:", base)
    print("train samples:", len(train_set), "seeds:", seeds)
    print("training statistics (train split only): mean", mean.tolist(),
          "std", std.tolist())
    print("train change fraction %.5f -> pos_weight %s"
          % (fraction, "none" if pos_weight is None else "%.4f" % pos_weight))
    print("selection metric: average precision")

    summaries = []
    for seed in seeds:
        summaries.append(train_one(seed, cfg, args, root, output_dir, epochs,
                                   batch_size, base, lr, workers, pos_weight,
                                   device, mean, std, threshold))

    # --- multi-seed summary
    # The standard deviation row uses np.nanstd, which is the population
    # standard deviation (ddof=0), not the sample standard deviation (ddof=1).
    # With three seeds the two differ by a factor of 1.22, so the row label
    # says which one it is.
    summary_path = os.path.join(output_dir, "multi_seed_summary.csv")
    fields = ["seed", "best_epoch", "best_ap", "best_f1", "best_iou",
              "best_auc", "pos_weight"]
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
        metrics = {
            "best_ap": np.asarray([s["best_ap"] for s in summaries], dtype=np.float64),
            "best_auc": np.asarray([s["best_auc"] for s in summaries], dtype=np.float64),
            "best_f1": np.asarray([s["best_f1"] for s in summaries], dtype=np.float64),
            "best_iou": np.asarray([s["best_iou"] for s in summaries], dtype=np.float64),
        }
        for stat, fn in (("mean", np.nanmean), ("std_population", np.nanstd)):
            writer.writerow({"seed": stat, **{
                key: float(fn(values)) for key, values in metrics.items()}})

    print()
    print("wrote", summary_path)
    print("seeds: %d" % len(summaries))
    for key, label in (("best_ap", "AP"), ("best_auc", "AUC"),
                       ("best_f1", "F1"), ("best_iou", "IoU")):
        values = np.asarray([s[key] for s in summaries], dtype=np.float64)
        print("best val %s: mean %.4f  std %.4f  per-seed %s"
              % (label, np.nanmean(values), np.nanstd(values),
                 ["%.4f" % value for value in values]))
    if len(summaries) < 3:
        print("NOTE: fewer than three seeds. A single-seed number is not a result.")
    print("The official test split was not opened.")


if __name__ == "__main__":
    main()
