"""Fine-tune on human-reviewed local patches after labels and index exist."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from terrashift.data.augment import PairedRandomGeometry
from terrashift.data.local_patches import BANDS, LocalManualPatchDataset, pair_image_loader
from terrashift.data.validation_windows import (ValidationLatticeDataset,
                                     read_validation_target)
from terrashift.evaluation.lattice import evaluate_lattice_all
from terrashift.training.resume import (capture_rng_state, check_compatibility,
                          restore_rng_state, resume_from_checkpoint,
                          truncate_train_log)
from terrashift.evaluation.unique_regions import weighted_training_statistics
from terrashift.models.siamese_unet import DiceBCELoss, SiameseUNet
from terrashift.training.checkpointing import save_epoch_checkpoints, write_checkpoint_atomic



def per_pair_log_values(val_metrics):
    """Per-pair validation values for the epoch log, or blanks when absent.

    ``evaluate_lattice_all`` returns a ``per_pair`` mapping; the aggregate columns
    alone hide the fact that Pair A and Pair B are separate scenes with very
    different positive rates (0.17 against 0.64 unique).
    """
    per_pair = (val_metrics or {}).get("per_pair") or {}
    values = {}
    for pair in ("A", "B"):
        entry = per_pair.get(pair) or {}
        values["val_ap_%s" % pair] = entry.get("ap")
        values["val_reviewed_%s" % pair] = entry.get("reviewed_pixels")
    return values






def instance_class_counts(dataset, max_patches=None):
    """Positive and negative reviewed-pixel counts over patch instances.

    Used when per-pair normalisation is supplied. The coverage-weighted statistics
    require a gapless patch lattice, and a density-selected index has gaps, so they
    cannot run here; and when the statistics are supplied by file their mean and std
    are not needed anyway. What is needed is the class balance the loss actually
    sees: the criterion averages a per-sample BCE and a per-sample Dice term, so
    every sample contributes equally regardless of how many reviewed pixels it has.
    """
    positive = negative = ignored = 0
    per_patch = []
    total = len(dataset) if max_patches is None else min(len(dataset), max_patches)
    for index in range(total):
        sample = dataset[index]
        valid = sample["valid_mask"].numpy().astype(bool)
        labels = sample["mask"].numpy()[valid]
        p = int((labels == 1).sum())
        n = int((labels == 0).sum())
        positive += p
        negative += n
        ignored += int((~valid).sum())
        per_patch.append((p, n))
    rates = [p / (p + n) for p, n in per_patch if (p + n) > 0]
    return {"positive": positive, "negative": negative, "ignored": ignored,
            "patches": total, "per_patch": per_patch,
            "pooled_rate": positive / max(1, positive + negative),
            "mean_sample_rate": float(sum(rates) / len(rates)) if rates else 0.0,
            "all_negative_patches": sum(1 for p, n in per_patch if p == 0),
            "all_positive_patches": sum(1 for p, n in per_patch if n == 0)}


def train_raw_rows(index_path):
    with Path(index_path).open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == "train"]


def run_training(cfg, args, output_dir, epochs, batch_size, base, lr,
                 workers, threshold, seeds, device):
    from train import resolve_amp_dtype, resolve_pos_weight, run_epoch, seed_everything

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
    # Fail fast rather than silently reverting to reviewed-pixel statistics. An
    # index that declares per-pair normalisation is unusable without its file.
    normalisation = None
    normalisation_path = getattr(args, "normalisation", None)
    declares_per_pair = str(index_meta.get("normalisation_scheme", "")).find("per-pair") >= 0
    if declares_per_pair and normalisation_path is None:
        raise SystemExit(
            "this index declares per-pair normalisation but no --normalisation file "
            "was given; refusing to fall back to single-set statistics (%s)"
            % index_meta.get("normalisation", "no path recorded"))
    if normalisation_path is not None:
        from terrashift.preprocessing.normalisation import NormalisationError, require_statistics
        pairs = sorted({row["pair"] for row in train_raw_rows(index_path)})
        try:
            normalisation = require_statistics(normalisation_path, bands, pairs)
        except NormalisationError as error:
            raise SystemExit("normalisation: %s" % error) from error
    # Statistics and class counts come from the raw reflectance values, so this
    # dataset must not normalise.
    train_raw = LocalManualPatchDataset(
        index_path, "train", manifest_path, registration_path, bands=bands)
    if normalisation is not None:
        counts = instance_class_counts(train_raw)
        statistics = {
            "positive": float(counts["positive"]),
            "negative": float(counts["negative"]),
            # Counts below are patch-INSTANCE occurrences: a geographic pixel
            # covered by four overlapping windows is counted four times.
            "reviewed_pixel_occurrences": counts["positive"] + counts["negative"],
            "patch_pixel_occurrences": (counts["positive"] + counts["negative"]
                                        + counts["ignored"]),
            "patch_instance_positive": counts["positive"],
            "patch_instance_negative": counts["negative"],
            "mean": None, "std": None, "instance_counts": counts,
            "weighting_rounding_error": 0.0,
        }
        from terrashift.preprocessing.normalisation import pooled_statistics
        mean, std = pooled_statistics(normalisation, bands)
        print("class balance over %d patch instances: positive %d, negative %d, "
              "all-negative patches %d, all-positive patches %d"
              % (counts["patches"], counts["positive"], counts["negative"],
                 counts["all_negative_patches"], counts["all_positive_patches"]))
    else:
        statistics = weighted_training_statistics(train_raw)
        mean, std = statistics["mean"], statistics["std"]
    # With per-pair statistics the datasets normalise by pair and the single
    # mean/std below is kept only for reporting; the checkpoint records the path to
    # the per-pair file so inference reproduces the same transform.
    train_set = LocalManualPatchDataset(
        index_path, "train", manifest_path, registration_path,
        mean=(None if normalisation is not None else mean),
        std=(None if normalisation is not None else std),
        normalisation=normalisation,
        transform=(PairedRandomGeometry() if cfg.get("augmentation", {}).get("enabled", False)
                   else None), bands=bands)
    # Validation uses the inference lattice, not the split-contained index: the
    # windows are the ones full-tile inference runs, so validation scores the
    # predictions that deployment will make. Training keeps the index untouched.
    label_dir = Path(data_cfg["manual_label_dir"])
    partition_dir = Path(data_cfg["local_partition_dir"])
    loader_for_pair = pair_image_loader(manifest_path, registration_path, bands=bands)
    lattice_datasets = {}
    pair_names = sorted(path.name.split("_")[1].upper()
                        for path in label_dir.glob("pair_[abAB]_v2") if path.is_dir())
    for pair in pair_names:
        labels_path = label_dir / ("pair_%s_v2" % pair.lower()) / ("manual_labels_%s.tif" % pair)
        partition_path = partition_dir / ("pair_%s_partition_10m.tif" % pair)
        if not labels_path.is_file() or not partition_path.is_file():
            raise FileNotFoundError("validation data missing for pair %s" % pair)
        target, col_offset, tile_width = read_validation_target(labels_path,
                                                                partition_path)
        lattice_datasets[pair] = ValidationLatticeDataset(
            [pair], target, col_offset, tile_width, patch_size, stride,
            image_loader=lambda row_off, col_off, size, _pair=pair: loader_for_pair(
                _pair, row_off, col_off, size),
            normalisation=normalisation)
    if not lattice_datasets:
        raise ValueError("no validation pairs found for the inference lattice")
    reviewed_train = statistics["positive"] + statistics["negative"]
    if reviewed_train <= 0:
        raise ValueError("training annotations contain no reviewed pixels")
    fraction = statistics["positive"] / reviewed_train
    sample_rate = statistics.get("instance_counts", {}).get("mean_sample_rate")
    # args.pos_weight wins; otherwise use the configured value; only when both are
    # absent does resolve_pos_weight derive one from the pooled fraction.
    configured = getattr(args, "pos_weight", None)
    if configured is None:
        configured = cfg["training"].get("pos_weight")
    pos_weight = resolve_pos_weight(configured, fraction)
    pos_weight_source = ("cli" if getattr(args, "pos_weight", None) is not None
                         else ("config" if cfg["training"].get("pos_weight") is not None
                               else "derived from pooled patch-instance fraction"))
    pos_weight_derivation = {
        "reduction": "valid-pixel BCE averaged within each sample, then averaged "
                     "over samples",
        "mean_per_sample_positive_fraction": sample_rate,
        "pooled_patch_instance_positive_fraction": fraction,
        "formula": "(1 - mean_per_sample_positive_fraction) / mean_per_sample_positive_fraction",
        "note": ("the pooled instance fraction (0.2973) counts a geographic pixel once "
                 "per covering window; it is not the ratio this reduction sees"),
    }
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=(device.type == "cuda"), drop_last=False)
    lattice_loaders = {
        pair: DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=workers,
                         pin_memory=(device.type == "cuda"), drop_last=False)
        for pair, dataset in lattice_datasets.items()}

    print("dataset: local human-reviewed patches; device:", device)
    print("train patches:", len(train_set), "validation windows:",
          sum(len(d) for d in lattice_datasets.values()),
          "on the inference lattice", {p: len(d) for p, d in lattice_datasets.items()})
    print("official OSCD test labels are not used")
    print("pooled patch-instance positive fraction %.6f | mean per-sample %.6f| "
          "pos_weight %s (%s)"
          % (fraction, sample_rate if sample_rate is not None else float("nan"),
             "none" if pos_weight is None else "%.4f" % pos_weight, pos_weight_source))

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
        # bf16 autocast, no GradScaler: fp16 overflowed on every step for this model.
        amp_enabled = bool(cfg["training"]["amp"])
        amp_dtype = resolve_amp_dtype(device, amp_enabled)
        precision = ("bf16" if amp_dtype is torch.bfloat16
                     else ("fp32" if device.type == "cuda" else "cpu"))
        scaler = torch.amp.GradScaler("cuda", enabled=False)
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
            "validation_window_count": sum(len(d) for d in lattice_datasets.values()),
            "validation_windows_per_pair": {p: len(d) for p, d in lattice_datasets.items()},
            # Occurrence counts over patch instances, NOT unique geographic pixels:
            # overlapping windows repeat the same ground pixel (48-57 % of
            # occurrences are repeats at 256 px / stride 128).
            "train_reviewed_pixel_occurrences": statistics.get(
                "reviewed_pixel_occurrences", statistics.get("unique_pixels")),
            "train_patch_pixel_occurrences": statistics.get(
                "patch_pixel_occurrences", statistics.get("patch_pixels")),
            "train_class_counts_patch_instances": {
                "positive": statistics["positive"],
                "negative": statistics["negative"]},
            "train_patch_instance_counts": {
                "patches": statistics.get("instance_counts", {}).get("patches"),
                "all_negative": statistics.get("instance_counts", {}).get(
                    "all_negative_patches"),
                "all_positive": statistics.get("instance_counts", {}).get(
                    "all_positive_patches")},
            "count_basis": ("reviewed-pixel occurrences summed over patch instances; "
                            "overlapping windows repeat geographic pixels"),
            "validation_metrics_basis": "unique_geographic_pixels",
            "validation_aggregation_rule": "equal-weight mean of overlapping inference window probabilities",
            "validation_window_source": "terrashift.inference._window_origins via terrashift.data.validation_windows",
            "validation_target_source": "validation-only sanitised raster; test columns never read",
            "patch_size": patch_size,
            "patch_stride": stride,
            "min_labeled_fraction": index_meta["min_labeled_fraction"],
            "overlapping_patches_repeat_pixels_in_training": True,
            "test_dataset_opened": False,
            "test_index_included": bool(index_meta.get("test_index_included", False)),
            "bands": list(bands), "mean_train_only": mean.tolist(),
            "std_train_only": std.tolist(),
            "normalization": (
                "per-pair per-band z-score, pooled over both dates, unique TRAIN "
                "window pixels" if normalisation is not None else
                "per-band z-score from train labeled pixels; no inter-date correction"),
            "normalisation_file": (None if normalisation is None
                                   else normalisation["_file"]),
            "normalisation_sha256": (None if normalisation is None
                                     else normalisation["_sha256"]),
            "normalisation_statistics": (None if normalisation is None
                                         else normalisation["pairs"]),
            "normalisation_pooled_dates": True,
            "train_change_fraction": fraction,
            "loss": {"type": cfg["training"].get("loss", "bce_dice"),
                     "bce_weight": cfg["training"]["bce_weight"],
                     "dice_weight": cfg["training"]["dice_weight"]},
            "pos_weight": pos_weight,
            "pos_weight_source": pos_weight_source,
            "pos_weight_derivation": pos_weight_derivation,
            "seed": seed, "epochs": epochs,
            "batch_size": batch_size, "base_filters": base, "lr": lr,
            "workers": workers, "threshold": threshold,
            "selection_metric": "average_precision",
            "augmentation": cfg.get("augmentation", {"enabled": False}),
            "checkpoint_policy": {
                "last": "written after every completed epoch",
                "best": "written only when validation average_precision improves",
                "selection_metric": "average_precision",
                "selection_direction": "higher is better",
                "per_epoch_checkpoint_files": False,
                "resume_supported": True,
                "resume_granularity": "epoch boundary",
                "resume_note": ("--resume loads last.pt, restores model and AdamW "
                                "state plus RNG and best metadata, and continues at "
                                "the next epoch. Mid-epoch recovery is NOT supported: "
                                "an interrupted epoch is redone"),
                "atomic_writes": True,
                "rng_state_location": "last.pt only, so best.pt stays "
                                      "weights_only-loadable for inference",
            },
            "precision": {"amp_enabled": amp_enabled, "autocast_dtype": precision,
                          "grad_scaler": "disabled: bf16 needs none",
                          "gradient_clipping": "none in the training loop"},
            "device": str(device),
            "cli_overrides": {key: (str(value) if isinstance(value, Path) else value)
                              for key, value in vars(args).items()
                              if value is not None},
            "ignored_label_value": 255,
            "official_test_labels_read": False,
        }
        resume_requested = bool(getattr(args, "resume", False))
        start_epoch = 1
        best_ap = -1.0
        best_epoch = None
        best_metrics = None
        if resume_requested:
            resumed = resume_from_checkpoint(run_dir, device="cpu")
            check_compatibility(resumed["checkpoint"]["run_config"], resolved)
            model.load_state_dict(resumed["checkpoint"]["model_state_dict"], strict=True)
            optimizer.load_state_dict(resumed["checkpoint"]["optimizer_state_dict"])
            start_epoch = resumed["start_epoch"]
            best_ap = resumed["best_ap"]
            best_epoch = resumed["best_epoch"]
            best_metrics = resumed["best_metrics"]
            print("resume: continuing at epoch %d after checkpoint epoch %d "
                  "(best AP %.4f at epoch %s)"
                  % (start_epoch, resumed["last_epoch"], best_ap, best_epoch))
            repair = resumed.get("best_repaired") or {}
            if repair.get("repaired"):
                print("resume: best.pt was stale and has been restored from last.pt "
                      "(best epoch %d)" % repair["best_epoch"])
            elif repair:
                print("resume: best.pt unchanged (%s)" % repair.get("reason"))
        else:
            (run_dir / "run_config.json").write_text(
                json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
        log_path = run_dir / "train_log.csv"
        if resume_requested:
            dropped = truncate_train_log(log_path, start_epoch - 1)
            print("resume: train_log rows kept %d, dropped %d (uncommitted)"
                  % (dropped["kept"], dropped["dropped"]))
        fields = ["epoch", "train_loss", "train_ap", "train_auc", "train_f1",
                  "val_loss", "val_ap", "val_auc", "val_f1", "val_iou",
                  "val_ap_A", "val_ap_B", "val_reviewed_A", "val_reviewed_B"]
        best = ({key: best_metrics.get(key, float("nan")) for key in ("ap", "auc", "f1", "iou")}
                if best_metrics else {key: float("nan")
                                      for key in ("ap", "auc", "f1", "iou")})
        # Restore RNG after model, optimizer and loaders exist, before the first
        # iterator of the resumed run is created.
        if resume_requested:
            restore_rng_state(resumed["rng_state"])
        log_mode = "a" if resume_requested and log_path.is_file() else "w"
        with log_path.open(log_mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if log_mode == "w":
                writer.writeheader()
            for epoch in range(start_epoch, epochs + 1):
                train_metrics = run_epoch(
                    model, train_loader, criterion, device, optimizer=optimizer,
                    scaler=scaler, amp=amp_enabled,
                    max_batches=args.max_train_batches, collect=True,
                    threshold=threshold)
                val_metrics = evaluate_lattice_all(
                    model, {pair: (loader, lattice_datasets[pair])
                            for pair, loader in lattice_loaders.items()},
                    device, threshold=threshold, criterion=criterion,
                    max_batches=args.max_val_batches)
                row = {
                    "epoch": epoch, "train_loss": train_metrics["loss"],
                    "train_ap": train_metrics["ap"], "train_auc": train_metrics["auc"],
                    "train_f1": train_metrics["f1"], "val_loss": val_metrics["loss"],
                    "val_ap": val_metrics["ap"], "val_auc": val_metrics["auc"],
                    "val_f1": val_metrics["f1"], "val_iou": val_metrics["iou"],
                }
                row.update(per_pair_log_values(val_metrics))
                writer.writerow(row)
                fh.flush()
                print("seed %d epoch %d/%d train AP %.4f | val AP %.4f F1 %.4f IoU %.4f" % (
                    seed, epoch, epochs, train_metrics["ap"], val_metrics["ap"],
                    val_metrics["f1"], val_metrics["iou"]))
                payload = {
                    "epoch": epoch, "epochs_planned": epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "bands": list(bands), "mean": mean.tolist(),
                    "std": std.tolist(), "base": base, "seed": seed,
                    "pos_weight": pos_weight,
                    "pos_weight_derivation": pos_weight_derivation,
                    "val_metrics": val_metrics,
                    "precision": precision,
                    "run_config": resolved,
                }
                best_ap, best_epoch, best_metrics, improved = save_epoch_checkpoints(
                    run_dir, payload, val_metrics["ap"], best_ap, best_epoch,
                    best_metrics)
                if improved:
                    best = {key: val_metrics[key] for key in best}
                print("  checkpoint: last.pt epoch %d%s"
                      % (epoch, ", best.pt updated" if improved else ""))
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
