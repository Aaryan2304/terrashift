"""Validation-only dry run on the real pairs, through the trainer's own path.

Loads the OSCD checkpoint into the local architecture, builds the 128/64 validation
lattice per pair, and runs `evaluate_lattice_all`, the function `train_local_manual`
calls, so this is the real evaluation path, not a reimplementation.

Read-only: no index, no weights updated, no threshold tuning, no model selection,
no test labels. The metrics are the **untrained-for-this-domain baseline checkpoint**
and must not be read as detection performance.

    python scripts/inference/validation_dry_run.py --output outputs/audit/validation_dry_run/run.json
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.local_patches import BANDS, pair_image_loader
from terrashift.data.validation_windows import ValidationLatticeDataset, read_validation_target
from terrashift.models.siamese_unet import DiceBCELoss, SiameseUNet
from terrashift.evaluation.lattice import evaluate_lattice_all

PATCH = 128
STRIDE = 64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-dir", default=os.path.join(ROOT, "outputs", "labels"))
    parser.add_argument("--partition-dir",
                        default=os.path.join(ROOT, "outputs",
                                             "partitions_v2_observation_valid"))
    parser.add_argument("--manifest", default=os.path.join(ROOT, "configs",
                                                           "data_manifest.yaml"))
    parser.add_argument("--registration",
                        default=os.path.join(ROOT, "outputs", "audit",
                                             "registration_v2_observation_valid",
                                             "registration.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(state["bands"]) != tuple(BANDS):
        raise ValueError("checkpoint bands do not match the local band order")
    bands = tuple(BANDS)
    model = SiameseUNet(in_channels=len(bands), base=int(state["base"]))
    model.load_state_dict(state["model_state_dict"], strict=True)
    model = model.to(device).eval()

    # Local statistics, exactly as the trainer derives them: TRAIN patches only.
    loader_for_pair = pair_image_loader(args.manifest, args.registration, bands=bands)
    mean, std = _train_statistics(args, loader_for_pair, bands)
    print("train-only mean:", np.round(mean, 5).tolist())
    print("train-only std :", np.round(std, 5).tolist())

    loaders = {}
    image_range = {band: [np.inf, -np.inf] for band in bands}
    target_range = [255, 0]
    for pair in ("A", "B"):
        labels_path = os.path.join(args.labels_dir, "pair_%s_v2" % pair.lower(),
                                   "manual_labels_%s.tif" % pair)
        partition_path = os.path.join(args.partition_dir,
                                      "pair_%s_partition_10m.tif" % pair)
        target, col0, tile_width = read_validation_target(labels_path, partition_path)
        if np.any(target == 3):
            raise ValueError("a test label value reached the validation target")
        dataset = ValidationLatticeDataset(
            [pair], target, col0, tile_width, PATCH, STRIDE,
            image_loader=lambda r, c, s, _p=pair: loader_for_pair(_p, r, c, s),
            mean=mean, std=std)
        targets = [dataset[i] for i in range(len(dataset))]
        for sample in targets:
            image = sample["image1"].numpy()
            image_range_map = {bands[i]: [float(image[i].min()), float(image[i].max())]
                               for i in range(len(bands))}
            for band, (low, high) in image_range_map.items():
                image_range[band][0] = min(image_range[band][0], low)
                image_range[band][1] = max(image_range[band][1], high)
        valid = torch.stack([s["valid_mask"] for s in targets]).numpy()
        target_range[1] = max(target_range[1], int(valid.sum()))
        loaders[pair] = (DataLoader(targets, batch_size=args.batch_size, shuffle=False),
                         dataset)
        print("pair %s: %d windows, %d stripe pixels"
              % (pair, len(dataset), int(target.size)))

    criterion = DiceBCELoss(pos_weight=None).to(device)
    metrics = evaluate_lattice_all(model, loaders, device, threshold=0.5,
                                   criterion=criterion)
    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "patch": PATCH, "stride": STRIDE, "batch_size": args.batch_size,
        "train_only_mean": [float(v) for v in mean],
        "train_only_std": [float(v) for v in std],
        "checkpoint_stored_mean": [float(v) for v in state["mean"]],
        "checkpoint_stored_std": [float(v) for v in state["std"]],
        "image_value_range_zscored": {k: v for k, v in image_range.items()},
        "target_values": "0, 1 and the ignore value 255 only",
        "status": "baseline checkpoint evaluated through the trainer path; not a "
                  "detection result and not a model selection",
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1, default=float)
    print("wrote", args.output)
    print("reviewed validation pixels:", metrics["unique_valid_pixels"],
          "| positives:", metrics["unique_positive_pixels"],
          "| windows:", metrics["validation_windows"])
    print("AP %.4f | AUC %.4f | F1 %.4f | IoU %.4f | loss %.4f"
          % (metrics["ap"], metrics["auc"], metrics["f1"], metrics["iou"], metrics["loss"]))
    for pair, info in metrics["per_pair"].items():
        print("  %s: %d windows, %d pixels, votes min/mean/max %d/%.2f/%d, "
              "finite %.4f, AP %.4f"
              % (pair, info["windows"], info["reviewed_pixels"], info["votes"]["min"],
                 info["votes"]["mean"], info["votes"]["max"],
                 info["finite_score_fraction"], info["ap"]))


def _mean_std_from_channels(per_channel):
    """Per-band mean and std. Each channel divides by its own pixel count.

    A single counter shared across channels divides every mean by the channel
    count as well as the pixel count, which is the bug this function exists to
    prevent: it made a 0.138 mean report as 0.039.
    """
    values = [np.concatenate(channel).astype(np.float64) if len(channel)
              else np.empty(0, dtype=np.float64) for channel in per_channel]
    counts = np.array([v.size for v in values], dtype=np.int64)
    if not np.all(counts):
        raise ValueError("a band had no finite values: %s" % counts.tolist())
    sums = np.array([v.sum() for v in values], dtype=np.float64)
    squares = np.array([(v ** 2).sum() for v in values], dtype=np.float64)
    mean = sums / counts
    variance = np.maximum(squares / counts - mean ** 2, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _train_statistics(args, loader, bands, windows=25):
    """Per-band mean/std from TRAIN-labelled imagery, as the trainer does it."""
    import rasterio
    from terrashift.data.local_patches import patch_origins

    collected = [[] for _ in bands]
    for pair in ("A", "B"):
        labels_path = os.path.join(args.labels_dir, "pair_%s_v2" % pair.lower(),
                                   "manual_labels_%s.tif" % pair)
        partition_path = os.path.join(args.partition_dir,
                                      "pair_%s_partition_10m.tif" % pair)
        with rasterio.open(labels_path) as handle:
            labels = handle.read(1)
        with rasterio.open(partition_path) as handle:
            partition = handle.read(1)
        train = (labels != 255) & (partition == 1)
        if not train.any():
            continue
        # Same sampling rule as the preprocessing audit: lattice windows that
        # contain reviewed train pixels, taken at a fixed stride.
        kept = [(r, c) for r in patch_origins(labels.shape[0], PATCH, PATCH // 2)
                for c in patch_origins(labels.shape[1], PATCH, PATCH // 2)
                if train[r:r + PATCH, c:c + PATCH].any()]
        if not kept:
            continue
        step = max(1, len(kept) // windows)
        for r0, c0 in kept[::step][:windows]:
            inside = train[r0:r0 + PATCH, c0:c0 + PATCH]
            if inside.any():
                for date_index in (0, 1):
                    image = loader(pair, r0, c0, PATCH)[date_index]
                    for channel in range(len(bands)):
                        values = image[channel][inside]
                        values = values[np.isfinite(values)]
                        collected[channel].append(values)
    return _mean_std_from_channels(collected)


if __name__ == "__main__":
    main()
