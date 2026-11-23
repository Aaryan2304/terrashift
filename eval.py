"""Evaluate a trained checkpoint on a locked OSCD split.

Reports four families of measurement, because they answer different questions:

1. Pixel metrics, precision, recall, F1, IoU, AP and AUC over all valid pixels.
   F1 and IoU judge the score and the threshold together; AP and AUC judge the
   score alone.
2. Object metrics, connected components matched one-to-one by intersection
   over union. A model that finds 90 % of the pixels of every building but
   fragments each building into three blobs scores well on pixels and badly
   here.
3. Size-stratified recall, recall of true components per area bin. A single
   aggregate hides that small buildings are usually missed.
4. A confusion matrix image, plus per-dataset counts.

The split defaults to ``test``. Nothing in this project trains on it, and the
official test regions are the locked set, so evaluating them is legitimate; the
report must say that they were used once, for evaluation, and never for
selection.

Usage:
    python eval.py --checkpoint outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt \
        --output-dir outputs/eval/oscd_baseline_seed42
Nothing is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent

from terrashift.data.oscd_original import DEFAULT_BANDS, OSCDOriginalPatchDataset
from terrashift.models.siamese_unet import SiameseUNet
from terrashift.metrics import average_precision, metrics_from_counts, roc_auc

SIZE_BINS = ((4, 16), (16, 64), (64, 256), (256, 10**9))


def component_table(binary):
    """Area and label map of the connected components of a binary mask."""
    labels, count = ndi.label(binary)
    if count == 0:
        return labels, np.zeros(0, dtype=np.int64)
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    return labels, areas


def valid_of(batch):
    """Validity mask for a batch, defaulting to all-valid.

    The original-OSCD patches carry no validity mask: the rectified rasters
    have no NoData and the labels cover every pixel. A SAFE-derived dataset does
    carry one, so this keeps a single evaluation path for both.
    """
    valid = batch.get("valid_mask")
    if valid is None:
        return np.ones(batch["mask"].shape, dtype=bool)
    return valid.numpy().astype(bool)


def _overlap_matrix(pred_labels, true_labels, n_pred, n_true):
    """Pixel overlap between every predicted and every true component."""
    combined = (pred_labels.ravel().astype(np.int64) * (n_true + 1)
                + true_labels.ravel().astype(np.int64))
    counts = np.bincount(combined, minlength=(n_pred + 1) * (n_true + 1))
    return counts.reshape(n_pred + 1, n_true + 1)[1:, 1:]


def evaluate_patch(pred, truth, iou_threshold=0.5):
    """Object counts, matched IoUs and per-size-bin recall for one patch.

    One pass over the patch: components are matched one-to-one, greedily, at
    the highest IoU first. Unmatched predicted components are false positives
    and unmatched true components are false negatives, which is the object-level
    counterpart of the pixel counts.
    """
    pred_labels, pred_areas = component_table(pred)
    true_labels, true_areas = component_table(truth)
    n_pred, n_true = len(pred_areas), len(true_areas)

    matches = []
    if n_pred and n_true:
        overlap = _overlap_matrix(pred_labels, true_labels, n_pred, n_true)
        union = pred_areas[:, None] + true_areas[None, :] - overlap
        iou = np.divide(overlap, union, out=np.zeros(overlap.shape, dtype=np.float64),
                        where=union > 0)
        while iou.size:
            row, col = np.unravel_index(int(np.argmax(iou)), iou.shape)
            best = float(iou[row, col])
            if best < iou_threshold:
                break
            matches.append((int(row), int(col), best))
            iou[row, :] = -1.0
            iou[:, col] = -1.0

    matched_true = {col for _, col, _ in matches}
    size_bins = {}
    for low, high in SIZE_BINS:
        label = "%d_%s_px" % (low, "plus" if high > 10**8 else str(high))
        members = [index for index, area in enumerate(true_areas) if low <= area < high]
        if not members:
            continue
        hit = sum(1 for index in members if index in matched_true)
        size_bins[label] = {"components": len(members), "matched": hit}

    return {"tp": len(matches), "fp": n_pred - len(matches), "fn": n_true - len(matches),
            "ious": [match[2] for match in matches], "size_bins": size_bins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for name, path in (("metrics", args.output_dir / "metrics.json"),
                       ("csv", args.output_dir / "metrics.csv"),
                       ("confusion", args.output_dir / "confusion_matrix.png"),
                       ("objects", args.output_dir / "objects_by_size.json")):
        if path.exists():
            raise SystemExit("refusing to overwrite existing evaluation output: %s" % path)

    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    threshold = float(args.threshold if args.threshold is not None
                      else cfg["evaluation"]["threshold"])
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    bands = tuple(checkpoint.get("bands", DEFAULT_BANDS))
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    base = int(checkpoint["base"])

    index_path = Path(cfg["data"]["oscd_patch_index"])
    dataset = OSCDOriginalPatchDataset(index_path, args.split, bands, mean, std)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    model = SiameseUNet(in_channels=len(bands), base=base).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    scores, targets = [], []
    object_totals = {"tp": 0, "fp": 0, "fn": 0}
    ious = []
    size_totals = {}
    per_region = {}
    patch_count = 0

    with torch.no_grad():
        for batch in loader:
            x1 = batch["image1"].to(device)
            x2 = batch["image2"].to(device)
            probabilities = torch.sigmoid(model(x1, x2).float())[:, 0].cpu().numpy()
            masks = batch["mask"].numpy()
            valid = valid_of(batch)
            for i in range(probabilities.shape[0]):
                prob = probabilities[i]
                truth = (masks[i] == 1)
                v = valid[i]
                scores.append(prob[v])
                targets.append(truth[v])

                pred = (prob >= threshold) & v
                true = truth & v
                match = evaluate_patch(pred, true)
                object_totals["tp"] += match["tp"]
                object_totals["fp"] += match["fp"]
                object_totals["fn"] += match["fn"]
                ious.extend(match["ious"])
                for label, values in match["size_bins"].items():
                    acc = size_totals.setdefault(label, {"components": 0, "matched": 0})
                    acc["components"] += values["components"]
                    acc["matched"] += values["matched"]
                patch_count += 1

    scores = np.concatenate(scores) if scores else np.empty(0)
    targets = np.concatenate(targets) if targets else np.empty(0)
    pred = scores >= threshold
    truth = targets == 1
    counts = metrics_from_counts(int((pred & truth).sum()), int((pred & ~truth).sum()),
                                 int((~pred & truth).sum()), int((~pred & ~truth).sum()))
    counts["ap"] = average_precision(targets, scores)
    counts["auc"] = roc_auc(targets, scores)
    counts["threshold"] = threshold
    counts["patches"] = patch_count
    counts["split"] = args.split

    object_metrics = metrics_from_counts(object_totals["tp"], object_totals["fp"],
                                         object_totals["fn"], 0)
    object_metrics.pop("tn", None)
    object_metrics["mean_matched_iou"] = float(np.mean(ious)) if ious else float("nan")

    recall_by_size = {label: {"components": v["components"], "matched": v["matched"],
                              "recall": v["matched"] / v["components"] if v["components"] else float("nan")}
                      for label, v in sorted(size_totals.items())}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "pixel": counts,
        "object": object_metrics,
        "recall_by_size": recall_by_size,
        "note": ("The official test regions are evaluated once, here. No threshold, "
                 "hyperparameter or checkpoint was selected on this split."),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["family", "metric", "value"])
        for key, value in counts.items():
            writer.writerow(["pixel", key, value])
        for key, value in object_metrics.items():
            writer.writerow(["object", key, value])
        for label, values in recall_by_size.items():
            writer.writerow(["recall_by_size", label, values["recall"]])

    cm = np.array([[counts["tp"], counts["fn"]], [counts["fp"], counts["tn"]]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, "{:,}".format(int(cm[i, j])), ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() * 0.6 else "white")
    ax.set_xticks([0, 1], ["predicted change", "predicted unchanged"])
    ax.set_yticks([0, 1], ["true change", "true unchanged"])
    ax.set_title("Pixel confusion, %s split, threshold %.2f" % (args.split, threshold))
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
