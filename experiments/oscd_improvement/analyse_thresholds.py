"""Threshold, framing and post-processing analysis on stitched validation data.

Three questions, all answered from one saved prediction file:

1. **Framing.** The baseline evaluates object metrics per 128 px patch. Objects
   that cross a patch edge are split, and the edge patches overlap, so the same
   object can be counted several times. This compares that definition with
   scene-level counting using the same matching function.
2. **Threshold.** The baseline uses 0.5. This sweeps it and reports pixel and
   object metrics, selecting on validation only.
3. **Post-processing.** Minimum component area, opening and closing, measured on
   both pixel and object metrics.

Writes CSV, JSON and a PR figure into the output directory. Never overwrites.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data.oscd_original import patch_origins
from eval import evaluate_patch
from utils.metrics import average_precision, metrics_from_counts, roc_auc

PATCH_SIZE = 128
# 0.05 to 0.95 in steps of 0.05, plus a finer tail where the model saturates.
THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 20)] + [0.97, 0.99]
MIN_AREA_PX = [0, 4, 16, 64, 256, 1024]


def load(path):
    data = np.load(path)
    cities = sorted({key.split("_", 1)[1] for key in data.files if key.startswith("prob_")})
    return {city: {"prob": data["prob_" + city], "truth": data["truth_" + city]}
            for city in cities}


def pixel_counts(prob, truth, threshold):
    pred = prob >= threshold
    target = truth == 1
    return (int((pred & target).sum()), int((pred & ~target).sum()),
            int((~pred & target).sum()), int((~pred & ~target).sum()))


def object_counts_scene(prob, truth, threshold):
    """Object metrics over the whole scene, one matching pass."""
    result = evaluate_patch(prob >= threshold, truth == 1)
    return result["tp"], result["fp"], result["fn"], result["ious"]


def object_counts_patchwise(prob, truth, threshold):
    """The baseline framing: one matching pass per 128 px window."""
    tp = fp = fn = 0
    ious = []
    height, width = prob.shape
    for r0 in patch_origins(height, PATCH_SIZE, PATCH_SIZE):
        for c0 in patch_origins(width, PATCH_SIZE, PATCH_SIZE):
            window = np.s_[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]
            result = evaluate_patch(prob[window] >= threshold, truth[window] == 1)
            tp += result["tp"]
            fp += result["fp"]
            fn += result["fn"]
            ious.extend(result["ious"])
    return tp, fp, fn, ious


def summarise(counts, ious):
    tp, fp, fn = counts
    metrics = metrics_from_counts(tp, fp, fn, 0)
    metrics.pop("tn", None)
    metrics["mean_matched_iou"] = float(np.mean(ious)) if ious else float("nan")
    metrics["tp"], metrics["fp"], metrics["fn"] = tp, fp, fn
    return metrics


def remove_small(mask, min_area):
    if min_area <= 0:
        return mask
    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def clean(mask, open_size, close_size, min_area):
    out = mask
    if open_size > 1:
        out = ndi.binary_opening(out, np.ones((open_size, open_size), dtype=bool))
    if close_size > 1:
        out = ndi.binary_closing(out, np.ones((close_size, close_size), dtype=bool))
    return remove_small(out, min_area)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    for name in ("thresholds.csv", "postprocess.csv", "analysis.json", "pr_curve.png"):
        if os.path.exists(os.path.join(args.output_dir, name)):
            raise SystemExit("refusing to overwrite %s" % name)
    os.makedirs(args.output_dir, exist_ok=True)

    data = load(args.predictions)
    print("cities:", {city: info["prob"].shape for city, info in data.items()})

    # ---------- 1. framing at the baseline threshold
    print("\n=== object metric framing at threshold 0.5 ===")
    framing = {}
    for city, info in data.items():
        scene = object_counts_scene(info["prob"], info["truth"], 0.5)
        patch = object_counts_patchwise(info["prob"], info["truth"], 0.5)
        framing[city] = {"scene": summarise(scene[:3], scene[3]),
                         "patch": summarise(patch[:3], patch[3])}
        print("%-10s scene  tp %4d fp %5d fn %3d  F1 %.4f"
              % (city, scene[0], scene[1], scene[2],
                 framing[city]["scene"]["f1"]))
        print("%-10s patch  tp %4d fp %5d fn %3d  F1 %.4f"
              % ("", patch[0], patch[1], patch[2], framing[city]["patch"]["f1"]))

    # ---------- 2. threshold sweep, scene-level objects
    print("\n=== threshold sweep (scene-level objects) ===")
    rows = []
    all_scores, all_targets = [], []
    for city, info in data.items():
        all_scores.append(info["prob"].ravel())
        all_targets.append((info["truth"] == 1).ravel())
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)

    for threshold in THRESHOLDS:
        tp = fp = fn = tn = 0
        otp = ofp = ofn = 0
        ious = []
        for info in data.values():
            t, f, m, n = pixel_counts(info["prob"], info["truth"], threshold)
            tp += t; fp += f; fn += m; tn += n
            a, b, c, d = object_counts_scene(info["prob"], info["truth"], threshold)
            otp += a; ofp += b; ofn += c; ious.extend(d)
        pixel = metrics_from_counts(tp, fp, fn, tn)
        obj = summarise((otp, ofp, ofn), ious)
        rows.append({
            "threshold": threshold,
            "precision": round(pixel["precision"], 4), "recall": round(pixel["recall"], 4),
            "f1": round(pixel["f1"], 4), "iou": round(pixel["iou"], 4),
            "tp": tp, "fp": fp, "fn": fn,
            "object_precision": round(obj["precision"], 4),
            "object_recall": round(obj["recall"], 4),
            "object_f1": round(obj["f1"], 4),
            "object_tp": otp, "object_fp": ofp, "object_fn": ofn,
            "object_mean_iou": round(obj["mean_matched_iou"], 4),
        })
        print("  thr %.2f  pixel P %.3f R %.3f F1 %.3f | object P %.3f R %.3f F1 %.4f"
              " (tp %d fp %d fn %d)"
              % (threshold, pixel["precision"], pixel["recall"], pixel["f1"],
                 obj["precision"], obj["recall"], obj["f1"], otp, ofp, ofn))

    best_f1 = max(rows, key=lambda r: r["f1"])
    best_obj = max(rows, key=lambda r: r["object_f1"])
    ap = average_precision(all_targets, all_scores)
    auc = roc_auc(all_targets, all_scores)
    print("\n  AP %.4f  AUC %.4f" % (ap, auc))
    print("  best pixel F1 threshold %.2f (F1 %.4f)" % (best_f1["threshold"], best_f1["f1"]))
    print("  best object F1 threshold %.2f (F1 %.4f)"
          % (best_obj["threshold"], best_obj["object_f1"]))

    with open(os.path.join(args.output_dir, "thresholds.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---------- 3. post-processing at the best pixel-F1 threshold
    print("\n=== post-processing at threshold %.2f ===" % best_f1["threshold"])
    threshold = best_f1["threshold"]
    post = []
    for open_size, close_size, min_area in (
            (1, 1, 0), (1, 1, 4), (1, 1, 16), (1, 1, 64), (1, 1, 256),
            (3, 1, 0), (3, 1, 16), (3, 5, 16), (3, 5, 64)):
        tp = fp = fn = 0
        otp = ofp = ofn = 0
        ious = []
        for info in data.values():
            mask = clean(info["prob"] >= threshold, open_size, close_size, min_area)
            truth = info["truth"] == 1
            tp += int((mask & truth).sum())
            fp += int((mask & ~truth).sum())
            fn += int((~mask & truth).sum())
            a, b, c, d = object_counts_scene(info["prob"] * mask, info["truth"], 0.5)
            otp += a; ofp += b; ofn += c; ious.extend(d)
        pixel = metrics_from_counts(tp, fp, fn, 0)
        obj = summarise((otp, ofp, ofn), ious)
        post.append({
            "open": open_size, "close": close_size, "min_area_px": min_area,
            "precision": round(pixel["precision"], 4),
            "recall": round(pixel["recall"], 4), "f1": round(pixel["f1"], 4),
            "iou": round(pixel["iou"], 4),
            "object_precision": round(obj["precision"], 4),
            "object_recall": round(obj["recall"], 4),
            "object_f1": round(obj["f1"], 4),
            "object_tp": otp, "object_fp": ofp, "object_fn": ofn,
        })
        print("  open %d close %d min %4d  pixel F1 %.4f | object F1 %.4f (fp %d)"
              % (open_size, close_size, min_area, pixel["f1"], obj["f1"], ofp))

    with open(os.path.join(args.output_dir, "postprocess.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(post[0]))
        writer.writeheader()
        writer.writerows(post)

    summary = {
        "predictions": os.path.abspath(args.predictions),
        "thresholds": THRESHOLDS,
        "average_precision": ap, "roc_auc": auc,
        "best_pixel_f1_threshold": best_f1["threshold"],
        "best_pixel_f1": best_f1["f1"],
        "best_object_f1_threshold": best_obj["threshold"],
        "best_object_f1": best_obj["object_f1"],
        "framing_at_0.5": framing,
        "postprocess": post,
        "notes": [
            "Validation split only; the official test regions were not read.",
            "Scene-level object counting reuses eval.evaluate_patch on the full "
            "scene array, so the matching rule is identical to the baseline.",
        ],
    }
    with open(os.path.join(args.output_dir, "analysis.json"), "w") as handle:
        json.dump(summary, handle, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.plot([r["recall"] for r in rows], [r["precision"] for r in rows],
                  marker="o", label="pixel PR")
        for row in rows:
            axis.annotate("%.2f" % row["threshold"], (row["recall"], row["precision"]),
                          fontsize=6, xytext=(2, 2), textcoords="offset points")
        axis.set_xlabel("recall")
        axis.set_ylabel("precision")
        axis.set_title("Validation precision-recall, baseline seed 42")
        axis.grid(alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "pr_curve.png"), dpi=130)
    except Exception as error:                       # noqa: BLE001 - figure is optional
        print("figure skipped:", error)

    print("\nwrote", args.output_dir)


if __name__ == "__main__":
    main()
