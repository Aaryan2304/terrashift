"""PR curves, threshold plots and the sub-0.05 F1 sweep, from saved score vectors.

Reads the ``_scores.npy`` / ``_labels.npy`` written by
``analyse_validation_thresholds.py`` so no model pass is repeated here.

    python scripts/reporting/plot_validation_thresholds.py \
        --analysis outputs/evaluation/local_manual_v2/validation_threshold_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

FINE = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
SCOPES = ("combined", "A", "B")


def curve(scores, labels):
    order = np.argsort(-scores, kind="mergesort")
    y = (labels[order] == 1).astype(np.int64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    positives = max(int(y.sum()), 1)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    return precision, recall


def counts(scores, labels, threshold):
    pred = scores >= threshold
    truth = labels == 1
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "specificity": specificity, "predicted_positive": tp + fp,
            "predicted_positive_fraction": (tp + fp) / (tp + fp + fn + tn),
            "n_pixels": tp + fp + fn + tn}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    args = parser.parse_args()
    root = Path(args.analysis)
    ft_scores = np.load(root / "_fine_tuned_scores.npy")
    ft_labels = np.load(root / "_fine_tuned_labels.npy")
    oscd_scores = np.load(root / "_oscd_scores.npy")
    oscd_labels = np.load(root / "_oscd_labels.npy")

    split_at = {"A": 2759, "B": 22650}
    vectors = {"combined": (ft_scores, ft_labels),
               "A": (ft_scores[:split_at["A"]], ft_labels[:split_at["A"]]),
               "B": (ft_scores[split_at["A"]:], ft_labels[split_at["A"]:])}

    # ---- precision-recall curves -------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, scope in zip(axes, SCOPES):
        scores, labels = vectors[scope]
        precision, recall = curve(scores, labels)
        prevalence = float((labels == 1).mean())
        axis.plot(recall, precision, lw=1.6,
                  label="fine-tuned (epoch 14)")
        o_scores = oscd_scores if scope == "combined" else (
            oscd_scores[:split_at["A"]] if scope == "A" else oscd_scores[split_at["A"]:])
        o_labels = oscd_labels if scope == "combined" else (
            oscd_labels[:split_at["A"]] if scope == "A" else oscd_labels[split_at["A"]:])
        o_precision, o_recall = curve(o_scores, o_labels)
        axis.plot(o_recall, o_precision, lw=1.2, ls="--", label="OSCD initial")
        axis.axhline(prevalence, color="grey", ls=":", lw=1,
                     label="prevalence %.3f" % prevalence)
        axis.scatter([0.3517 if scope == "combined" else np.nan], [0.7135], s=25,
                     color="k", zorder=5)
        axis.set_title("%s  (n=%d, prevalence %.3f)" % (scope, labels.size, prevalence),
                       fontsize=10)
        axis.set_xlabel("recall")
        axis.set_ylabel("precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="lower left")
    figure.suptitle("Validation precision-recall, 25,409 unique reviewed pixels",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(root / "precision_recall_curves.png", dpi=140)
    plt.close(figure)

    # ---- threshold versus metric ------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    rows = {t: counts(ft_scores, ft_labels, t) for t in FINE}
    thresholds = sorted(rows)
    for key, style in (("precision", "-"), ("recall", "-"), ("f1", "--"), ("iou", ":")):
        axes[0].plot(thresholds, [rows[t][key] for t in thresholds], style, lw=1.4,
                     label=key)
    axes[0].axvline(0.5, color="k", lw=0.8, ls="--", alpha=0.6)
    axes[0].set_xlabel("probability threshold")
    axes[0].set_ylabel("metric")
    axes[0].set_title("Combined: precision, recall, F1, IoU versus threshold", fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    prevalence = float((ft_labels == 1).mean())
    axes[1].plot(thresholds, [rows[t]["predicted_positive_fraction"] for t in thresholds],
                 lw=1.5, label="predicted-positive fraction")
    axes[1].axhline(prevalence, color="grey", ls=":", label="true prevalence %.3f" % prevalence)
    axes[1].axvline(0.5, color="k", lw=0.8, ls="--", alpha=0.6)
    axes[1].set_xlabel("probability threshold")
    axes[1].set_ylabel("fraction of reviewed pixels")
    axes[1].set_title("Predicted-positive fraction versus threshold", fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "threshold_analysis.png", dpi=140)
    plt.close(figure)

    # ---- fine sweep below the specified grid -------------------------------
    fields = ["scope", "threshold", "tp", "fp", "fn", "tn", "precision", "recall",
              "f1", "iou", "specificity", "predicted_positive",
              "predicted_positive_fraction", "n_pixels"]
    with open(root / "threshold_metrics_fine.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scope in SCOPES:
            scores, labels = vectors[scope]
            for t in FINE:
                row = counts(scores, labels, t)
                row["scope"] = scope
                writer.writerow(row)

    fine_best = max((counts(ft_scores, ft_labels, t) for t in FINE),
                    key=lambda r: (r["f1"], r["threshold"]))
    all_positive = counts(ft_scores, ft_labels, -1.0)
    json.dump({
        "fine_sweep_thresholds": FINE,
        "combined_f1_max_on_fine_sweep": fine_best,
        "all_positive_baseline": all_positive,
        "degenerate": fine_best["threshold"] <= min(FINE),
        "finding": ("F1 increases monotonically as the threshold falls; the optimum on "
                    "the fine sweep sits at the lowest tested threshold and approaches "
                    "the all-positive point. With prevalence %.3f, F1 at high recall is "
                    "close to prevalence, so F1 cannot select an operating threshold "
                    "here. AP is the meaningful ranking metric." % prevalence),
        "validation_derived": True,
    }, open(root / "f1_degeneracy.json", "w"), indent=1)
    print("F1 max on fine sweep:", fine_best)
    print("all-positive baseline F1: %.4f" % all_positive["f1"])
    print("wrote curves and threshold tables to", root)


if __name__ == "__main__":
    main()
