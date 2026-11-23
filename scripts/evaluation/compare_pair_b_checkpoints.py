"""Pair B: why did fine-tuning lower validation AP from 0.6974 to 0.6448?

Diagnostic only. Reuses the saved score vectors from the validation threshold analysis
after verifying their provenance and ordering, reuses ``predict_pair`` and
``LatticeAccumulator`` for the spatial maps, and reuses the project's metric functions.

    python scripts/evaluation/compare_pair_b_checkpoints.py \
        --output outputs/evaluation/local_manual_v2/pair_b_checkpoint_comparison
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from matplotlib import pyplot as plt
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.sessions import build_datasets, load_model, predict_pair
from terrashift.geospatial.rendering import rgb_background
from terrashift.metrics import average_precision, roc_auc

PAIR = "B"
SPLIT_A = 2759
HASHES = {
    "fine_tuned": "4c083af28d6e6ee43c7ca117355e794ad570412310c7856026b717f283de0546",
    "oscd": "4d9c381dba370becc3ac3ff93431db65929cf21378287191a9f131fae9080d1a",
    "normalisation": "8aeea033eb2bec5012ebbceb7eca825814b6e43bb699f9e2eaee0c5bba955d2a",
}
EXPECTED = {"reviewed": 22650, "positive": 11940, "negative": 10710}
COLOURS = {"both_correct": (0.15, 0.70, 0.25), "regression": (0.90, 0.15, 0.15),
           "improvement": (0.15, 0.40, 0.95), "both_wrong": (0.45, 0.45, 0.45)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(scores, labels, threshold=0.5):
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
    return {"ap": float(average_precision(labels, scores)),
            "roc_auc": float(roc_auc(labels, scores)),
            "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "specificity": specificity,
            "predicted_positive": tp + fp,
            "predicted_positive_fraction": (tp + fp) / labels.size,
            "n_pixels": int(labels.size)}


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(values.size), "mean": float(values.mean()),
            "std": float(values.std()), "median": float(np.median(values)),
            "p01": float(np.percentile(values, 1)), "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "min": float(values.min()), "max": float(values.max())}


def precision_at_recall(scores, labels, target):
    order = np.argsort(-scores, kind="mergesort")
    y = (labels[order] == 1).astype(np.int64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    recall = tp / max(int(y.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    idx = np.searchsorted(recall, target, side="left")
    if idx >= recall.size:
        return None
    return float(precision[idx])


def recall_at_precision(scores, labels, target):
    order = np.argsort(-scores, kind="mergesort")
    y = (labels[order] == 1).astype(np.int64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    valid = np.flatnonzero(precision >= target)
    if valid.size == 0:
        return None
    return float(recall[valid[-1]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis",
                        default="outputs/evaluation/local_manual_v2/validation_threshold_analysis")
    parser.add_argument("--fine-tuned", default="outputs/checkpoints/local_manual_v2/seed_42/best.pt")
    parser.add_argument("--oscd",
                        default="outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt")
    parser.add_argument("--normalisation", default="outputs/dataset/v2/normalisation.json")
    parser.add_argument("--labels-dir", default="outputs/labels")
    parser.add_argument("--partition-dir", default="outputs/partitions_v2_observation_valid")
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    parser.add_argument("--registration",
                        default="outputs/audit/registration_v2_observation_valid/registration.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing to overwrite %s" % out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- provenance -------------------------------------------------------
    live = {"fine_tuned": sha256(args.fine_tuned), "oscd": sha256(args.oscd),
            "normalisation": sha256(args.normalisation)}
    for key, expected in HASHES.items():
        if live[key] != expected:
            raise SystemExit("%s hash mismatch: %s" % (key, live[key]))
    source = Path(args.analysis)
    ft_all = np.load(source / "_fine_tuned_scores.npy")
    oa_all = np.load(source / "_oscd_scores.npy")
    ft_labels = np.load(source / "_fine_tuned_labels.npy")
    oscd_labels = np.load(source / "_oscd_labels.npy")
    if not np.array_equal(ft_labels, oscd_labels):
        raise SystemExit("label vectors differ between checkpoints; ordering not shared")
    if ft_all.size != oa_all.size or ft_all.size != ft_labels.size:
        raise SystemExit("score vector sizes disagree")

    ft = ft_all[SPLIT_A:]
    oscd = oa_all[SPLIT_A:]
    labels = ft_labels[SPLIT_A:]
    counts = {"reviewed": int(labels.size), "positive": int((labels == 1).sum()),
              "negative": int((labels == 0).sum())}
    if counts != EXPECTED:
        raise SystemExit("Pair B counts differ from expected: %s vs %s" % (counts, EXPECTED))
    print("provenance verified; Pair B counts", counts)

    # ---- 3. reproduce the baseline comparison -----------------------------
    pair_b = {"fine_tuned": metrics(ft, labels), "oscd": metrics(oscd, labels)}
    pair_a = {"oscd": float(average_precision(ft_labels[:SPLIT_A], oa_all[:SPLIT_A])),
              "fine_tuned": float(average_precision(ft_labels[:SPLIT_A], ft_all[:SPLIT_A]))}
    print("Pair B:", {k: {m: round(v, 4) for m, v in d.items()
                          if isinstance(v, float)} for k, d in pair_b.items()})
    print("Pair A sanity:", {k: round(v, 4) for k, v in pair_a.items()})

    # ---- 4A. score distributions by class ---------------------------------
    distribution_rows = []
    for name, scores in (("oscd", oscd), ("fine_tuned", ft)):
        for truth, mask in (("change", labels == 1), ("no_change", labels == 0)):
            row = describe(scores[mask])
            row.update({"checkpoint": name, "class": truth})
            distribution_rows.append(row)
    with open(out / "score_distribution.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distribution_rows[0]))
        writer.writeheader()
        writer.writerows(distribution_rows)

    separation = {}
    for name, scores in (("oscd", oscd), ("fine_tuned", ft)):
        change = scores[labels == 1]
        nothing = scores[labels == 0]
        separation[name] = {
            "mean_gap": float(change.mean() - nothing.mean()),
            "median_gap": float(np.median(change) - np.median(nothing)),
            "change_median": float(np.median(change)),
            "no_change_median": float(np.median(nothing)),
        }

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.linspace(0, 1, 51)
    for name, scores, axis in (("OSCD", oscd, axes[0]), ("fine-tuned", ft, axes[1])):
        axis.hist(scores[labels == 1], bins=bins, alpha=0.65, density=True,
                  label="change (n=%d)" % int((labels == 1).sum()))
        axis.hist(scores[labels == 0], bins=bins, alpha=0.65, density=True,
                  label="no-change (n=%d)" % int((labels == 0).sum()))
        axis.set_title("%s score distribution" % name, fontsize=10)
        axis.set_xlabel("score")
        axis.set_ylabel("density")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    for truth, style in ((labels == 1, "-"), (labels == 0, "--")):
        label = "change" if truth[0] else "no-change"
        axes[2].plot(np.sort(oscd[truth]), np.linspace(0, 1, int(truth.sum())),
                     ls=style, lw=1.3, color="tab:grey",
                     label="OSCD %s" % label)
        axes[2].plot(np.sort(ft[truth]), np.linspace(0, 1, int(truth.sum())),
                     ls=style, lw=1.3, color="tab:orange",
                     label="fine-tuned %s" % label)
    axes[2].set_title("Empirical CDF, both checkpoints", fontsize=10)
    axes[2].set_xlabel("score")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    figure.suptitle("Pair B score distributions by ground-truth class", fontsize=11)
    figure.tight_layout()
    figure.savefig(out / "score_distributions.png", dpi=140)
    plt.close(figure)

    # ---- 4B. per-pixel deltas ---------------------------------------------
    delta = ft - oscd
    delta_rows = []
    for truth, mask in (("change", labels == 1), ("no_change", labels == 0),
                        ("all", np.ones(labels.shape, dtype=bool))):
        row = describe(delta[mask])
        row["class"] = truth
        delta_rows.append(row)
    with open(out / "score_delta_statistics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].hist(delta[labels == 1], bins=np.linspace(-1, 1, 81), alpha=0.7,
                 label="change pixels")
    axes[0].hist(delta[labels == 0], bins=np.linspace(-1, 1, 81), alpha=0.7,
                 label="no-change pixels")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("fine-tuned score − OSCD score")
    axes[0].set_ylabel("pixels")
    axes[0].set_title("Score difference by ground-truth class", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    step = max(1, labels.size // 8000)
    axes[1].scatter(oscd[::step], ft[::step], s=3, alpha=0.35,
                    c=np.where(labels[::step] == 1, "tab:red", "tab:blue"))
    axes[1].plot([0, 1], [0, 1], color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("OSCD score")
    axes[1].set_ylabel("fine-tuned score")
    axes[1].set_title("Per-pixel scores (red = change, blue = no-change)", fontsize=10)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "score_difference_distribution.png", dpi=140)
    figure.savefig(out / "score_scatter.png", dpi=140)
    plt.close(figure)

    # ---- 4C. transitions at 0.5 -------------------------------------------
    o_pred = oscd >= 0.5
    f_pred = ft >= 0.5
    truth = labels == 1
    o_ok = o_pred == truth
    f_ok = f_pred == truth
    transitions = {
        "both_correct": int((o_ok & f_ok).sum()),
        "regression_oscd_correct_ft_wrong": int((o_ok & ~f_ok).sum()),
        "improvement_oscd_wrong_ft_correct": int((~o_ok & f_ok).sum()),
        "both_wrong": int((~o_ok & ~f_ok).sum()),
        "oscd_tp_to_ft_fn": int((o_pred & truth & ~f_pred).sum()),
        "oscd_tn_to_ft_fp": int((~o_pred & ~truth & f_pred).sum()),
        "oscd_fn_to_ft_tp": int((~o_pred & truth & f_pred).sum()),
        "oscd_fp_to_ft_tn": int((o_pred & ~truth & ~f_pred).sum()),
    }
    rows = []
    for key, value in transitions.items():
        rows.append({"transition": key, "pixels": value,
                     "percent_of_reviewed": 100.0 * value / labels.size})
    with open(out / "prediction_transitions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["transition", "pixels",
                                                    "percent_of_reviewed"])
        writer.writeheader()
        writer.writerows(rows)
    print("transitions:", json.dumps(transitions, indent=1))

    # ---- 6. precision-recall comparison -----------------------------------
    pr_rows = []
    for level in (0.25, 0.50, 0.75, 0.90):
        pr_rows.append({"kind": "precision_at_recall", "level": level,
                        "oscd": precision_at_recall(oscd, labels, level),
                        "fine_tuned": precision_at_recall(ft, labels, level)})
    for level in (0.50, 0.60, 0.70, 0.80, 0.90):
        pr_rows.append({"kind": "recall_at_precision", "level": level,
                        "oscd": recall_at_precision(oscd, labels, level),
                        "fine_tuned": recall_at_precision(ft, labels, level)})
    with open(out / "precision_recall_comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "level", "oscd", "fine_tuned"])
        writer.writeheader()
        writer.writerows(pr_rows)

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for name, scores, style in (("OSCD", oscd, "-"), ("fine-tuned", ft, "-")):
        order = np.argsort(-scores, kind="mergesort")
        y = (labels[order] == 1).astype(np.int64)
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        recall = tp / max(int(y.sum()), 1)
        precision = tp / np.maximum(tp + fp, 1)
        axis.plot(recall, precision, style, lw=1.6,
                  label="%s (AP %.4f)" % (name, average_precision(labels, scores)))
    axis.axhline(float((labels == 1).mean()), color="grey", ls=":", lw=1,
                 label="prevalence %.3f" % float((labels == 1).mean()))
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9)
    axis.set_title("Pair B precision-recall, 22,650 reviewed pixels", fontsize=10)
    figure.tight_layout()
    figure.savefig(out / "precision_recall_comparison.png", dpi=150)
    plt.close(figure)

    json.dump({"hashes": live, "counts": counts, "pair_b": pair_b, "pair_a_ap": pair_a,
               "separation": separation, "transitions": transitions,
               "software": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "numpy": np.__version__},
               "evaluation_inputs": {"windows": 69, "patch": 256, "stride": 128,
                                     "aggregation": "LatticeAccumulator equal-weight mean, "
                                                    "strip-core clipped",
                                     "target_values": "0 and 1 only; 255 excluded"}},
              open(out / "comparison_metrics.json", "w"), indent=1)

    # ---- 5. spatial maps --------------------------------------------------
    datasets, _, _ = build_datasets(args)
    entry = datasets[PAIR]
    target = entry["target"]
    reviewed = target != IGNORE_VALUE
    maps = {}
    for name, path in (("oscd", args.oscd), ("fine_tuned", args.fine_tuned)):
        model, _ = load_model(path, torch.device(args.device))
        acc, _, _ = predict_pair(model, PAIR, entry, torch.device(args.device))
        maps[name] = np.where(acc.count > 0, acc.total / np.maximum(acc.count, 1), np.nan)
        del model
        torch.cuda.empty_cache()
    with rasterio.open(entry["labels_path"]) as handle:
        base, crs = handle.transform, handle.crs
    transform = Affine(base.a, base.b, base.c + entry["col_offset"] * base.a,
                       base.d, base.e, base.f)

    def write(path, array, dtype, nodata):
        with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                           width=array.shape[1], count=1, dtype=dtype, crs=crs,
                           transform=transform, nodata=nodata,
                           compress="deflate") as dst:
            dst.write(array.astype(dtype), 1)

    write(out / "pair_B_probability_oscd.tif", np.nan_to_num(maps["oscd"], nan=-9999),
          "float32", -9999)
    write(out / "pair_B_probability_fine_tuned.tif",
          np.nan_to_num(maps["fine_tuned"], nan=-9999), "float32", -9999)
    write(out / "pair_B_score_difference.tif",
          np.nan_to_num(maps["fine_tuned"] - maps["oscd"], nan=-9999), "float32", -9999)
    write(out / "pair_B_ground_truth_reviewed.tif",
          np.where(reviewed, target, 255), "uint8", 255)

    transition_map = np.zeros(target.shape, dtype=np.uint8)   # 0 = unreviewed
    o_pred_map = (maps["oscd"] >= 0.5)
    f_pred_map = (maps["fine_tuned"] >= 0.5)
    truth_map = target == 1
    o_ok_map = o_pred_map == truth_map
    f_ok_map = f_pred_map == truth_map
    transition_map[reviewed & o_ok_map & f_ok_map] = 1
    transition_map[reviewed & o_ok_map & ~f_ok_map] = 2
    transition_map[reviewed & ~o_ok_map & f_ok_map] = 3
    transition_map[reviewed & ~o_ok_map & ~f_ok_map] = 4
    write(out / "pair_B_prediction_transitions.tif", transition_map, "uint8", 0)

    rgb = rgb_background(args, PAIR, entry, target.shape)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, key, title in ((axes[0, 0], "oscd", "OSCD probability"),
                             (axes[0, 1], "fine_tuned", "Fine-tuned probability")):
        im = axis.imshow(np.where(reviewed, maps[key], np.nan), cmap="magma", vmin=0, vmax=1)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        figure.colorbar(im, ax=axis, fraction=0.035)
    diff = np.where(reviewed, maps["fine_tuned"] - maps["oscd"], np.nan)
    im = axes[1, 0].imshow(diff, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 0].set_title("Fine-tuned minus OSCD", fontsize=9)
    axes[1, 0].axis("off")
    figure.colorbar(im, ax=axes[1, 0], fraction=0.035)

    axes[1, 1].imshow(rgb)
    colour_layer = np.zeros((*target.shape, 3))
    legend = [(1, "both correct", COLOURS["both_correct"]),
              (2, "regression", COLOURS["regression"]),
              (3, "improvement", COLOURS["improvement"]),
              (4, "both wrong", COLOURS["both_wrong"])]
    for value, _, colour in legend:
        colour_layer[transition_map == value] = colour
    axes[1, 1].imshow(np.dstack([colour_layer,
                                 np.isin(transition_map, [1, 2, 3, 4]).astype(float) * 0.75]))
    axes[1, 1].set_title("Prediction transitions at 0.50", fontsize=9)
    axes[1, 1].axis("off")
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=10,
                          markerfacecolor=colour, markeredgecolor="k", label=label)
               for _, label, colour in legend]
    handles.append(plt.Line2D([], [], marker="s", linestyle="", markersize=10,
                              markerfacecolor="none", markeredgecolor="grey",
                              label="unreviewed (transparent)"))
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    figure.suptitle("Pair B checkpoint comparison, 22,650 reviewed pixels", fontsize=11)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(out / "pair_B_checkpoint_comparison.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.imshow(rgb)
    colour_layer = np.zeros((*target.shape, 3))
    for value, _, colour in legend:
        colour_layer[transition_map == value] = colour
    axis.imshow(np.dstack([colour_layer,
                           np.isin(transition_map, [1, 2, 3, 4]).astype(float) * 0.8]))
    axis.axis("off")
    axis.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    axis.set_title("Pair B prediction transitions at 0.50 on Sentinel-2 true colour",
                   fontsize=10)
    figure.tight_layout()
    figure.savefig(out / "pair_B_prediction_transitions.png", dpi=150)
    plt.close(figure)

    print("wrote", out)


if __name__ == "__main__":
    main()
