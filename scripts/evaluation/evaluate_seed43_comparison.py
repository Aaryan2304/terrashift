"""Five-checkpoint comparison: two weights x two seeds, plus the largest Pair B region.

Reuses the saved score vectors for the three checkpoints already evaluated and runs the
two seed-43 checkpoints through the same canonical predictor. Everything downstream, validation mask, lattice, aggregation, metric functions, is shared.

    python scripts/evaluation/evaluate_seed43_comparison.py \
        --output outputs/evaluation/local_manual_v2_seed43_comparison
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.sessions import PAIRS, build_datasets, load_model, predict_pair
from terrashift.evaluation.lattice import LatticeAccumulator
from terrashift.metrics import average_precision, roc_auc

SPLIT_A = 2759
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
CONFIGS = {
    "oscd": "outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt",
    "local_0.834_s42": "outputs/checkpoints/local_manual_v2/seed_42/best.pt",
    "local_1.0_s42": "outputs/checkpoints/local_manual_v2_posweight_1p0/seed_42/best.pt",
    "local_0.834_s43": "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt",
    "local_1.0_s43": "outputs/checkpoints/local_manual_v2_seed43_posweight_1p0/seed_43/best.pt",
}
RUNS = {
    "local_0.834_s42": "outputs/checkpoints/local_manual_v2/seed_42",
    "local_1.0_s42": "outputs/checkpoints/local_manual_v2_posweight_1p0/seed_42",
    "local_0.834_s43": "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43",
    "local_1.0_s43": "outputs/checkpoints/local_manual_v2_seed43_posweight_1p0/seed_43",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cls_counts(scores, labels, threshold):
    pred = scores >= threshold
    truth = labels == 1
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "predicted_positive_fraction": (tp + fp) / labels.size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--previous-analysis",
                        default="outputs/evaluation/local_manual_v2/validation_threshold_analysis")
    parser.add_argument("--posweight-analysis",
                        default="outputs/evaluation/local_manual_v2_posweight_1p0")
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
    device = torch.device(args.device)

    for path in CONFIGS.values():
        if not Path(path).is_file():
            raise SystemExit("missing checkpoint: %s" % path)

    datasets, _, _ = build_datasets(args)
    labels_all = np.load(Path(args.previous_analysis) / "_fine_tuned_labels.npy")

    vectors = {
        "oscd": np.load(Path(args.previous_analysis) / "_oscd_scores.npy"),
        "local_0.834_s42": np.load(Path(args.previous_analysis) / "_fine_tuned_scores.npy"),
        "local_1.0_s42": np.load(Path(args.posweight_analysis) / "_local_1p0_scores.npy"),
    }
    for name in ("local_0.834_s43", "local_1.0_s43"):
        model, state = load_model(CONFIGS[name], device)
        collected = []
        for pair in PAIRS:
            _, scores, labels = predict_pair(model, pair, datasets[pair], device)
            expected = labels_all[:SPLIT_A] if pair == "A" else labels_all[SPLIT_A:]
            if not np.array_equal(labels, expected):
                raise SystemExit("%s pair %s label vector differs" % (name, pair))
            collected.append(scores)
        vectors[name] = np.concatenate(collected)
        np.save(out / ("_%s_scores.npy" % name), vectors[name])
        print("%s: predicted %d pixels, pos_weight %r" % (name, vectors[name].size,
                                                          state.get("pos_weight")))
        del model
        torch.cuda.empty_cache()

    scopes = {"combined": (slice(None), labels_all),
              "A": (slice(0, SPLIT_A), labels_all[:SPLIT_A]),
              "B": (slice(SPLIT_A, None), labels_all[SPLIT_A:])}

    # ---- five-checkpoint table -------------------------------------------
    rows = []
    for name, vector in vectors.items():
        for scope, (sl, lab) in scopes.items():
            metrics = cls_counts(vector[sl], lab, 0.5)
            metrics.update({"checkpoint": name, "scope": scope,
                            "ap": float(average_precision(lab, vector[sl])),
                            "roc_auc": float(roc_auc(lab, vector[sl])),
                            "positives": int((lab == 1).sum()),
                            "negatives": int((lab == 0).sum()),
                            "prevalence": float((lab == 1).mean())})
            rows.append(metrics)
    fields = ["checkpoint", "scope", "ap", "roc_auc", "threshold", "tp", "fp", "fn", "tn",
              "precision", "recall", "f1", "iou", "specificity",
              "predicted_positive_fraction", "positives", "negatives", "prevalence"]
    with open(out / "checkpoint_comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pair_rows = [r for r in rows if r["scope"] == "B"]
    with open(out / "pair_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)

    # ---- training history ------------------------------------------------
    history = []
    for name, run in RUNS.items():
        for row in csv.DictReader(open(Path(run) / "train_log.csv")):
            history.append({"checkpoint": name, "epoch": int(row["epoch"]),
                            "train_loss": float(row["train_loss"]),
                            "train_ap": float(row["train_ap"]),
                            "val_loss": float(row["val_loss"]),
                            "val_ap": float(row["val_ap"]),
                            "val_f1": float(row["val_f1"]), "val_iou": float(row["val_iou"]),
                            "val_ap_A": float(row["val_ap_A"]),
                            "val_ap_B": float(row["val_ap_B"])})
    with open(out / "training_history.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best_epoch = {name: max((h for h in history if h["checkpoint"] == name),
                            key=lambda h: h["val_ap"])["epoch"] for name in RUNS}

    # ---- threshold metrics ----------------------------------------------
    threshold_rows = []
    for name, vector in vectors.items():
        for scope in ("B",):
            sl, lab = scopes[scope]
            for threshold in THRESHOLDS:
                row = cls_counts(vector[sl], lab, threshold)
                row.update({"checkpoint": name, "scope": scope})
                threshold_rows.append(row)
    with open(out / "threshold_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)

    # ---- largest Pair B component ---------------------------------------
    entry = datasets["B"]
    target = entry["target"]
    acc = LatticeAccumulator(target.shape, entry["col_offset"])
    zeros = np.zeros((256, 256)); ones = np.ones((256, 256), dtype=bool)
    for batch in DataLoader(entry["dataset"], batch_size=4, shuffle=False, num_workers=0):
        for item in range(len(batch["row_off"])):
            acc.add(int(batch["row_off"][item]), int(batch["col_off"][item]), zeros, ones,
                    core_offset=int(batch["core_offset"][item]),
                    core_count=int(batch["core_count"][item]))
    covered = (acc.count > 0) & (target != IGNORE_VALUE)
    change = covered & (target == 1)
    comp, n_comp = ndi.label(change, structure=np.ones((3, 3)))
    sizes = ndi.sum(change, comp, index=range(1, n_comp + 1))
    biggest = int(np.argmax(sizes)) + 1
    rows_c, cols_c = np.nonzero(covered)
    comp_per_pixel = comp[rows_c, cols_c]
    # comp_per_pixel already spans only Pair B reviewed pixels
    in_largest = comp_per_pixel == biggest
    component_rows = []
    for name, vector in vectors.items():
        values = vector[SPLIT_A:][in_largest]
        row = {"checkpoint": name, "component_pixels": int(values.size),
               "mean_score": float(values.mean()), "median_score": float(np.median(values)),
               "std_score": float(values.std()), "min_score": float(values.min()),
               "max_score": float(values.max())}
        for threshold in THRESHOLDS:
            row["recall_%.2f" % threshold] = float((values >= threshold).mean())
        component_rows.append(row)
    with open(out / "component_threshold_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)
    print("largest component pixels:", int(in_largest.sum()))

    # ---- seed summary ----------------------------------------------------
    summary_rows = []
    for weight, s42, s43 in (("0.834", "local_0.834_s42", "local_0.834_s43"),
                             ("1.0", "local_1.0_s42", "local_1.0_s43")):
        for scope in ("combined", "A", "B"):
            def value(name, key):
                return next(r[key] for r in rows if r["checkpoint"] == name
                            and r["scope"] == scope)
            for key in ("ap", "roc_auc", "precision", "recall", "f1", "iou",
                        "predicted_positive_fraction"):
                vals = [float(value(s42, key)), float(value(s43, key))]
                summary_rows.append({"weight": weight, "scope": scope, "metric": key,
                                     "seed42": vals[0], "seed43": vals[1],
                                     "mean": float(np.mean(vals)),
                                     "std": float(np.std(vals, ddof=0))})
    with open(out / "seed_comparison_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # ---- plots -----------------------------------------------------------
    b_labels = labels_all[SPLIT_A:]
    figure, axis = plt.subplots(figsize=(7, 5.4))
    for name in CONFIGS:
        order = np.argsort(-vectors[name][SPLIT_A:], kind="mergesort")
        y = (b_labels[order] == 1).astype(np.int64)
        tp = np.cumsum(y); fp = np.cumsum(1 - y)
        axis.plot(tp / max(int(y.sum()), 1), tp / np.maximum(tp + fp, 1), lw=1.4,
                  label="%s (AP %.3f)" % (name, average_precision(b_labels,
                                                                  vectors[name][SPLIT_A:])))
    axis.set_xlabel("recall"); axis.set_ylabel("precision"); axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02); axis.grid(alpha=0.3); axis.legend(fontsize=7)
    axis.set_title("Pair B precision-recall, 22,650 reviewed pixels", fontsize=10)
    figure.tight_layout()
    figure.savefig(out / "pair_B_precision_recall.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for name in CONFIGS:
        values = vectors[name][SPLIT_A:][in_largest]
        axes[0].plot(THRESHOLDS, [(values >= t).mean() for t in THRESHOLDS], lw=1.3,
                     label=name)
        metrics = [cls_counts(vectors[name][SPLIT_A:], b_labels, t) for t in THRESHOLDS]
        axes[1].plot(THRESHOLDS, [m["precision"] for m in metrics], lw=1.3, label=name)
        axes[2].plot(THRESHOLDS, [m["recall"] for m in metrics], lw=1.3, label=name)
    for axis, title in zip(axes, ("Largest component (10,749 px) recall",
                                  "Global Pair B precision",
                                  "Global Pair B recall")):
        axis.set_xlabel("threshold"); axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3); axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(out / "pair_B_threshold_analysis.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for name in CONFIGS:
        values = vectors[name][SPLIT_A:][in_largest]
        metrics = [cls_counts(vectors[name][SPLIT_A:], b_labels, t) for t in THRESHOLDS]
        axis.plot([m["precision"] for m in metrics],
                  [(values >= t).mean() for t in THRESHOLDS], lw=1.4, label=name)
    axis.set_xlabel("global Pair B precision")
    axis.set_ylabel("largest-component recall")
    axis.set_title("Recovering the largest change region costs global precision", fontsize=10)
    axis.grid(alpha=0.3); axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(out / "component_recall_vs_global_precision.png", dpi=150)
    plt.close(figure)

    payload = {
        "hashes": {name: sha256(path) for name, path in CONFIGS.items()},
        "normalisation_sha256": sha256(args.normalisation),
        "best_epoch": best_epoch,
        "thresholds": THRESHOLDS,
        "largest_component_pixels": int(in_largest.sum()),
        "counts": {"reviewed": int(labels_all.size),
                   "A": int(labels_all[:SPLIT_A].size),
                   "B": int(labels_all[SPLIT_A:].size),
                   "B_positives": int((labels_all[SPLIT_A:] == 1).sum()),
                   "A_positives": int((labels_all[:SPLIT_A] == 1).sum())},
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "numpy": np.__version__},
        "test_split_accessed": False,
        "note": "two seeds per weight; descriptive only, no significance test",
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "component_pixels": int(in_largest.sum())},
                     indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
