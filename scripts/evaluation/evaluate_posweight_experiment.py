"""Three-checkpoint comparison for the pos_weight experiment.

OSCD initial, local fine-tune at pos_weight 0.834, local fine-tune at pos_weight 1.0,
all evaluated on the same 159 canonical validation windows and the same unique reviewed
pixels. Reuses the saved score vectors for the first two checkpoints and the canonical
predictor for the third.

    python scripts/evaluation/evaluate_posweight_experiment.py \
        --output outputs/evaluation/local_manual_v2_posweight_1p0
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.sessions import PAIRS, build_datasets, load_model, predict_pair
from terrashift.metrics import average_precision, roc_auc

SPLIT_A = 2759
THRESHOLD = 0.5
EXPECTED = {"A": 2759, "B": 22650, "total": 25409}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(scores, labels):
    pred = scores >= THRESHOLD
    truth = labels == 1
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"ap": float(average_precision(labels, scores)),
            "roc_auc": float(roc_auc(labels, scores)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "predicted_positive_fraction": (tp + fp) / labels.size,
            "n_pixels": int(labels.size)}


def pr_curve(scores, labels):
    order = np.argsort(-scores, kind="mergesort")
    y = (labels[order] == 1).astype(np.int64)
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    return tp / max(int(y.sum()), 1), tp / np.maximum(tp + fp, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis",
                        default="outputs/evaluation/local_manual_v2/validation_threshold_analysis")
    parser.add_argument("--new-run",
                        default="outputs/checkpoints/local_manual_v2_posweight_1p0/seed_42")
    parser.add_argument("--previous-run", default="outputs/checkpoints/local_manual_v2/seed_42")
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

    # ---- reuse the saved vectors for the first two checkpoints ------------
    source = Path(args.analysis)
    saved = {"oscd": np.load(source / "_oscd_scores.npy"),
             "local_0.834": np.load(source / "_fine_tuned_scores.npy")}
    labels_all = np.load(source / "_fine_tuned_labels.npy")

    # ---- run the new checkpoint through the same path ---------------------
    datasets, _, _ = build_datasets(args)
    device = torch.device(args.device)
    new_scores = []
    model, state = load_model(Path(args.new_run) / "best.pt", device)
    if float(state.get("pos_weight", -1)) != 1.0:
        raise SystemExit("new checkpoint pos_weight is %r, expected 1.0"
                         % state.get("pos_weight"))
    for pair in PAIRS:
        _, scores, labels = predict_pair(model, pair, datasets[pair], device)
        if not np.array_equal(labels, labels_all[:SPLIT_A] if pair == "A"
                              else labels_all[SPLIT_A:]):
            raise SystemExit("pair %s label vector differs from the saved one" % pair)
        new_scores.append(scores)
    del model
    torch.cuda.empty_cache()
    saved["local_1.0"] = np.concatenate(new_scores)
    np.save(out / "_local_1p0_scores.npy", saved["local_1.0"])

    counts = {"A": int(labels_all[:SPLIT_A].size),
              "B": int(labels_all[SPLIT_A:].size),
              "total": int(labels_all.size)}
    if counts != EXPECTED:
        raise SystemExit("counts differ: %s" % counts)

    # ---- per-checkpoint, per-scope metrics -------------------------------
    scopes = {"combined": (slice(None), labels_all),
              "A": (slice(0, SPLIT_A), labels_all[:SPLIT_A]),
              "B": (slice(SPLIT_A, None), labels_all[SPLIT_A:])}
    comparison = []
    for name, vector in saved.items():
        for scope, (sl, lab) in scopes.items():
            row = {"checkpoint": name, "scope": scope, **metrics(vector[sl], lab)}
            comparison.append(row)
    with open(out / "checkpoint_comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    pair_rows = [r for r in comparison if r["scope"] == "B"]
    with open(out / "pair_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    # ---- training history for both local runs ---------------------------
    history = []
    for label, run in (("local_0.834", args.previous_run), ("local_1.0", args.new_run)):
        for row in csv.DictReader(open(Path(run) / "train_log.csv")):
            history.append({"run": label, "epoch": int(row["epoch"]),
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
    best_epoch = {}
    for run in ("local_0.834", "local_1.0"):
        rows = [h for h in history if h["run"] == run]
        best_epoch[run] = max(rows, key=lambda h: h["val_ap"])["epoch"]

    # ---- Pair B: PR curves and change-score distributions ---------------
    b_labels = labels_all[SPLIT_A:]
    figure, axis = plt.subplots(figsize=(6.6, 5.2))
    for name, style in (("oscd", ":"), ("local_0.834", "-"), ("local_1.0", "-")):
        scores = saved[name][SPLIT_A:]
        recall, precision = pr_curve(scores, b_labels)
        axis.plot(recall, precision, style, lw=1.6,
                  label="%s (AP %.4f)" % (name, average_precision(b_labels, scores)))
    axis.axhline(float((b_labels == 1).mean()), color="grey", lw=0.8, ls="--")
    axis.set_xlabel("recall"); axis.set_ylabel("precision"); axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02); axis.grid(alpha=0.3); axis.legend(fontsize=8)
    axis.set_title("Pair B precision-recall, 22,650 reviewed pixels", fontsize=10)
    figure.tight_layout()
    figure.savefig(out / "pair_B_precision_recall.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for axis, name in zip(axes, ("oscd", "local_0.834", "local_1.0")):
        scores = saved[name][SPLIT_A:]
        axis.hist(scores[b_labels == 1], bins=np.linspace(0, 1, 51), alpha=0.7,
                  density=True, label="change")
        axis.hist(scores[b_labels == 0], bins=np.linspace(0, 1, 51), alpha=0.7,
                  density=True, label="no-change")
        axis.set_title("%s\nchange median %.3f / no-change median %.3f"
                       % (name, np.median(scores[b_labels == 1]),
                          np.median(scores[b_labels == 0])), fontsize=9)
        axis.set_xlabel("score"); axis.legend(fontsize=8); axis.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    figure.suptitle("Pair B change-pixel score distribution by checkpoint", fontsize=11)
    figure.tight_layout()
    figure.savefig(out / "pair_B_score_distributions.png", dpi=140)
    plt.close(figure)

    # ---- largest Pair B component --------------------------------------
    import torch as _torch
    from torch.utils.data import DataLoader
    from terrashift.evaluation.lattice import LatticeAccumulator
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
    largest = comp == biggest
    rows_c, cols_c = np.nonzero(covered)
    # map the raster component onto vector positions
    vector_component = np.full(labels_all.shape, -1, dtype=np.int32)
    comp_map = np.full(target.shape, -1, dtype=np.int32)
    comp_map[rows_c, cols_c] = comp[rows_c, cols_c]
    vector_component[:SPLIT_A] = -2
    vector_component[SPLIT_A:] = comp_map[rows_c, cols_c]
    in_largest = vector_component[SPLIT_A:] == biggest   # Pair B slice
    largest_stats = {}
    for name in saved:
        scores = saved[name][SPLIT_A:]
        vals = scores[in_largest]
        largest_stats[name] = {"pixels": int(vals.size),
                               "tp": int((vals >= THRESHOLD).sum()),
                               "recall": float((vals >= THRESHOLD).mean()),
                               "mean": float(vals.mean()),
                               "median": float(np.median(vals))}
    print("largest component:", json.dumps(largest_stats, indent=1))

    payload = {
        "hashes": {"oscd": sha256("outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt"),
                   "local_0.834": sha256("outputs/checkpoints/local_manual_v2/seed_42/best.pt"),
                   "local_1.0": sha256(Path(args.new_run) / "best.pt"),
                   "normalisation": sha256(args.normalisation)},
        "threshold": THRESHOLD, "counts": counts,
        "best_epoch": best_epoch,
        "largest_pair_b_component": largest_stats,
        "component_sizes_desc": sorted([int(s) for s in sizes], reverse=True)[:3],
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "numpy": np.__version__},
        "test_split_accessed": False,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "software"}, indent=1))
    print("wrote", out)



if __name__ == "__main__":
    main()
