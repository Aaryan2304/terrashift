"""Audit: reconcile Pair B component statistics with the global confusion counts.

Read-only. Rebuilds the reviewed-pixel mask independently of any model (the coverage
count map depends only on the tiling and the target, not on the predictions), maps flat
score-vector indices back to raster positions, and checks every invariant the previous
component analysis should have satisfied.

    python scripts/evaluation/audit_pair_b_components.py \
        --output outputs/evaluation/local_manual_v2/pair_b_component_audit
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
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.sessions import BATCH, PAIRS, build_datasets, load_model, predict_pair
from terrashift.metrics import average_precision, roc_auc

PAIR = "B"
SPLIT_A = 2759
THRESHOLD = 0.5
OPERATOR = ">="
HASHES = {
    "fine_tuned": "4c083af28d6e6ee43c7ca117355e794ad570412310c7856026b717f283de0546",
    "oscd": "4d9c381dba370becc3ac3ff93431db65929cf21378287191a9f131fae9080d1a",
    "normalisation": "8aeea033eb2bec5012ebbceb7eca825814b6e43bb699f9e2eaee0c5bba955d2a",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(scores, labels, threshold=THRESHOLD):
    pred = scores >= threshold
    truth = labels == 1
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "operator": OPERATOR, "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "predicted_positive": tp + fp,
            "predicted_positive_fraction": (tp + fp) / labels.size,
            "n_pixels": int(labels.size)}


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max())}


def coverage_mask(model_free_dataset, entry):
    """Covered-pixel mask built with a constant scorer: counts do not depend on values."""
    from terrashift.evaluation.lattice import LatticeAccumulator

    target = entry["target"]
    acc = LatticeAccumulator(target.shape, entry["col_offset"])
    loader = DataLoader(model_free_dataset, batch_size=BATCH, shuffle=False, num_workers=0)
    zeros = np.zeros((256, 256), dtype=np.float64)
    ones = np.ones((256, 256), dtype=bool)
    for batch in loader:
        for item in range(len(batch["row_off"])):
            acc.add(int(batch["row_off"][item]), int(batch["col_off"][item]), zeros, ones,
                    core_offset=int(batch["core_offset"][item]),
                    core_count=int(batch["core_count"][item]))
    return (acc.count > 0) & (target != IGNORE_VALUE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis",
                        default="outputs/evaluation/local_manual_v2/validation_threshold_analysis")
    parser.add_argument("--previous",
                        default="outputs/evaluation/local_manual_v2/pair_b_checkpoint_comparison")
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

    live = {"fine_tuned": sha256(args.fine_tuned), "oscd": sha256(args.oscd),
            "normalisation": sha256(args.normalisation)}
    for key, expected in HASHES.items():
        if live[key] != expected:
            raise SystemExit("%s hash mismatch" % key)

    source = Path(args.analysis)
    ft_all = np.load(source / "_fine_tuned_scores.npy")
    oa_all = np.load(source / "_oscd_scores.npy")
    labels_all = np.load(source / "_fine_tuned_labels.npy")
    if not np.array_equal(labels_all, np.load(source / "_oscd_labels.npy")):
        raise SystemExit("label vectors differ between checkpoints")
    ft = ft_all[SPLIT_A:]
    oscd = oa_all[SPLIT_A:]
    labels = labels_all[SPLIT_A:]

    # ---- 3. independent global metrics ------------------------------------
    global_metrics = {"fine_tuned": classify(ft, labels), "oscd": classify(oscd, labels),
                      "counts": {"reviewed": int(labels.size),
                                 "change": int((labels == 1).sum()),
                                 "no_change": int((labels == 0).sum())}}
    for name in ("fine_tuned", "oscd"):
        global_metrics[name]["ap"] = float(average_precision(labels, ft if name == "fine_tuned" else oscd))
        global_metrics[name]["roc_auc"] = float(roc_auc(labels, ft if name == "fine_tuned" else oscd))
        global_metrics[name]["change_score_stats"] = stats((ft if name == "fine_tuned" else oscd)[labels == 1])
        global_metrics[name]["no_change_score_stats"] = stats((ft if name == "fine_tuned" else oscd)[labels == 0])
    print("global:", {k: {m: round(v, 4) for m, v in d.items() if isinstance(v, float)}
                      for k, d in global_metrics.items() if k != "counts"})

    # ---- 6. alignment: vector index -> raster position --------------------
    datasets, _, _ = build_datasets(args)
    entry = datasets[PAIR]
    target = entry["target"]
    covered = coverage_mask(entry["dataset"], entry)
    if int(covered.sum()) != labels.size:
        raise SystemExit("coverage mask has %d pixels but the vector has %d"
                         % (int(covered.sum()), labels.size))
    rows_c, cols_c = np.nonzero(covered)          # row-major, matches boolean indexing
    raster_change = covered & (target == 1)
    if int(raster_change.sum()) != int((labels == 1).sum()):
        raise SystemExit("raster change count %d != vector change count %d"
                         % (int(raster_change.sum()), int((labels == 1).sum())))
    labels_grid = np.full(target.shape, -1, dtype=np.int32)
    component_labels, n_components = ndi.label(covered & (target == 1),
                                               structure=np.ones((3, 3)))
    labels_grid[rows_c, cols_c] = component_labels[rows_c, cols_c]
    vector_component = labels_grid[rows_c, cols_c]
    if np.any(vector_component[labels == 1] <= 0):
        raise SystemExit("a reviewed change pixel is not assigned to a component")

    # ---- 4. per-component metrics, both sources ---------------------------
    device = torch.device(args.device)
    maps = {}
    for name, path in (("oscd", args.oscd), ("fine_tuned", args.fine_tuned)):
        model, _ = load_model(path, device)
        acc, scores_check, labels_check = predict_pair(model, PAIR, entry, device)
        maps[name] = np.where(acc.count > 0, acc.total / np.maximum(acc.count, 1), np.nan)
        # the same pass must reproduce the saved vector
        if not np.allclose(scores_check, ft if name == "fine_tuned" else oscd,
                           atol=0, rtol=0):
            delta = np.abs(scores_check - (ft if name == "fine_tuned" else oscd))
            print("NOTE %s: this pass differs from the saved vector, max |d| = %.3e, "
                  "%d pixels differ" % (name, delta.max(), int((delta > 0).sum())))
        del model
        torch.cuda.empty_cache()

    component_rows = []
    source_rows = []
    for index in range(1, n_components + 1):
        in_component = vector_component == index            # vector order, component ids
        region = component_labels == index                  # raster order
        entry_row = {"component": index,
                     "change_pixels": int(in_component.sum()),
                     "raster_change_pixels": int(region.sum())}
        for name, vector, key in (("oscd", oscd, "oscd"), ("fine_tuned", ft, "fine_tuned")):
            from_vector = vector[in_component]
            from_raster = maps[key][region]
            entry_row["%s_tp" % name] = int((from_vector >= THRESHOLD).sum())
            entry_row["%s_fn" % name] = int((from_vector < THRESHOLD).sum())
            entry_row["%s_recall" % name] = (entry_row["%s_tp" % name]
                                             / max(int(from_vector.size), 1))
            entry_row["%s_mean" % name] = float(np.mean(from_vector))
            entry_row["%s_median" % name] = float(np.median(from_vector))
            entry_row["%s_std" % name] = float(np.std(from_vector))
            entry_row["%s_min" % name] = float(from_vector.min())
            entry_row["%s_max" % name] = float(from_vector.max())
            source_rows.append({"component": index, "checkpoint": name,
                                "tp_vector": entry_row["%s_tp" % name],
                                "tp_raster": int(np.nansum(from_raster >= THRESHOLD)),
                                "raster_nan_pixels": int(np.isnan(from_raster).sum()),
                                "max_abs_diff_vector_vs_raster": float(
                                    np.nanmax(np.abs(from_raster - from_vector)))
                                if np.isfinite(from_raster).any() else None})
        component_rows.append(entry_row)

    with open(out / "component_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)

    # ---- invariants -------------------------------------------------------
    ft_tp_sum = sum(r["fine_tuned_tp"] for r in component_rows)
    ft_fn_sum = sum(r["fine_tuned_fn"] for r in component_rows)
    change_sum = sum(r["change_pixels"] for r in component_rows)
    weighted_recall = ft_tp_sum / max(change_sum, 1)
    checks = {
        "components": int(n_components),
        "change_pixels_sum": change_sum,
        "change_pixels_expected": global_metrics["counts"]["change"],
        "change_pixels_match": change_sum == global_metrics["counts"]["change"],
        "fine_tuned_tp_sum": ft_tp_sum,
        "fine_tuned_tp_global": global_metrics["fine_tuned"]["tp"],
        "tp_sum_matches_global": ft_tp_sum == global_metrics["fine_tuned"]["tp"],
        "fine_tuned_fn_sum": ft_fn_sum,
        "fine_tuned_fn_global": global_metrics["fine_tuned"]["fn"],
        "fn_sum_matches_global": ft_fn_sum == global_metrics["fine_tuned"]["fn"],
        "weighted_recall": weighted_recall,
        "global_recall": global_metrics["fine_tuned"]["recall"],
        "weighted_recall_matches": abs(weighted_recall
                                       - global_metrics["fine_tuned"]["recall"]) < 1e-12,
        "oscd_tp_sum": sum(r["oscd_tp"] for r in component_rows),
        "oscd_tp_global": global_metrics["oscd"]["tp"],
        "every_change_pixel_assigned_once": bool(
            np.all(vector_component[labels == 1] > 0)
            and np.unique(np.concatenate([np.flatnonzero(labels == i)
                                          for i in range(1, n_components + 1)])).size == labels.size
            - int((labels == 0).sum())),
    }
    print("invariants:", json.dumps(checks, indent=1))
    json.dump({"hashes": live, "threshold": THRESHOLD, "operator": OPERATOR,
               "global": global_metrics, "invariants": checks,
               "component_size_pixels": [int(r["change_pixels"]) for r in component_rows],
               "vector_vs_raster": source_rows},
              open(out / "alignment_audit.json", "w"), indent=1)
    json.dump(global_metrics, open(out / "global_metrics_recomputed.json", "w"), indent=1)

    # ---- grouped statistics, exact sizes, no invented boundaries ----------
    sizes = np.array([r["change_pixels"] for r in component_rows])
    order = np.argsort(sizes)
    group_rows = []
    for position, index in enumerate(order):
        row = component_rows[index]
        group_rows.append({"rank_by_size": position + 1, **row})
    with open(out / "component_group_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    print("component sizes (ascending):", sorted(sizes.tolist()))
    print("wrote", out)


if __name__ == "__main__":
    main()
