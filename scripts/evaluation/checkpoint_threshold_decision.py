"""Checkpoint and operating-threshold decision analysis (validation only).

No training, no model inference, no CUDA: every number comes from the saved validation
score vectors and the reviewed validation labels. Ranking metrics and count metrics come
from terrashift.metrics, the same implementation every earlier report in this project used, so
AP values here are comparable with those reports rather than close to them.

The Pair B component-8 mask is recovered from an existing diagnosis raster, so component
recall can be swept over the full score range instead of only the five audited thresholds.

    python scripts/evaluation/checkpoint_threshold_decision.py \
        --output outputs/evaluation/checkpoint_threshold_decision
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]

from terrashift.metrics import average_precision, metrics_from_counts, roc_auc

SPLIT_A = 2759
GRID = [round(0.01 * i, 2) for i in range(1, 100)]
SHORTLIST_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
CHECKPOINTS = {
    "oscd": {
        "path": "outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt",
        "scores": "outputs/evaluation/local_manual_v2/validation_threshold_analysis/_oscd_scores.npy",
        "run_dir": None},
    "local_0.834_s43": {
        "path": "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt",
        "scores": "outputs/evaluation/local_manual_v2_seed43_comparison/_local_0.834_s43_scores.npy",
        "run_dir": "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43"},
    "local_1.0_s43": {
        "path": "outputs/checkpoints/local_manual_v2_seed43_posweight_1p0/seed_43/best.pt",
        "scores": "outputs/evaluation/local_manual_v2_seed43_comparison/_local_1.0_s43_scores.npy",
        "run_dir": "outputs/checkpoints/local_manual_v2_seed43_posweight_1p0/seed_43"},
}
FINE_TUNED = ("local_0.834_s43", "local_1.0_s43")
LABELS = "outputs/evaluation/local_manual_v2/validation_threshold_analysis/_fine_tuned_labels.npy"
RECORDED = "outputs/evaluation/local_manual_v2_seed43_comparison/checkpoint_comparison.csv"
DIAGNOSIS = ("outputs/evaluation/local_manual_v2_seed43_comparison/spatial_audit/"
             "rasters/pair_B_oscd_diagnosis_0.50.tif")


def sha256(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def slice_for(pair):
    return slice(0, SPLIT_A) if pair == "A" else slice(SPLIT_A, None)


def num(value):
    """CSV cell: empty for undefined, six decimals for floats."""
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if np.isnan(value) else round(value, 6)
    return value


def counts_at(labels, scores, threshold):
    pred = scores >= threshold
    truth = labels == 1
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    return metrics_from_counts(tp, fp, fn, tn)


def component8_mask():
    """Pair B component mask over reviewed pixels, recovered from an existing raster."""
    with rasterio.open(DIAGNOSIS) as handle:
        base = handle.read(1)
    reviewed = base != 0
    labels = np.zeros(base.shape, dtype=np.int8)
    labels[reviewed] = np.isin(base[reviewed], (2, 4)).astype(np.int8)
    comp, n_comp = ndi.label(labels == 1, structure=np.ones((3, 3)))
    sizes = {int(i): int((comp == i).sum()) for i in range(1, n_comp + 1)}
    biggest = max(sizes, key=lambda i: sizes[i])
    rows, cols = np.nonzero(reviewed)
    if n_comp != 9 or sizes[biggest] != 10749:
        raise SystemExit("component recovery differs from the audit: %d components, "
                         "largest %d" % (n_comp, sizes[biggest]))
    return comp[rows, cols] == biggest, sizes, n_comp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing to overwrite %s" % out)
    out.mkdir(parents=True, exist_ok=True)

    labels_all = np.load(LABELS)
    pairs = {"A": slice_for("A"), "B": slice_for("B")}
    scores = {name: np.load(cfg["scores"]).astype(np.float32)
              for name, cfg in CHECKPOINTS.items()}
    for name, vector in scores.items():
        if vector.size != labels_all.size:
            raise SystemExit("%s: vector %d != labels %d"
                             % (name, vector.size, labels_all.size))

    recorded = {}
    if Path(RECORDED).exists():
        for row in csv.DictReader(open(RECORDED)):
            recorded[(row["checkpoint"], row["scope"])] = float(row["ap"])

    # ---- checkpoint verification --------------------------------------------
    verification = []
    for name, cfg in CHECKPOINTS.items():
        best_epoch, logged, epochs = "n/a", "n/a", "n/a"
        if cfg["run_dir"]:
            history = list(csv.DictReader(open(Path(cfg["run_dir"]) / "train_log.csv")))
            top = max(history, key=lambda r: float(r["val_ap"]))
            best_epoch, logged, epochs = int(top["epoch"]), round(float(top["val_ap"]), 6), len(history)
        pooled = average_precision(labels_all, scores[name])
        a_ap = average_precision(labels_all[pairs["A"]], scores[name][pairs["A"]])
        b_ap = average_precision(labels_all[pairs["B"]], scores[name][pairs["B"]])
        macro = (a_ap + b_ap) / 2
        verification.append({
            "checkpoint": name, "path": cfg["path"], "sha256": sha256(cfg["path"]),
            "best_epoch": best_epoch, "epochs_trained": epochs,
            "val_ap_logged_at_best_epoch": logged,
            "ap_pooled_micro": round(pooled, 6),
            "ap_pair_A": round(a_ap, 6), "ap_pair_B": round(b_ap, 6),
            "ap_macro_mean_of_pairs": round(macro, 6),
            "micro_minus_macro": round(pooled - macro, 6),
            "auc_pooled_micro": round(roc_auc(labels_all, scores[name]), 6),
            "auc_pair_A": round(roc_auc(labels_all[pairs["A"]], scores[name][pairs["A"]]), 6),
            "auc_pair_B": round(roc_auc(labels_all[pairs["B"]], scores[name][pairs["B"]]), 6),
            "recorded_pooled_ap": recorded.get((name, "combined"), "n/a"),
            "pooled_ap_matches_recorded": (
                round(pooled, 4) == round(recorded[(name, "combined")], 4)
                if (name, "combined") in recorded else "n/a"),
            "pair_A_pixels": 2759, "pair_B_pixels": 22650,
        })
    with open(out / "checkpoint_verification.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(verification[0]))
        writer.writeheader()
        writer.writerows(verification)

    # ---- full threshold sweep ----------------------------------------------
    comp8, sizes, n_comp = component8_mask()
    labels_b = labels_all[pairs["B"]]
    sweep = []
    for name in FINE_TUNED:
        comp8_scores = scores[name][pairs["B"]][comp8]
        for scope in ("combined", "A", "B"):
            if scope == "combined":
                lab, sc = labels_all, scores[name]
            else:
                lab, sc = labels_all[pairs[scope]], scores[name][pairs[scope]]
            for threshold in GRID:
                row = counts_at(lab, sc, threshold)
                sweep.append({
                    "checkpoint": name, "scope": scope, "threshold": threshold,
                    "tp": row["tp"], "fp": row["fp"], "fn": row["fn"], "tn": row["tn"],
                    "precision": num(row["precision"]), "recall": num(row["recall"]),
                    "f1": num(row["f1"]), "iou": num(row["iou"]),
                    "predicted_positive_fraction": round((row["tp"] + row["fp"]) / lab.size, 6),
                    "component8_recall": round(float((comp8_scores >= threshold).mean()), 6)
                    if scope == "B" else ""})
    for name in FINE_TUNED:
        grouped = {}
        for row in sweep:
            if row["checkpoint"] == name and row["scope"] in ("A", "B"):
                grouped.setdefault(row["threshold"], []).append(row)
        for threshold, rows in sorted(grouped.items()):
            if len(rows) != 2:
                continue
            micro = counts_at(labels_all, scores[name], threshold)
            sweep.append({
                "checkpoint": name, "scope": "macro", "threshold": threshold,
                "tp": micro["tp"], "fp": micro["fp"], "fn": micro["fn"], "tn": micro["tn"],
                "precision": num(float(np.nanmean([r["precision"] for r in rows]))),
                "recall": num(float(np.nanmean([r["recall"] for r in rows]))),
                "f1": num(float(np.nanmean([r["f1"] for r in rows]))),
                "iou": num(float(np.nanmean([r["iou"] for r in rows]))),
                "predicted_positive_fraction": round(float(np.mean(
                    [r["predicted_positive_fraction"] for r in rows])), 6),
                "component8_recall": ""})
    sweep.sort(key=lambda r: (r["checkpoint"], r["scope"], r["threshold"]))
    with open(out / "threshold_metrics.csv", "w", newline="") as handle:
        fields = ["checkpoint", "scope", "threshold", "tp", "fp", "fn", "tn", "precision",
                  "recall", "f1", "iou", "predicted_positive_fraction", "component8_recall"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sweep)

    def rows_for(name, scope):
        return [r for r in sweep if r["checkpoint"] == name and r["scope"] == scope]

    def best_row(name, scope, key):
        rows = [r for r in rows_for(name, scope) if r[key] != ""]
        rows.sort(key=lambda r: (float(r[key]), -r["threshold"]), reverse=True)
        return rows[0]

    def pivot(name, scope="B"):
        """Lowest threshold at which precision is still >= 0.50 (FP no longer exceeds TP)."""
        candidates = [r["threshold"] for r in rows_for(name, scope)
                      if r["precision"] != "" and float(r["precision"]) >= 0.5]
        return min(candidates) if candidates else None

    # ---- operating points --------------------------------------------------
    operating = []
    for name in FINE_TUNED:
        picks = [(t, "conventional threshold grid") for t in SHORTLIST_GRID]
        f1_pooled = best_row(name, "combined", "f1")
        f1_pairb = best_row(name, "B", "f1")
        piv = pivot(name, "B")
        picks.append((f1_pooled["threshold"], "maximum pooled F1 (informational)"))
        picks.append((f1_pairb["threshold"], "maximum Pair B F1 (informational)"))
        if piv is not None:
            picks.append((piv, "highest recall with Pair B precision still >= 0.50 "
                               "(informational)"))
        seen = set()
        for threshold, rationale in picks:
            if threshold in seen:
                continue
            seen.add(threshold)
            combined = counts_at(labels_all, scores[name], threshold)
            a = counts_at(labels_all[pairs["A"]], scores[name][pairs["A"]], threshold)
            b = counts_at(labels_all[pairs["B"]], scores[name][pairs["B"]], threshold)
            operating.append({
                "checkpoint": name, "threshold": threshold, "rationale": rationale,
                "micro_tp": combined["tp"], "micro_fp": combined["fp"],
                "micro_fn": combined["fn"], "micro_tn": combined["tn"],
                "micro_precision": num(combined["precision"]),
                "micro_recall": num(combined["recall"]),
                "micro_f1": num(combined["f1"]), "micro_iou": num(combined["iou"]),
                "predicted_positive_fraction": round(
                    (combined["tp"] + combined["fp"]) / labels_all.size, 6),
                "macro_precision": num(float(np.nanmean([a["precision"], b["precision"]]))),
                "macro_recall": num(float(np.nanmean([a["recall"], b["recall"]]))),
                "pair_A_precision": num(a["precision"]), "pair_A_recall": num(a["recall"]),
                "pair_A_tp": a["tp"], "pair_A_fp": a["fp"], "pair_A_fn": a["fn"],
                "pair_B_precision": num(b["precision"]), "pair_B_recall": num(b["recall"]),
                "pair_B_tp": b["tp"], "pair_B_fp": b["fp"], "pair_B_fn": b["fn"],
                "component8_recall": round(float(
                    (scores[name][pairs["B"]][comp8] >= threshold).mean()), 6),
                "fp_per_1000_reviewed": round(combined["fp"] / labels_all.size * 1000, 1)})
    with open(out / "operating_points.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operating[0]))
        writer.writeheader()
        writer.writerows(operating)

    # ---- matched-recall comparison -----------------------------------------
    def threshold_for_recall(name, target):
        rows = sorted(rows_for(name, "B"), key=lambda r: float(r["recall"]))
        recalls = np.array([float(r["recall"]) for r in rows])
        if target > recalls.max() or target < recalls.min():
            return None
        index = min(max(int(np.searchsorted(recalls, target)), 1), len(rows) - 1)
        lo, hi = rows[index - 1], rows[index]
        span = float(hi["recall"]) - float(lo["recall"])
        weight = 0.0 if span == 0 else (target - float(lo["recall"])) / span
        return round(float(lo["threshold"]) + weight *
                     (float(hi["threshold"]) - float(lo["threshold"])), 4)

    matched = []
    for target in (0.50, 0.60, 0.65, 0.70, 0.75):
        entry = {"pair_B_recall": target}
        for name in FINE_TUNED:
            threshold = threshold_for_recall(name, target)
            if threshold is None:
                entry[name + "_threshold"] = "unreachable"
                entry[name + "_precision"] = "unreachable"
                entry[name + "_fp"] = "unreachable"
                continue
            b = counts_at(labels_b, scores[name][pairs["B"]], threshold)
            entry[name + "_threshold"] = threshold
            entry[name + "_precision"] = num(b["precision"])
            entry[name + "_fp"] = b["fp"]
        matched.append(entry)

    # ---- plots -------------------------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, scope in zip(axes, ("combined", "A", "B")):
        for name in CHECKPOINTS:
            if scope == "combined":
                lab, sc = labels_all, scores[name]
            else:
                lab, sc = labels_all[pairs[scope]], scores[name][pairs[scope]]
            order = np.argsort(-sc, kind="mergesort")
            y = (lab[order] == 1).astype(np.int64)
            tp = np.cumsum(y); fp = np.cumsum(1 - y)
            axis.plot(tp / max(int(y.sum()), 1), tp / np.maximum(tp + fp, 1), lw=1.5,
                      label="%s (AP %.3f)" % (name, average_precision(lab, sc)))
        axis.set_xlabel("recall"); axis.set_ylabel("precision")
        axis.set_xlim(0, 1); axis.set_ylim(0, 1.02); axis.grid(alpha=0.3)
        axis.legend(fontsize=7, loc="lower left")
        axis.set_title("%s precision-recall" % ("pooled (micro)" if scope == "combined"
                                                else "Pair " + scope), fontsize=10)
    figure.suptitle("Validation precision-recall, 25,409 reviewed pixels — test split "
                    "not used", fontsize=11)
    figure.tight_layout()
    figure.savefig(out / "precision_recall_curves.png", dpi=150)
    plt.close(figure)

    # ---- manifest ----------------------------------------------------------
    manifest = {
        "status": "complete",
        "scope": "validation only",
        "device": "cpu",
        "model_inference_rerun": False,
        "test_split_accessed": False,
        "threshold_locked": False,
        "sources": {
            "labels": {"path": LABELS, "sha256": sha256(LABELS)},
            "scores": {n: {"path": c["scores"], "sha256": sha256(c["scores"])}
                       for n, c in CHECKPOINTS.items()},
            "checkpoints": {n: {"path": c["path"], "sha256": sha256(c["path"])}
                            for n, c in CHECKPOINTS.items()},
            "component_mask_from": {"path": DIAGNOSIS, "sha256": sha256(DIAGNOSIS)},
            "recorded_ap_from": {"path": RECORDED, "sha256": sha256(RECORDED)}},
        "counts": {"reviewed_total": int(labels_all.size),
                   "reviewed_A": SPLIT_A, "reviewed_B": int(labels_all.size - SPLIT_A),
                   "positives_A": int((labels_all[pairs["A"]] == 1).sum()),
                   "positives_B": int((labels_all[pairs["B"]] == 1).sum()),
                   "components": n_comp, "component_sizes": sizes},
        "threshold_grid": {"min": GRID[0], "max": GRID[-1], "step": 0.01, "points": len(GRID)},
        "score_range": {n: [round(float(scores[n].min()), 6), round(float(scores[n].max()), 6)]
                        for n in CHECKPOINTS},
        "matched_recall": matched,
        "outputs": ["checkpoint_verification.csv", "threshold_metrics.csv",
                    "operating_points.csv", "precision_recall_curves.png",
                    "checkpoint_threshold_decision.md", "decision_manifest.json"],
        "software": {"numpy": np.__version__, "rasterio": rasterio.__version__},
    }
    (out / "decision_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    print("verification")
    for row in verification:
        print("  %-16s ep %-4s pooled %.4f A %.4f B %.4f macro %.4f (micro-macro %+.4f) "
              "AUC %.4f/%.4f/%.4f match %s"
              % (row["checkpoint"], row["best_epoch"], row["ap_pooled_micro"],
                 row["ap_pair_A"], row["ap_pair_B"], row["ap_macro_mean_of_pairs"],
                 row["micro_minus_macro"], row["auc_pooled_micro"], row["auc_pair_A"],
                 row["auc_pair_B"], row["pooled_ap_matches_recorded"]))
    print("operating points")
    for row in operating:
        print("  %-16s @%.2f %-52s P %s R %s F1 %s | AR %s | BP %s BR %s | c8 %.3f | +%.3f"
              % (row["checkpoint"], row["threshold"], row["rationale"][:52],
                 row["micro_precision"], row["micro_recall"], row["micro_f1"],
                 row["pair_A_recall"], row["pair_B_precision"], row["pair_B_recall"],
                 row["component8_recall"], row["predicted_positive_fraction"]))
    print("matched recall")
    for row in matched:
        print(" ", row)
    print("wrote", out)


if __name__ == "__main__":
    main()
