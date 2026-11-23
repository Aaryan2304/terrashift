"""Validation-only threshold and threshold-trade-off analysis for baseline v2.

Reuses the canonical validation machinery end to end: the same
``ValidationLatticeDataset``, the same ``LatticeAccumulator`` (equal-weight mean over
the windows the inference tiler would run, clipped to each strip core), and the same
``unique_validation_metrics``. Nothing here invents a second lattice or a second
aggregation rule.

One probability per unique reviewed geographic pixel. Target values 0 and 1 only;
255 is excluded everywhere. Test labels are never opened.

    python scripts/evaluation/analyse_validation_thresholds.py \
        --output outputs/evaluation/local_manual_v2/validation_threshold_analysis
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.local_patches import BANDS, pair_image_loader
from terrashift.data.manifest import load_manifest, pair_config
from terrashift.preprocessing.normalisation import require_statistics
from terrashift.data.safeloader import SafeProduct
from terrashift.data.validation_windows import (IGNORE_VALUE, ValidationLatticeDataset,
                                     read_validation_target)
from terrashift.models.siamese_unet import SiameseUNet
from terrashift.evaluation.lattice import LatticeAccumulator, evaluate_lattice
from terrashift.metrics import average_precision
from terrashift.evaluation.unique_regions import unique_validation_metrics
from terrashift.evaluation.sessions import BATCH, PAIRS, PATCH, STRIDE, build_datasets, load_model, predict_pair
from terrashift.geospatial.rendering import rgb_background

EXPECTED = {"A": 2759, "B": 22650, "total": 25409}
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
REPORTED = {"ap": 0.7017, "precision": 0.7135, "recall": 0.3517, "f1": 0.4712,
            "iou": 0.3082, "pred_fraction": 0.2633,
            "tp": 4773, "fp": 1916, "fn": 8800, "tn": 9920}
HASHES = {
    "fine_tuned": "4c083af28d6e6ee43c7ca117355e794ad570412310c7856026b717f283de0546",
    "oscd": "4d9c381dba370becc3ac3ff93431db65929cf21378287191a9f131fae9080d1a",
    "normalisation": "8aeea033eb2bec5012ebbceb7eca825814b6e43bb699f9e2eaee0c5bba955d2a",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classification(scores, labels, threshold):
    pred = scores >= threshold
    truth = labels == 1
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "specificity": specificity,
            "predicted_positive": tp + fp,
            "predicted_positive_fraction": (tp + fp) / total if total else 0.0,
            "n_pixels": total}


def best_f1_threshold(scores, labels, grid=None):
    """Threshold on a fixed grid that maximises F1 (validation-derived)."""
    grid = grid or THRESHOLDS
    rows = [classification(scores, labels, t) for t in grid]
    return max(rows, key=lambda row: (row["f1"], row["threshold"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fine-tuned", default="outputs/checkpoints/local_manual_v2/seed_42/best.pt")
    parser.add_argument("--oscd", default="outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt")
    parser.add_argument("--normalisation", default="outputs/dataset/v2/normalisation.json")
    parser.add_argument("--labels-dir", default="outputs/labels")
    parser.add_argument("--partition-dir", default="outputs/partitions_v2_observation_valid")
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    parser.add_argument("--registration",
                        default="outputs/audit/registration_v2_observation_valid/registration.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out_dir = Path(args.output)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit("refusing to overwrite %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- hashes ------------------------------------------------------------
    hashes = {"fine_tuned": sha256(args.fine_tuned), "oscd": sha256(args.oscd),
              "normalisation": sha256(args.normalisation)}
    for key, expected in HASHES.items():
        if hashes[key] != expected:
            raise SystemExit("%s hash mismatch: %s != %s" % (key, hashes[key], expected))
    print("hashes verified")

    device = torch.device(args.device)
    datasets, manifest, normalisation = build_datasets(args)

    results = {}
    for name, path in (("fine_tuned", args.fine_tuned), ("oscd", args.oscd)):
        model, state = load_model(path, device)
        results[name] = {"epoch": int(state.get("epoch", -1)), "pairs": {},
                         "scores": [], "labels": []}
        for pair in PAIRS:
            acc, scores, labels = predict_pair(model, pair, datasets[pair], device)
            results[name]["pairs"][pair] = {
                "scores": scores, "labels": labels, "accumulator": acc}
            results[name]["scores"].append(scores)
            results[name]["labels"].append(labels)
            print("%s pair %s: %d reviewed pixels (%d positive)"
                  % (name, pair, scores.size, int((labels == 1).sum())))
        results[name]["scores"] = np.concatenate(results[name]["scores"])
        results[name]["labels"] = np.concatenate(results[name]["labels"])
        del model
        torch.cuda.empty_cache()

    # ---- count gate --------------------------------------------------------
    counts = {}
    for pair in PAIRS:
        counts[pair] = int(results["fine_tuned"]["pairs"][pair]["scores"].size)
    counts["total"] = int(results["fine_tuned"]["scores"].size)
    if counts != EXPECTED:
        raise SystemExit("reviewed pixel counts differ from expected: %s vs %s"
                         % (counts, EXPECTED))
    print("counts match expected:", counts)

    np.save(out_dir / "_fine_tuned_scores.npy", results["fine_tuned"]["scores"])
    np.save(out_dir / "_fine_tuned_labels.npy", results["fine_tuned"]["labels"])
    np.save(out_dir / "_oscd_scores.npy", results["oscd"]["scores"])
    np.save(out_dir / "_oscd_labels.npy", results["oscd"]["labels"])

    # ---- reproduce the reported metrics ------------------------------------
    ft = results["fine_tuned"]
    at_050 = classification(ft["scores"], ft["labels"], 0.5)
    ap_existing = float(average_precision(ft["labels"], ft["scores"]))
    try:
        from sklearn.metrics import average_precision_score
        ap_sklearn = float(average_precision_score(ft["labels"], ft["scores"]))
    except ImportError:
        ap_sklearn = None
    reproduced = {
        "ap": ap_existing, "ap_sklearn": ap_sklearn,
        "precision": at_050["precision"], "recall": at_050["recall"],
        "f1": at_050["f1"], "iou": at_050["iou"],
        "pred_fraction": at_050["predicted_positive_fraction"],
        "tp": at_050["tp"], "fp": at_050["fp"], "fn": at_050["fn"], "tn": at_050["tn"],
    }
    mismatches = {}
    for key, expected in REPORTED.items():
        got = reproduced.get(key)
        if got is None:
            continue
        if abs(float(got) - float(expected)) > (1e-4 if isinstance(expected, float) else 0):
            mismatches[key] = {"reported": expected, "reproduced": got}
    print("reproduced @0.5:", {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in reproduced.items()})
    print("mismatches:", mismatches or "none")

    json.dump({"hashes": hashes, "counts": counts, "reproduced": reproduced,
               "reported": REPORTED, "mismatches": mismatches,
               "checkpoint_epochs": {k: v["epoch"] for k, v in results.items()},
               "software": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "numpy": np.__version__}},
              open(out_dir / "validation_metrics.json", "w"), indent=1)

    # ---- threshold tables --------------------------------------------------
    rows = []
    for scope in ("combined", "A", "B"):
        if scope == "combined":
            scores, labels = ft["scores"], ft["labels"]
        else:
            scores = ft["pairs"][scope]["scores"]
            labels = ft["pairs"][scope]["labels"]
        for threshold in THRESHOLDS:
            row = classification(scores, labels, threshold)
            row["scope"] = scope
            rows.append(row)
    fields = ["scope", "threshold", "tp", "fp", "fn", "tn", "precision", "recall",
              "f1", "iou", "specificity", "predicted_positive",
              "predicted_positive_fraction", "n_pixels"]
    with open(out_dir / "threshold_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # ---- F1-maximising thresholds (validation-derived) ---------------------
    f1_rows = {}
    for scope in ("combined", "A", "B"):
        scores = ft["scores"] if scope == "combined" else ft["pairs"][scope]["scores"]
        labels = ft["labels"] if scope == "combined" else ft["pairs"][scope]["labels"]
        best = best_f1_threshold(scores, labels)
        f1_rows[scope] = best
    print("F1-max (validation-derived):", {k: (v["threshold"], round(v["f1"], 4))
                                           for k, v in f1_rows.items()})

    # ---- per-pair metrics with prevalence ----------------------------------
    pair_rows = []
    for scope in ("combined", "A", "B"):
        scores = ft["scores"] if scope == "combined" else ft["pairs"][scope]["scores"]
        labels = ft["labels"] if scope == "combined" else ft["pairs"][scope]["labels"]
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        row = classification(scores, labels, 0.5)
        row.update({"scope": scope, "positives": positives, "negatives": negatives,
                    "prevalence": positives / (positives + negatives),
                    "ap": float(average_precision(labels, scores)),
                    "ap_f1max_threshold": f1_rows[scope]["threshold"],
                    "ap_f1max_f1": f1_rows[scope]["f1"],
                    "ap_f1max_precision": f1_rows[scope]["precision"],
                    "ap_f1max_recall": f1_rows[scope]["recall"]})
        pair_rows.append(row)
    with open(out_dir / "pair_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    # ---- checkpoint comparison --------------------------------------------
    comparison = []
    for scope in ("combined", "A", "B"):
        entry = {"scope": scope}
        for name in ("oscd", "fine_tuned"):
            data = results[name]
            scores = data["scores"] if scope == "combined" else data["pairs"][scope]["scores"]
            labels = data["labels"] if scope == "combined" else data["pairs"][scope]["labels"]
            entry["%s_ap" % name] = float(average_precision(labels, scores))
            entry["%s_prevalence" % name] = float((labels == 1).mean())
            entry["%s_pixels" % name] = int(scores.size)
        comparison.append(entry)
    with open(out_dir / "checkpoint_comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print("comparison:", json.dumps(comparison, indent=1))

    json.dump({"thresholds": {k: v for k, v in f1_rows.items()},
               "validation_derived": True,
               "note": ("All thresholds here are selected on validation pixels and are "
                        "therefore not unbiased estimates of test performance")},
              open(out_dir / "threshold_selection.json", "w"), indent=1)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
