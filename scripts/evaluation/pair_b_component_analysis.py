"""Component-level spatial analysis of the Pair B regression.

Groups reviewed change pixels into connected regions and asks, per region size,
whether each checkpoint detected it. This answers "large construction areas or small
structures" with counts instead of impressions.

    python scripts/evaluation/pair_b_component_analysis.py \
        --output outputs/evaluation/local_manual_v2/pair_b_checkpoint_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.sessions import build_datasets, load_model, predict_pair

PAIR = "B"


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
    device = torch.device(args.device)

    datasets, _, _ = build_datasets(args)
    entry = datasets[PAIR]
    target = entry["target"]
    reviewed = target != IGNORE_VALUE

    maps = {}
    for name, path in (("oscd", args.oscd), ("fine_tuned", args.fine_tuned)):
        model, _ = load_model(path, device)
        acc, _, _ = predict_pair(model, PAIR, entry, device)
        maps[name] = np.where(acc.count > 0, acc.total / np.maximum(acc.count, 1), np.nan)
        del model
        torch.cuda.empty_cache()

    change = reviewed & (target == 1)
    labels, count = ndi.label(change, structure=np.ones((3, 3)))
    sizes = ndi.sum(change, labels, index=range(1, count + 1))
    rows = []
    for index in range(1, count + 1):
        region = labels == index
        pixels = int(region.sum())
        o_detected = float((maps["oscd"][region] >= 0.5).mean())
        f_detected = float((maps["fine_tuned"][region] >= 0.5).mean())
        rows.append({"component": index, "pixels": pixels,
                     "oscd_detected_fraction": o_detected,
                     "fine_tuned_detected_fraction": f_detected,
                     "oscd_mean_score": float(np.nanmean(maps["oscd"][region])),
                     "fine_tuned_mean_score": float(np.nanmean(maps["fine_tuned"][region]))})

    with open(out / "pair_b_component_analysis.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    size_array = np.array([r["pixels"] for r in rows])
    quantiles = np.percentile(size_array, [25, 50, 75]) if size_array.size else [0, 0, 0]
    buckets = {"small (<= p25)": size_array <= quantiles[0],
               "medium (p25-p75)": (size_array > quantiles[0]) & (size_array <= quantiles[2]),
               "large (> p75)": size_array > quantiles[2]}
    summary = {}
    for name, mask in buckets.items():
        if not mask.any():
            continue
        selected = [rows[i] for i in np.flatnonzero(mask)]
        summary[name] = {
            "components": int(len(selected)),
            "pixels": int(sum(r["pixels"] for r in selected)),
            "mean_component_pixels": float(np.mean([r["pixels"] for r in selected])),
            "oscd_detected_fraction": float(np.mean([r["oscd_detected_fraction"]
                                                     for r in selected])),
            "fine_tuned_detected_fraction": float(np.mean([r["fine_tuned_detected_fraction"]
                                                           for r in selected])),
            "oscd_mean_score": float(np.mean([r["oscd_mean_score"] for r in selected])),
            "fine_tuned_mean_score": float(np.mean([r["fine_tuned_mean_score"]
                                                    for r in selected])),
        }

    payload = {"components": int(count),
               "component_size_pixels": {"min": int(size_array.min()) if size_array.size else 0,
                                         "p25": float(quantiles[0]),
                                         "median": float(quantiles[1]),
                                         "p75": float(quantiles[2]),
                                         "max": int(size_array.max()) if size_array.size else 0},
               "by_size_bucket": summary,
               "note": ("detected_fraction is the share of a component's pixels the "
                        "checkpoint scores at or above 0.50")}
    (out / "pair_b_component_analysis.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
