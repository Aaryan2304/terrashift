"""Across-seed comparison at a fixed, validation-selected operating point.

Every run is scored the same way so the seeds are comparable: scene-level
average precision (threshold-free), and pixel and object metrics at threshold
0.90 with a 64-pixel minimum component.

    python experiments/oscd_improvement/across_seeds.py \
        --root experiments/oscd_improvement \
        --output experiments/oscd_improvement/results/across_seeds.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from eval import component_table, evaluate_patch
from utils.metrics import average_precision, metrics_from_counts

THRESHOLD = 0.90
MIN_AREA_PX = 64


def score(path, threshold=THRESHOLD, min_area=MIN_AREA_PX):
    data = np.load(path)
    cities = sorted({k.split("_", 1)[1] for k in data.files if k.startswith("prob_")})
    scores, targets = [], []
    tp = fp = fn = tn = 0
    otp = ofp = ofn = 0
    for city in cities:
        prob, truth = data["prob_" + city], data["truth_" + city]
        scores.append(prob.ravel())
        targets.append((truth == 1).ravel())
        mask = prob >= threshold
        if min_area > 0:
            labels, areas = component_table(mask)
            if areas.size:
                keep = np.zeros(areas.size + 1, dtype=bool)
                keep[1:] = areas >= min_area
                mask = keep[labels]
        target = truth == 1
        tp += int((mask & target).sum()); fp += int((mask & ~target).sum())
        fn += int((~mask & target).sum()); tn += int((~mask & ~target).sum())
        result = evaluate_patch(mask, target)
        otp += result["tp"]; ofp += result["fp"]; ofn += result["fn"]
    pixel = metrics_from_counts(tp, fp, fn, tn)
    obj = metrics_from_counts(otp, ofp, ofn, 0)
    return {
        "ap": float(average_precision(np.concatenate(targets), np.concatenate(scores))),
        "pixel_precision": pixel["precision"], "pixel_recall": pixel["recall"],
        "pixel_f1": pixel["f1"],
        "object_tp": otp, "object_fp": ofp, "object_fn": ofn,
        "object_precision": obj["precision"], "object_recall": obj["recall"],
        "object_f1": obj["f1"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prediction_dir = os.path.join(args.root, "predictions")
    rows = []
    for path in sorted(glob.glob(os.path.join(prediction_dir, "*_val.npz"))):
        name = os.path.basename(path).replace("_val.npz", "")
        match = re.match(r"(.*)_seed(\d+)$", name)
        if not match:
            continue
        family, seed = match.group(1), int(match.group(2))
        result = score(path)
        result.update({"run": name, "family": family, "seed": seed})
        rows.append(result)

    if not rows:
        raise SystemExit("no prediction files found in %s" % prediction_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("operating point: threshold %.2f, min area %d px\n" % (THRESHOLD, MIN_AREA_PX))
    header = ("%-22s %6s %7s %7s %7s %6s %6s %6s %7s"
              % ("run", "seed", "AP", "pixF1", "objF1", "objTP", "objFP", "objFN", ""))
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["family"], r["seed"])):
        print("%-22s %6d %7.4f %7.4f %7.4f %6d %6d %6d"
              % (row["family"], row["seed"], row["ap"], row["pixel_f1"],
                 row["object_f1"], row["object_tp"], row["object_fp"], row["object_fn"]))

    print("\nsummary (sample SD, ddof=1; a single seed has no SD):")
    families = {}
    for row in rows:
        families.setdefault(row["family"], []).append(row)
    for family, group in sorted(families.items()):
        aps = [r["ap"] for r in group]
        f1s = [r["object_f1"] for r in group]
        line = "%-22s seeds %d  AP %.4f" % (family, len(group), statistics.mean(aps))
        if len(group) > 1:
            line += " +/- %.4f (SD)" % statistics.stdev(aps)
        line += "   object F1 %.4f" % statistics.mean(f1s)
        if len(group) > 1:
            line += " +/- %.4f (SD)" % statistics.stdev(f1s)
        print(line)

    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    summary = {}
    for family, group in families.items():
        aps = [r["ap"] for r in group]
        f1s = [r["object_f1"] for r in group]
        summary[family] = {
            "seeds": [r["seed"] for r in group],
            "ap_values": aps, "ap_mean": statistics.mean(aps),
            "ap_sd": statistics.stdev(aps) if len(group) > 1 else None,
            "object_f1_values": f1s, "object_f1_mean": statistics.mean(f1s),
            "object_f1_sd": statistics.stdev(f1s) if len(group) > 1 else None,
        }
    with open(summary_path, "w") as handle:
        json.dump({"threshold": THRESHOLD, "min_area_px": MIN_AREA_PX,
                   "operating_point_note": ("fixed for comparability; each model's "
                                            "own best threshold is in results/*/thresholds.csv"),
                   "families": summary}, handle, indent=1)
    print("\nwrote", args.output)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
