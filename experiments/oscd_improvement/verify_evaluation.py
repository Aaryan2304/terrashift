"""Verification of the scene-level evaluation and the area filter.

Four checks, each answering a question the report claims:

1. **Coverage and overlap.** Every canvas pixel written at least once, and how
   many are written more than once.
2. **Reconstruction rule.** The saved canvas must equal, pixel for pixel, the
   prediction of the window that the collector wrote last. Edge windows are
   re-run and compared, and the disagreement between the two windows over the
   shared border is measured.
3. **Connectivity.** The evaluator's labelling convention is demonstrated on a
   synthetic pair of blobs that touch only at a corner.
4. **Area thresholds.** Several thresholds, in pixels and in square metres at
   both 20 m and 10 m ground sampling, with the number of genuine labelled
   objects that each threshold removes.

    python experiments/oscd_improvement/verify_evaluation.py \
        --predictions experiments/oscd_improvement/predictions/baseline_seed42_val.npz \
        --checkpoint outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt \
        --output experiments/oscd_improvement/results/verification.json
"""

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data.oscd_original import OSCDOriginalPatchDataset, patch_origins
from eval import component_table, evaluate_patch
from models.siamese_unet import SiameseUNet
from utils.metrics import metrics_from_counts

PATCH = 128
AREA_THRESHOLDS_PX = [0, 4, 8, 16, 32, 64, 128, 192, 256, 512]
GSM20_M2 = 20.0 * 20.0
GSM10_M2 = 10.0 * 10.0


def coverage_report(index_path, split="validation"):
    """Windows per city, canvas size, and how often each pixel is written."""
    rows = [r for r in csv.DictReader(open(index_path)) if r["split"] == split]
    dims, origins = {}, {}
    for row in rows:
        city = row["city"]
        dims[city] = (max(dims.get(city, (0, 0))[0], int(row["row_off"]) + int(row["height"])),
                      max(dims.get(city, (0, 0))[1], int(row["col_off"]) + int(row["width"])))
    for city, (height, width) in dims.items():
        origins[city] = (patch_origins(height, PATCH, PATCH),
                         patch_origins(width, PATCH, PATCH))

    out = {}
    for city, (height, width) in dims.items():
        counts = np.zeros((height, width), dtype=np.int32)
        for r0 in origins[city][0]:
            for c0 in origins[city][1]:
                counts[r0:r0 + PATCH, c0:c0 + PATCH] += 1
        out[city] = {
            "canvas": [height, width],
            "windows": len(origins[city][0]) * len(origins[city][1]),
            "pixels": int(counts.size),
            "uncovered": int((counts == 0).sum()),
            "once": int((counts == 1).sum()),
            "twice": int((counts == 2).sum()),
            "more_than_twice": int((counts > 2).sum()),
            "overlap_fraction": round(float((counts > 1).mean()), 6),
        }
    return out


def reconstruction_check(predictions_path, checkpoint_path, index_path, config_path,
                         device="cuda"):
    """Re-run the edge windows and confirm the canvas holds the last writer."""
    cfg = yaml.safe_load(open(config_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    bands = tuple(checkpoint.get("bands"))
    model = SiameseUNet(in_channels=len(bands), base=checkpoint.get("base", 16))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    dataset = OSCDOriginalPatchDataset(index_path, "validation", bands,
                                       checkpoint["mean"], checkpoint["std"])
    saved = np.load(predictions_path)

    by_city = collections.defaultdict(list)
    for index, row in enumerate(dataset.rows):
        by_city[row["city"]].append((index, row))

    report = {}
    for city, entries in by_city.items():
        height = max(int(r["row_off"]) + int(r["height"]) for _, r in entries)
        width = max(int(r["col_off"]) + int(r["width"]) for _, r in entries)
        row_origins = patch_origins(height, PATCH, PATCH)
        col_origins = patch_origins(width, PATCH, PATCH)
        # Windows that share a border with a later window.
        edge = []
        for index, row in entries:
            r0, c0 = int(row["row_off"]), int(row["col_off"])
            if r0 != row_origins[-1] and c0 != col_origins[-1]:
                continue
            edge.append((index, r0, c0))
        if not edge:
            continue

        saved_prob = saved["prob_" + city]
        worst = 0.0
        checked = 0
        for index, r0, c0 in edge:
            sample = dataset[index]
            with torch.no_grad():
                probability = torch.sigmoid(
                    model(sample["image1"][None].to(device),
                          sample["image2"][None].to(device)))[0, 0].cpu().numpy()
            window = saved_prob[r0:r0 + PATCH, c0:c0 + PATCH]
            # The canvas must match this window wherever no later window covers it.
            later = np.zeros((PATCH, PATCH), dtype=bool)
            for r1 in row_origins:
                for c1 in col_origins:
                    if (r1, c1) <= (r0, c0):
                        continue
                    overlap_rows = max(0, min(r0 + PATCH, r1 + PATCH) - max(r0, r1))
                    overlap_cols = max(0, min(c0 + PATCH, c1 + PATCH) - max(c0, c1))
                    if overlap_rows and overlap_cols:
                        top = max(r0, r1) - r0
                        left = max(c0, c1) - c0
                        later[top:top + overlap_rows, left:left + overlap_cols] = True
            if (~later).any():
                delta = float(np.abs(window[~later] - probability[~later]).max())
                worst = max(worst, delta)
                checked += 1
        report[city] = {"edge_windows_checked": checked,
                        "max_abs_difference_outside_later_windows": worst}
    return report


def connectivity_check():
    """Two blobs touching only at a corner: 4-connectivity splits them."""
    mask = np.zeros((6, 6), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    labels, areas = component_table(mask)
    return {"pixels": 2, "components_found": int(len(areas)),
            "interpretation": ("4-connectivity: diagonal neighbours are separate "
                               "objects" if len(areas) == 2 else
                               "8-connectivity: diagonal neighbours are one object"),
            "scipy_default_structure": "cross (4-connectivity)"}


def area_sweep(predictions, threshold):
    """Object counts and lost true detections for each minimum area."""
    data = np.load(predictions)
    cities = sorted({k.split("_", 1)[1] for k in data.files if k.startswith("prob_")})
    rows = []
    for min_area in AREA_THRESHOLDS_PX:
        tp = fp = fn = 0
        lost = 0
        small_truth = 0
        for city in cities:
            prob, truth = data["prob_" + city], data["truth_" + city]
            mask = prob >= threshold
            if min_area > 0:
                labels, areas = component_table(mask)
                if areas.size:
                    keep = np.zeros(areas.size + 1, dtype=bool)
                    keep[1:] = areas >= min_area
                    mask = keep[labels]
            result = evaluate_patch(mask, truth == 1)
            tp += result["tp"]; fp += result["fp"]; fn += result["fn"]
            # Truth objects smaller than the threshold cannot be matched by a
            # prediction of at least min_area pixels at IoU 0.5.
            _, truth_areas = component_table(truth == 1)
            small_truth += int((truth_areas < min_area).sum())
        metrics = metrics_from_counts(tp, fp, fn, 0)
        rows.append({
            "min_area_px": min_area,
            "area_m2_20m": int(min_area * GSM20_M2),
            "area_m2_10m": int(min_area * GSM10_M2),
            "km2_20m": round(min_area * GSM20_M2 / 1e6, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "truth_objects_below_threshold": small_truth,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", default=os.path.join(ROOT, "outputs", "patches",
                                                        "oscd_original_index.csv"))
    parser.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-reconstruction", action="store_true",
                        help="skip the GPU re-run of edge windows")
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    report = {"coverage": coverage_report(args.index),
              "connectivity": connectivity_check(),
              "area_sweep": area_sweep(args.predictions, args.threshold),
              "threshold": args.threshold}
    if args.skip_reconstruction:
        report["reconstruction"] = None
    else:
        report["reconstruction"] = reconstruction_check(
            args.predictions, args.checkpoint, args.index, args.config)
    report["iou_threshold"] = 0.5

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)

    print("coverage:")
    for city, info in report["coverage"].items():
        print("  %-9s %s windows %d  uncovered %d  twice %d (%.3f%%)"
              % (city, info["canvas"], info["windows"], info["uncovered"],
                 info["twice"], 100 * info["overlap_fraction"]))
    print("connectivity:", report["connectivity"]["components_found"],
          "component(s) for", report["connectivity"]["pixels"], "diagonal pixels ->",
          report["connectivity"]["interpretation"])
    if report["reconstruction"]:
        for city, info in report["reconstruction"].items():
            print("reconstruction %-9s windows %d  max |canvas - window| %.2e"
                  % (city, info["edge_windows_checked"],
                     info["max_abs_difference_outside_later_windows"]))
    print("\narea sweep at threshold %.2f:" % args.threshold)
    print("  %-6s %-10s %-10s %6s %5s %5s %7s %7s %7s %8s"
          % ("px", "m2@20m", "m2@10m", "tp", "fp", "fn", "P", "R", "F1", "small"))
    for row in report["area_sweep"]:
        print("  %-6d %-10d %-10d %6d %5d %5d %7.3f %7.3f %7.3f %8d"
              % (row["min_area_px"], row["area_m2_20m"], row["area_m2_10m"],
                 row["tp"], row["fp"], row["fn"], row["precision"],
                 row["recall"], row["f1"], row["truth_objects_below_threshold"]))
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()
