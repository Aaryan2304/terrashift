"""Compare overlapping-window aggregation rules on the validation split.

The saved canvases hold only the last-writer-wins result, so this re-runs the
model once over the 62 validation windows and keeps every window prediction.
From those it builds three canvases:

    last      the current rule: later window in index order overwrites earlier
    mean      arithmetic mean of every window covering the pixel
    blend     weighted mean, each window tapered towards its border so the
              prediction from the patch centre dominates

For each it reports AP, pixel precision/recall/F1 and object
precision/recall/F1 at the same operating point (threshold 0.90, 64 px minimum
area, scene-level matching). It also measures, inside the overlap regions only,
how much the three canvases differ and how much the two contributing windows
disagreed to begin with.

    python experiments/oscd_improvement/aggregate_overlaps.py \
        --checkpoint outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt \
        --output experiments/oscd_improvement/results/aggregation.json
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data.oscd_original import OSCDOriginalPatchDataset
from eval import component_table, evaluate_patch
from models.siamese_unet import SiameseUNet
from utils.metrics import average_precision, metrics_from_counts, roc_auc

PATCH = 128
THRESHOLD = 0.90
MIN_AREA = 64


def taper(size=PATCH, low=0.1):
    """Raised-cosine window from 1.0 at the centre to `low` at the border."""
    index = np.arange(size, dtype=np.float64)
    ramp = 0.5 * (1.0 + np.cos(np.pi * (2 * index / (size - 1) - 1)))
    return low + (1.0 - low) * ramp


def window_weight():
    one_d = taper()
    return np.outer(one_d, one_d)


def build_canvases(windows, dims):
    """Three aggregations from a list of (city, r0, c0, prob)."""
    last = {city: np.zeros(shape, np.float32) for city, shape in dims.items()}
    plain = {city: np.zeros(shape, np.float32) for city, shape in dims.items()}
    weighted = {city: np.zeros(shape, np.float32) for city, shape in dims.items()}
    weight = {city: np.zeros(shape, np.float32) for city, shape in dims.items()}
    counts = {city: np.zeros(shape, np.int32) for city, shape in dims.items()}
    kernel = window_weight().astype(np.float32)

    for city, r0, c0, prob in windows:
        target = last[city]
        target[r0:r0 + PATCH, c0:c0 + PATCH] = prob          # later wins
        plain[city][r0:r0 + PATCH, c0:c0 + PATCH] += prob
        weighted[city][r0:r0 + PATCH, c0:c0 + PATCH] += prob * kernel
        weight[city][r0:r0 + PATCH, c0:c0 + PATCH] += kernel
        counts[city][r0:r0 + PATCH, c0:c0 + PATCH] += 1

    mean = {}
    blend = {}
    for city in dims:
        covered = counts[city] > 0
        mean[city] = np.where(covered,
                              plain[city] / np.maximum(counts[city], 1), 0.0).astype(np.float32)
        blend[city] = np.where(covered,
                               weighted[city] / np.maximum(weight[city], 1e-6), 0.0).astype(np.float32)
    return {"last": last, "mean": mean, "blend": blend}, counts


def apply_min_area(mask, min_area=MIN_AREA):
    if min_area <= 0:
        return mask
    labels, areas = component_table(mask)
    if not areas.size:
        return mask
    keep = np.zeros(areas.size + 1, dtype=bool)
    keep[1:] = areas >= min_area
    return keep[labels]


def score(canvases, truth, threshold=THRESHOLD):
    out = {}
    for name, per_city in canvases.items():
        scores, targets = [], []
        tp = fp = fn = tn = 0
        otp = ofp = ofn = 0
        for city, prob in per_city.items():
            target = truth[city] == 1
            scores.append(prob.ravel())
            targets.append(target.ravel())
            mask = apply_min_area(prob >= threshold)
            tp += int((mask & target).sum()); fp += int((mask & ~target).sum())
            fn += int((~mask & target).sum()); tn += int((~mask & ~target).sum())
            result = evaluate_patch(mask, target)
            otp += result["tp"]; ofp += result["fp"]; ofn += result["fn"]
        pixel = metrics_from_counts(tp, fp, fn, tn)
        obj = metrics_from_counts(otp, ofp, ofn, 0)
        s = np.concatenate(scores); t = np.concatenate(targets)
        out[name] = {
            "ap": float(average_precision(t, s)), "auc": float(roc_auc(t, s)),
            "pixel_tp": tp, "pixel_fp": fp, "pixel_fn": fn,
            "pixel_precision": pixel["precision"], "pixel_recall": pixel["recall"],
            "pixel_f1": pixel["f1"],
            "object_tp": otp, "object_fp": ofp, "object_fn": ofn,
            "object_precision": obj["precision"], "object_recall": obj["recall"],
            "object_f1": obj["f1"],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", default=os.path.join(ROOT, "outputs", "patches",
                                                        "oscd_original_index.csv"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    bands = tuple(checkpoint.get("bands"))
    model = SiameseUNet(in_channels=len(bands), base=checkpoint.get("base", 16))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device).eval()

    dataset = OSCDOriginalPatchDataset(args.index, "validation", bands,
                                       checkpoint["mean"], checkpoint["std"])
    rows = dataset.rows
    dims = {}
    for row in rows:
        city = row["city"]
        h = int(row["row_off"]) + int(row["height"])
        w = int(row["col_off"]) + int(row["width"])
        dh, dw = dims.get(city, (0, 0))
        dims[city] = (max(dh, h), max(dw, w))

    windows, truth = [], {}
    for city in dims:
        truth[city] = np.zeros(dims[city], np.uint8)
    batch_size = 8
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        samples = [dataset[i] for i in range(start, stop)]
        x1 = torch.stack([s["image1"] for s in samples]).to(args.device)
        x2 = torch.stack([s["image2"] for s in samples]).to(args.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(x1, x2))[:, 0].float().cpu().numpy()
        for offset, index in enumerate(range(start, stop)):
            row = rows[index]
            city = row["city"]
            r0, c0 = int(row["row_off"]), int(row["col_off"])
            h, w = int(row["height"]), int(row["width"])
            windows.append((city, r0, c0, probabilities[offset]))
            truth[city][r0:r0 + h, c0:c0 + w] = samples[offset]["mask"].numpy().astype(np.uint8)
        print("  %d/%d" % (stop, len(dataset)), end="\r")
    print()

    canvases, counts = build_canvases(windows, dims)
    results = score(canvases, truth)

    # --- how much do the methods differ, inside overlap regions only
    overlap_stats = {}
    for city in dims:
        twice = counts[city] > 1
        if not twice.any():
            continue
        last = canvases["last"][city][twice]
        mean = canvases["mean"][city][twice]
        blend = canvases["blend"][city][twice]
        overlap_stats[city] = {
            "overlap_pixels": int(twice.sum()),
            "last_vs_mean_mae": float(np.abs(last - mean).mean()),
            "last_vs_mean_max": float(np.abs(last - mean).max()),
            "last_vs_blend_mae": float(np.abs(last - blend).mean()),
            "mean_vs_blend_mae": float(np.abs(mean - blend).mean()),
            "mean_vs_blend_max": float(np.abs(mean - blend).max()),
            # Pixels whose detection state at the operating threshold depends on
            # the aggregation rule. This is the number that matters for a
            # detection register: a flip is one object appearing or vanishing.
            "flips_last_vs_mean": int(((last >= THRESHOLD) != (mean >= THRESHOLD)).sum()),
            "flips_last_vs_blend": int(((last >= THRESHOLD) != (blend >= THRESHOLD)).sum()),
        }

    # --- how much did the contributing windows disagree before aggregation
    per_city_windows = collections.defaultdict(list)
    for city, r0, c0, prob in windows:
        per_city_windows[city].append((r0, c0, prob))
    disagreement = {}
    for city, entries in per_city_windows.items():
        worst = 0.0
        pairs = 0
        for i, (r0, c0, p0) in enumerate(entries):
            for r1, c1, p1 in entries[i + 1:]:
                top = max(r0, r1); left = max(c0, c1)
                bottom = min(r0 + PATCH, r1 + PATCH); right = min(c0 + PATCH, c1 + PATCH)
                if bottom <= top or right <= left:
                    continue
                a = p0[top - r0:bottom - r0, left - c0:right - c0]
                b = p1[top - r1:bottom - r1, left - c1:right - c1]
                worst = max(worst, float(np.abs(a - b).max()))
                pairs += 1
        disagreement[city] = {"overlapping_window_pairs": pairs,
                              "max_abs_difference_between_windows": worst}

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": "validation",
        "threshold": THRESHOLD,
        "min_area_px": MIN_AREA,
        "windows": len(windows),
        "results": results,
        "overlap_region_differences": overlap_stats,
        "window_disagreement": disagreement,
        "notes": [
            "Official test split not read.",
            "Scene-level reconstruction and object matching identical for all three methods.",
            "The taper is a raised cosine from 1.0 at the patch centre to 0.1 at the border.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)

    print("\noperating point: threshold %.2f, min area %d px\n" % (THRESHOLD, MIN_AREA))
    header = ("%-8s %7s %7s %7s %7s %7s %7s %7s %7s %7s"
              % ("method", "AP", "AUC", "pxP", "pxR", "pxF1", "obP", "obR", "obF1", ""))
    print(header)
    print("-" * len(header))
    for name in ("last", "mean", "blend"):
        r = results[name]
        print("%-8s %7.4f %7.4f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f"
              % (name, r["ap"], r["auc"], r["pixel_precision"], r["pixel_recall"],
                 r["pixel_f1"], r["object_precision"], r["object_recall"], r["object_f1"]))

    print("\noverlap-region differences:")
    for city, s in overlap_stats.items():
        print("  %-9s %d px  last-vs-mean MAE %.2e  last-vs-blend MAE %.2e  mean-vs-blend MAE %.2e"
              % (city, s["overlap_pixels"], s["last_vs_mean_mae"],
                 s["last_vs_blend_mae"], s["mean_vs_blend_mae"]))
    print("\nwindow disagreement before aggregation:")
    for city, s in disagreement.items():
        print("  %-9s pairs %d  max |window - window| %.4f"
              % (city, s["overlapping_window_pairs"],
                 s["max_abs_difference_between_windows"]))
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()
