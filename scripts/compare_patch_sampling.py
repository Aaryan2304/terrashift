"""Compare training patch sampling strategies on the Pair A annotations.

The configured labelled-fraction threshold (0.05) produces no Pair A patches,
because annotations cover 0.0147 % of the tile. This measures what each
alternative would actually yield, before anything is baked into the config.

Strategies compared:

    grid        every grid-aligned window whose reviewed fraction clears a
                threshold, as the indexer does today;
    centred     one window centred on each annotation centroid;
    balanced    grid windows split into change-containing and no_change-only
                pools, sampled to a 1:1 ratio.

For each: unique patches, patches per split, labelled pixels per class, spatial
spread, and how much the selected windows overlap each other.

    python scripts/compare_patch_sampling.py \
        --labels outputs/labels/pair_a_provisional/manual_labels_A.tif \
        --partition outputs/partitions_v2_observation_valid/pair_A_partition_10m.tif \
        --output outputs/labels/pair_a_qa/patch_sampling.json
"""

import argparse
import json
import os
import sys

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.oscd_original import patch_origins

SPLIT_NAMES = {1: "train", 2: "validation", 3: "test"}
THRESHOLDS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
PATCH_SIZES = [256, 128]


def window_stats(label, partition, r0, c0, size):
    """Class counts and split id for one window, or None if it spans splits."""
    part = partition[r0:r0 + size, c0:c0 + size]
    if part.shape != (size, size):
        return None
    codes = np.unique(part)
    if codes.size != 1 or codes[0] == 0:
        return None
    lab = label[r0:r0 + size, c0:c0 + size]
    reviewed = lab != 255
    return {
        "split": int(codes[0]),
        "r0": int(r0), "c0": int(c0), "size": int(size),
        "reviewed": int(reviewed.sum()),
        "change": int((lab == 1).sum()),
        "no_change": int((lab == 0).sum()),
        "reviewed_fraction": float(reviewed.mean()),
    }


def grid_pool(label, partition, size, stride, include_test=False):
    """Split-contained windows, mirroring the indexer: test is excluded by default."""
    pool = []
    for r0 in patch_origins(label.shape[0], size, stride):
        for c0 in patch_origins(label.shape[1], size, stride):
            stats = window_stats(label, partition, r0, c0, size)
            if stats is None:
                continue
            if stats["split"] == 3 and not include_test:
                continue
            pool.append(stats)
    return pool


def select_by_threshold(pool, threshold):
    return [p for p in pool if p["reviewed_fraction"] >= threshold]


def spatial_spread(patches):
    if not patches:
        return {"count": 0}
    rows = np.array([p["r0"] for p in patches])
    cols = np.array([p["c0"] for p in patches])
    centroids = np.column_stack([rows + np.array([p["size"] for p in patches]) / 2,
                                 cols + np.array([p["size"] for p in patches]) / 2])
    if len(patches) == 1:
        nearest = [float("nan")]
    else:
        nearest = []
        for index in range(len(centroids)):
            others = np.delete(centroids, index, axis=0)
            distances = np.hypot(*(others - centroids[index]).T)
            nearest.append(float(distances.min()))
    return {
        "count": len(patches),
        "row_span": [int(rows.min()), int(rows.max())],
        "col_span": [int(cols.min()), int(cols.max())],
        "median_nearest_neighbour_px": float(np.nanmedian(nearest)) if len(patches) > 1 else None,
    }


def coverage(patches, shape):
    """Fraction of the covered area that is covered by more than one window."""
    counts = np.zeros(shape, dtype=np.int16)
    for p in patches:
        counts[p["r0"]:p["r0"] + p["size"], p["c0"]:p["c0"] + p["size"]] += 1
    covered = counts > 0
    if not covered.any():
        return {"covered_px": 0, "multi_covered_fraction": 0.0}
    return {
        "covered_px": int(covered.sum()),
        "multi_covered_fraction": round(float((counts > 1).sum() / covered.sum()), 4),
    }


def summarise(name, patches, shape):
    per_split = {name_: [p for p in patches if p["split"] == code]
                 for code, name_ in SPLIT_NAMES.items()}
    return {
        "strategy": name,
        "patches": len(patches),
        "per_split": {k: len(v) for k, v in per_split.items()},
        "change_px": int(sum(p["change"] for p in patches)),
        "no_change_px": int(sum(p["no_change"] for p in patches)),
        "patches_with_change": int(sum(1 for p in patches if p["change"] > 0)),
        "patches_without_change": int(sum(1 for p in patches if p["change"] == 0)),
        "spread": spatial_spread(patches),
        "coverage": coverage(patches, shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    with rasterio.open(args.labels) as src:
        label = src.read(1)
    with rasterio.open(args.partition) as src:
        partition = src.read(1)
    if label.shape != partition.shape:
        raise SystemExit("labels and partition are not on the same grid")
    shape = label.shape
    reviewed_total = int((label != 255).sum())
    print("tile %s, reviewed %d px (%.4f%%)" % (shape, reviewed_total,
                                                100 * reviewed_total / label.size))

    report = {"tile": list(shape), "reviewed_px": reviewed_total,
              "reviewed_fraction": reviewed_total / label.size, "results": {}}

    for size in PATCH_SIZES:
        stride = size // 2
        pool = grid_pool(label, partition, size, stride)
        print("patch %d stride %d: %d split-contained windows" % (size, stride, len(pool)))
        for threshold in THRESHOLDS:
            selected = select_by_threshold(pool, threshold)
            key = "grid_%d_thr%.3f" % (size, threshold)
            report["results"][key] = summarise(key, selected, shape)
            print("  threshold %.3f -> %4d patches (%d with change)"
                  % (threshold, len(selected),
                     sum(1 for p in selected if p["change"] > 0)))

        # balanced: equal numbers of change-containing and no_change-only windows,
        # taken largest-reviewed-first so the selection is deterministic.
        with_change = sorted([p for p in pool if p["change"] > 0],
                             key=lambda p: (-p["reviewed"], p["r0"], p["c0"]))
        without = sorted([p for p in pool if p["change"] == 0],
                         key=lambda p: (-p["reviewed"], p["r0"], p["c0"]))
        take = min(len(with_change), len(without))
        balanced = with_change[:take] + without[:take]
        key = "balanced_%d" % size
        report["results"][key] = summarise(key, balanced, shape)
        print("  balanced -> %d patches (%d + %d)" % (len(balanced), take, take))

    # annotation-centred: one window per annotation centroid, snapped to the grid
    centres = []
    labels_map, count = ndi.label(label == 1)
    for index in range(1, count + 1):
        rows, cols = np.where(labels_map == index)
        if rows.size == 0:
            continue
        centres.append((int(rows.mean()), int(cols.mean())))
    for size in PATCH_SIZES:
        patches = []
        for row, col in centres:
            r0 = max(0, min(shape[0] - size, row - size // 2))
            c0 = max(0, min(shape[1] - size, col - size // 2))
            stats = window_stats(label, partition, r0, c0, size)
            if stats is not None and stats["split"] != 3:
                patches.append(stats)
        key = "centred_%d" % size
        report["results"][key] = summarise(key, patches, shape)
        print("  annotation-centred %d -> %d patches from %d change components"
              % (size, len(patches), len(centres)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()
