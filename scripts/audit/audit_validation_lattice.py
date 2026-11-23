"""Read-only audit: can validation use the inference lattice, and what does it cost?

Compares the inference tiler, the windows holding at least one reviewed validation pixel, and the
split-contained index now in use. For each reviewed pixel it reports the votes from every set, how
many are context-safe, and the distance to the nearest training-labelled pixel. Only the validation
stripe is allocated and test labels are never read.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.local_patches import patch_origins
from terrashift.data.validation_windows import (DEFAULT_BLOCK_ROWS, DEFAULT_HALO_ROWS,
                                     VALIDATION_CODE, core_slice, inference_strips,
                                     select_windows, validation_target, window_grid)

RADIUS = 23
THEORETICAL_LEFT, THEORETICAL_RIGHT = 51, 48


def load_pair(pair, labels_dir, partition_dir):
    with rasterio.open(os.path.join(labels_dir, "pair_%s_v2" % pair.lower(),
                                    "manual_labels_%s.tif" % pair)) as dataset:
        labels = dataset.read(1)
    with rasterio.open(os.path.join(partition_dir,
                                    "pair_%s_partition_10m.tif" % pair)) as dataset:
        partition = dataset.read(1)
    return labels, partition


def stripe_bounds(partition):
    columns = np.flatnonzero(partition[0] == VALIDATION_CODE)
    if columns.size == 0:
        raise SystemExit("no validation stripe")
    return int(columns[0]), int(columns[-1]) + 1


def accumulate(windows, patch, shape, stripe, counters, radius):
    """Add each window's votes, and its safe votes, onto the stripe slice."""
    votes, safe = counters
    col_lo, col_hi = stripe
    for window in windows:
        offset, count = core_slice(window, patch)
        row0 = window["row0"] + offset
        row1 = row0 + count
        c0 = max(window["col0"], col_lo)
        c1 = min(window["col0"] + patch, col_hi)
        if row1 <= row0 or c1 <= c0:
            continue
        votes[row0:row1, c0 - col_lo:c1 - col_lo] += 1
        # Safe band: the part of the window at least `radius` from every border.
        s0 = max(window["row0"] + radius, row0)
        s1 = min(window["row0"] + patch - radius, row1)
        t0 = max(window["col0"] + radius, col_lo)
        t1 = min(window["col0"] + patch - radius, col_hi)
        if s1 > s0 and t1 > t0:
            safe[s0:s1, t0 - col_lo:t1 - col_lo] += 1


def full_context_safe(window, patch, row, col):
    """True when the output location's own support lies inside the patch."""
    local_row = row - window["row0"]
    local_col = col - window["col0"]
    return (THEORETICAL_LEFT <= local_row <= patch - 1 - THEORETICAL_RIGHT
            and THEORETICAL_LEFT <= local_col <= patch - 1 - THEORETICAL_RIGHT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-dir", default=os.path.join(ROOT, "outputs", "labels"))
    parser.add_argument("--partition-dir",
                        default=os.path.join(ROOT, "outputs", "partitions_v2_observation_valid"))
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    report = {"radius_px": RADIUS, "block_rows": DEFAULT_BLOCK_ROWS,
              "halo_rows": DEFAULT_HALO_ROWS, "patch_sizes": [128, 256],
              "pairs": {}, "note": ("read-only; no index written, test labels never read")}

    for pair in ("A", "B"):
        labels, partition = load_pair(pair, args.labels_dir, args.partition_dir)
        target = validation_target(labels, partition)
        col_lo, col_hi = stripe_bounds(partition)
        height = labels.shape[0]
        reviewed = np.argwhere(target != 255)
        entry = {"reviewed_validation_px": int(reviewed.shape[0]),
                 "stripe_columns": [col_lo, col_hi], "patch_sizes": {}}

        # Spatial isolation: distance to the nearest training-labelled pixel.
        training_labelled = (labels != 255) & (partition == 1)
        if training_labelled.any():
            distance = ndi.distance_transform_edt(~training_labelled)
            d = distance[reviewed[:, 0], reviewed[:, 1]]
            entry["isolation"] = {
                "min_px": float(d.min()), "median_px": float(np.median(d)),
                "p90_px": float(np.percentile(d, 90)), "max_px": float(d.max()),
                "pct_within_23px": round(100 * float((d <= 23).mean()), 2),
                "pct_within_50px": round(100 * float((d <= 50).mean()), 2),
                "pct_within_100px": round(100 * float((d <= 100).mean()), 2),
            }
        else:
            entry["isolation"] = {"note": "no training-labelled pixels in this pair"}

        for patch in (128, 256):
            stride = patch // 2
            shape = (height, col_hi - col_lo)

            proposed = select_windows(target, patch, stride,
                                      tile_width=labels.shape[1])
            # The current policy: split-contained windows over the whole tile.
            old = []
            for row0 in patch_origins(height, patch, stride):
                for col0 in patch_origins(labels.shape[1], patch, stride):
                    if col0 < col_lo or col0 + patch > col_hi:
                        continue          # containment: wholly inside the stripe
                    first, last = row0, row0 + patch
                    if (target[first:last, col0:col0 + patch] != 255).sum() < 1:
                        continue
                    old.append({"row0": row0, "col0": col0,
                                "core_start": 0, "core_stop": height})

            # Every proposed window must be in the canonical lattice of its strip;
            # every canonical window holding a reviewed pixel must be proposed.
            canonical = set()
            for core_start, core_stop, read_start, read_stop in inference_strips(height):
                for row0, col0 in window_grid(read_start, read_stop, labels.shape[1],
                                              patch, stride):
                    first, last = max(row0, core_start), min(row0 + patch, core_stop)
                    if last <= first:
                        continue
                    if (target[first:last, col0:min(col0 + patch, labels.shape[1])] != 255).any():
                        canonical.add((core_start, row0, col0))
            proposed_keys = {(w["core_start"], w["row0"], w["col0"]) for w in proposed}

            votes_new = np.zeros(shape, dtype=np.uint8)
            safe_new = np.zeros(shape, dtype=np.uint8)
            accumulate(proposed, patch, shape, (col_lo, col_hi), (votes_new, safe_new), RADIUS)

            votes_old = np.zeros(shape, dtype=np.uint8)
            safe_old = np.zeros(shape, dtype=np.uint8)
            accumulate(old, patch, shape, (col_lo, col_hi), (votes_old, safe_old), RADIUS)

            rows = reviewed[:, 0]
            cols = reviewed[:, 1] - col_lo
            v_new = votes_new[rows, cols].astype(np.int32)
            s_new = safe_new[rows, cols].astype(np.int32)
            v_old = votes_old[rows, cols].astype(np.int32)
            s_old = safe_old[rows, cols].astype(np.int32)

            fraction_new = np.divide(s_new, v_new, out=np.zeros_like(s_new, dtype=float),
                                     where=v_new > 0)
            fraction_old = np.divide(s_old, v_old, out=np.zeros_like(s_old, dtype=float),
                                     where=v_old > 0)

            # Context quality of the votes the containment rule removes.
            excluded_votes = int((v_new - v_old).sum())
            excluded_safe = int((s_new - s_old).sum())

            # Full theoretical context, counted exactly on the sample of windows.
            full_new = np.zeros(v_new.shape, dtype=np.int32)
            for window in proposed:
                offset, count = core_slice(window, patch)
                row0 = window["row0"] + offset
                row1 = row0 + count
                inside = ((rows >= row0) & (rows < row1)
                          & (reviewed[:, 1] >= window["col0"])
                          & (reviewed[:, 1] < window["col0"] + patch))
                if not inside.any():
                    continue
                local_rows = rows[inside] - window["row0"]
                local_cols = reviewed[inside, 1] - window["col0"]
                ok = ((local_rows >= THEORETICAL_LEFT)
                      & (local_rows <= patch - 1 - THEORETICAL_RIGHT)
                      & (local_cols >= THEORETICAL_LEFT)
                      & (local_cols <= patch - 1 - THEORETICAL_RIGHT))
                full_new[inside] += ok.astype(np.int32)

            entry["patch_sizes"][str(patch)] = {
                "stride": stride,
                "proposed_windows": len(proposed),
                "old_index_windows": len(old),
                "canonical_windows_with_reviewed_px": len(canonical),
                "proposed_equals_canonical": proposed_keys == canonical,
                "proposed_missing_from_canonical": len(proposed_keys - canonical),
                "canonical_missing_from_proposed": len(canonical - proposed_keys),
                "votes": {
                    "mean_new": round(float(v_new.mean()), 3),
                    "mean_old": round(float(v_old.mean()), 3),
                    "min_new": int(v_new.min()), "max_new": int(v_new.max()),
                    "identical_counts": bool(np.array_equal(v_new, v_old)),
                    "pct_pixels_with_fewer_old_votes": round(
                        100 * float((v_old < v_new).mean()), 2),
                },
                "safe": {
                    "mean_fraction_new": round(float(fraction_new.mean()), 4),
                    "mean_fraction_old": round(float(fraction_old.mean()), 4),
                    "pct_all_safe_new": round(100 * float((s_new == v_new).mean()), 2),
                    "pct_all_safe_old": round(100 * float((s_old == v_old).mean()), 2),
                    "mean_safe_votes_new": round(float(s_new.mean()), 3),
                    "mean_safe_votes_old": round(float(s_old.mean()), 3),
                },
                "excluded_by_containment": {
                    "votes": excluded_votes,
                    "safe_votes": excluded_safe,
                    "context_fraction_of_excluded": (
                        round(excluded_safe / excluded_votes, 4) if excluded_votes else None),
                    "context_fraction_of_retained": (
                        round(float(s_old.sum()) / float(v_old.sum()), 4)
                        if v_old.sum() else None),
                },
                "full_theoretical_context": {
                    "mean_safe_fraction_new": round(
                        float(np.divide(full_new, v_new, out=np.zeros_like(full_new, dtype=float),
                                        where=v_new > 0).mean()), 4),
                    "pct_all_safe_new": round(
                        100 * float((full_new == v_new).mean()), 2),
                },
            }
            print("%s %d/%d: proposed %d windows (canonical match %s), old %d | "
                  "votes new %.2f old %.2f | safe fraction new %.3f old %.3f | excluded votes %d"
                  % (pair, patch, stride, len(proposed), proposed_keys == canonical,
                     len(old), v_new.mean(), v_old.mean(), fraction_new.mean(),
                     fraction_old.mean(), excluded_votes))

        report["pairs"][pair] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
