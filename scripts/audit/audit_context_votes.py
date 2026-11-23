"""How many prediction votes a pixel gets, and how many are context-safe.

The 98.9 % figure elsewhere only says a pixel receives at least one prediction with enough context.
When overlapping predictions are averaged with equal weight, the question that matters is what share
of its votes are context-safe. Both are answered for the two candidate lattices using the real window
logic from ``terrashift.inference`` and the real strip loop from
``scripts/inference/predict_safe_pair.py``. A vote is context-safe when its output sits at least
``RADIUS`` pixels from every patch border; the measured 90 %-mass radius and the phase-dependent full
support are reported separately. Everything is read-only.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.local_patches import patch_origins
from terrashift.inference.sliding import _window_origins

TILE = 10980
BLOCK_ROWS = 512
HALO = 256
RADIUS = 23                 # measured 90 % magnitude/energy radius, conservative max
SUPPORT_LEFT, SUPPORT_RIGHT = 51, 48   # phase-dependent theoretical, worst case
VALIDATION_CODE = 2


def inference_row_votes(patch):
    """Votes per row under the real strip loop, for one patch size.

    Returns two arrays of length TILE: how many inference windows cover the row, and
    how many of those place the row at least RADIUS from the patch border.
    """
    votes = np.zeros(TILE, dtype=np.int16)
    safe_mass = np.zeros(TILE, dtype=np.int16)
    safe_full = np.zeros(TILE, dtype=np.int16)
    for core_start in range(0, TILE, BLOCK_ROWS):
        core_stop = min(core_start + BLOCK_ROWS, TILE)
        read_start = max(0, core_start - HALO)
        read_stop = min(TILE, core_stop + HALO)
        origins = [read_start + offset
                   for offset in _window_origins(read_stop - read_start, patch, patch // 2)]
        for origin in origins:
            stop = min(origin + patch, read_stop)
            rows = slice(core_start, core_stop)
            covered = np.arange(core_start, core_stop)
            inside = (covered >= origin) & (covered < stop)
            local = covered - origin
            votes[rows] += inside.astype(np.int16)
            safe_mass[rows] += (inside & (local >= RADIUS)
                                & (local <= patch - 1 - RADIUS)).astype(np.int16)
            safe_full[rows] += (inside & (local >= SUPPORT_LEFT)
                                & (local <= patch - 1 - SUPPORT_RIGHT)).astype(np.int16)
    return votes, safe_mass, safe_full


def inference_col_votes(patch):
    votes = np.zeros(TILE, dtype=np.int16)
    safe_mass = np.zeros(TILE, dtype=np.int16)
    safe_full = np.zeros(TILE, dtype=np.int16)
    for origin in _window_origins(TILE, patch, patch // 2):
        stop = min(origin + patch, TILE)
        covered = np.arange(TILE)
        inside = (covered >= origin) & (covered < stop)
        local = covered - origin
        votes += inside.astype(np.int16)
        safe_mass += (inside & (local >= RADIUS) & (local <= patch - 1 - RADIUS)).astype(np.int16)
        safe_full += (inside & (local >= SUPPORT_LEFT)
                      & (local <= patch - 1 - SUPPORT_RIGHT)).astype(np.int16)
    return votes, safe_mass, safe_full


def index_col_votes(patch, col_lo, col_hi):
    """Votes per column from split-contained validation windows in the index."""
    votes = np.zeros(TILE, dtype=np.int16)
    for origin in patch_origins(TILE, patch, patch // 2):
        if origin < col_lo or origin + patch > col_hi:
            continue
        votes[origin:origin + patch] += 1
    return votes


def index_row_votes(patch):
    votes = np.zeros(TILE, dtype=np.int16)
    for origin in patch_origins(TILE, patch, patch // 2):
        votes[origin:origin + patch] += 1
    return votes


def distribution(rows_pairs, cols_pairs, mask_rows=None, mask_cols=None):
    """Exact statistics over the 2-D product of per-axis (votes, safe) pairs."""
    total = 0.0
    weight_mean = 0.0
    buckets = {"ge_1_safe": 0.0, "ge_50": 0.0, "ge_90": 0.0, "all_safe": 0.0}
    fractions = []
    for (rv, rs), rc in rows_pairs:
        for (cv, cs), cc in cols_pairs:
            votes = rv * cv
            if votes == 0:
                continue
            safe = rs * cs
            weight = rc * cc
            fraction = safe / votes
            total += weight
            weight_mean += weight * fraction
            fractions.append((fraction, weight))
            if safe > 0:
                buckets["ge_1_safe"] += weight
            if fraction >= 0.5:
                buckets["ge_50"] += weight
            if fraction >= 0.9:
                buckets["ge_90"] += weight
            if safe == votes:
                buckets["all_safe"] += weight
    if total == 0:
        return {}
    fractions.sort()
    cumulative, median, p10 = 0.0, None, None
    for fraction, weight in fractions:
        cumulative += weight
        if p10 is None and cumulative >= 0.10 * total:
            p10 = fraction
        if median is None and cumulative >= 0.50 * total:
            median = fraction
    median = median if median is not None else fractions[-1][0]
    p10 = p10 if p10 is not None else fractions[0][0]
    return {
        "pixels": int(total),
        "mean_safe_fraction": round(weight_mean / total, 4),
        "median_safe_fraction": round(median, 4),
        "p10_safe_fraction": round(p10, 4),
        "pct_with_any_safe": round(100 * buckets["ge_1_safe"] / total, 2),
        "pct_at_least_50": round(100 * buckets["ge_50"] / total, 2),
        "pct_at_least_90": round(100 * buckets["ge_90"] / total, 2),
        "pct_all_safe": round(100 * buckets["all_safe"] / total, 2),
    }


def unique_pairs(votes, safe):
    pairs, counts = np.unique(np.stack([votes, safe], axis=1), axis=0, return_counts=True)
    return [(tuple(int(x) for x in pair), int(count)) for pair, count in zip(pairs, counts)]


def labelled_pixel_stats(pair, label_path, row_votes, row_safe, col_votes, col_safe):
    """Statistics restricted to reviewed validation pixels of one pair."""
    with rasterio.open(label_path) as dataset:
        labels = dataset.read(1)
    with rasterio.open(os.path.join(ROOT, "outputs", "partitions_v2_observation_valid",
                                    "pair_%s_partition_10m.tif" % pair)) as dataset:
        codes = dataset.read(1)
    reviewed = (labels != 255) & (codes == VALIDATION_CODE)
    rows, cols = np.nonzero(reviewed)
    if rows.size == 0:
        return {}
    votes = row_votes[rows].astype(np.int64) * col_votes[cols]
    safe = row_safe[rows].astype(np.int64) * col_safe[cols]
    fraction = safe / votes
    return {
        "reviewed_validation_pixels": int(rows.size),
        "mean_votes": round(float(votes.mean()), 3),
        "mean_safe_votes": round(float(safe.mean()), 3),
        "mean_safe_fraction": round(float(fraction.mean()), 4),
        "median_safe_fraction": round(float(np.median(fraction)), 4),
        "p10_safe_fraction": round(float(np.percentile(fraction, 10)), 4),
        "pct_with_any_safe": round(100 * float((safe > 0).mean()), 2),
        "pct_at_least_50": round(100 * float((fraction >= 0.5).mean()), 2),
        "pct_at_least_90": round(100 * float((fraction >= 0.9).mean()), 2),
        "pct_all_safe": round(100 * float((safe == votes).mean()), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-dir", default=os.path.join(ROOT, "outputs", "labels"))
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    report = {
        "tile": TILE, "block_rows": BLOCK_ROWS, "halo": HALO,
        "radius_px": RADIUS,
        "theoretical_full_support": {"left": SUPPORT_LEFT, "right": SUPPORT_RIGHT},
        "note": ("votes come from the real inference tiler including the vertical strip "
                 "halo; a vote is safe when its output location is >= radius from every "
                 "patch border on both axes"),
        "configurations": {},
    }

    for patch in (128, 256):
        stride = patch // 2
        vy, sy_mass, sy_full = inference_row_votes(patch)
        vx, sx_mass, sx_full = inference_col_votes(patch)

        row_pairs_mass = unique_pairs(vy, sy_mass)
        col_pairs_mass = unique_pairs(vx, sx_mass)
        row_pairs_full = unique_pairs(vy, sy_full)
        col_pairs_full = unique_pairs(vx, sx_full)

        entry = {
            "patch_size": patch, "stride": stride,
            "measured_radius": distribution(row_pairs_mass, col_pairs_mass),
            "theoretical_full": distribution(row_pairs_full, col_pairs_full),
            "votes_per_axis": {"row_votes_min": int(vy.min()), "row_votes_max": int(vy.max()),
                               "col_votes_min": int(vx.min()), "col_votes_max": int(vx.max())},
            "per_pixel": {},
            "validation_vs_inference": {},
        }

        for pair in ("A", "B"):
            label_path = os.path.join(args.labels_dir, "pair_%s_v2" % pair.lower(),
                                      "manual_labels_%s.tif" % pair)
            entry["per_pixel"][pair] = {
                "measured_radius": labelled_pixel_stats(pair, label_path, vy, sy_mass,
                                                        vx, sx_mass),
                "theoretical_full": labelled_pixel_stats(pair, label_path, vy, sy_full,
                                                         vx, sx_full),
            }
            # Validation index coverage of the same pixels.
            with rasterio.open(os.path.join(ROOT, "outputs", "partitions_v2_observation_valid",
                                            "pair_%s_partition_10m.tif" % pair)) as dataset:
                codes = dataset.read(1)
            stripe = np.flatnonzero(codes[0] == VALIDATION_CODE)
            col_lo, col_hi = int(stripe[0]), int(stripe[-1]) + 1
            ivx = index_col_votes(patch, col_lo, col_hi)
            ivy = index_row_votes(patch)
            with rasterio.open(label_path) as dataset:
                labels = dataset.read(1)
            reviewed = (labels != 255) & (codes == VALIDATION_CODE)
            rows, cols = np.nonzero(reviewed)
            inference_votes = vy[rows].astype(np.int64) * vx[cols]
            index_votes = ivy[rows].astype(np.int64) * ivx[cols]
            entry["validation_vs_inference"][pair] = {
                "reviewed_validation_pixels": int(rows.size),
                "mean_inference_votes": round(float(inference_votes.mean()), 3),
                "mean_index_votes": round(float(index_votes.mean()), 3),
                "mean_votes_missing_from_index": round(
                    float((inference_votes - index_votes).mean()), 3),
                "pct_of_inference_votes_in_index": round(
                    100 * float(index_votes.sum()) / max(float(inference_votes.sum()), 1), 2),
                "pct_pixels_with_fewer_index_votes": round(
                    100 * float((index_votes < inference_votes).mean()), 2),
            }
        report["configurations"]["%d_%d" % (patch, stride)] = entry

        a = entry["measured_radius"]
        b = entry["theoretical_full"]
        print("%d/%d: measured r%d -> %.1f%% of pixels get a safe vote, mean fraction %.3f, "
              "median %.3f, p10 %.3f, all-safe %.1f%%"
              % (patch, stride, RADIUS, a["pct_with_any_safe"], a["mean_safe_fraction"],
                 a["median_safe_fraction"], a["p10_safe_fraction"], a["pct_all_safe"]))
        print("        theoretical full support -> %.1f%% any safe, mean %.3f, all-safe %.1f%%"
              % (b["pct_with_any_safe"], b["mean_safe_fraction"], b["pct_all_safe"]))
        for pair in ("A", "B"):
            v = entry["validation_vs_inference"][pair]
            print("        pair %s: inference votes %.2f, index votes %.2f (%.1f%% retained)"
                  % (pair, v["mean_inference_votes"], v["mean_index_votes"],
                     v["pct_of_inference_votes_in_index"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
