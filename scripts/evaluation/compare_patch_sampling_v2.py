"""Corrected patch-sampling comparison for the local pairs.

The earlier rule called a window with no change pixels a negative. A window with no reviewed
pixels is empty, and supervision must not come from it. Categories: ``positive_only``,
``negative_only``, ``mixed``, ``empty``. Empty windows are counted and dropped, never sampled, and
no origin is duplicated to reach a target count.

Strategies are ``plain`` (threshold on reviewed fraction) and ``balanced`` (threshold, then pair
each positive window with an equal number of negative_only windows, largest first). Test windows
stay out of every pool and never choose a strategy.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os

import numpy as np
import rasterio
from terrashift.data.patch_sampling import (
    SPLITS,
    PairData,
    build_pool,
    patch_origins,
    select_plain,
)

ROOT = Path(__file__).resolve().parents[2]

SIZES = (256, 128)
THRESHOLDS = (0.001, 0.002, 0.005)


def _integral(mask):
    """Padded summed-area table of a boolean mask, int32."""
    height, width = mask.shape
    accumulator = np.zeros((height + 1, width + 1), dtype=np.int32)
    rows = mask.astype(np.int32)
    np.cumsum(rows, axis=0, out=rows)
    np.cumsum(rows, axis=1, out=rows)
    accumulator[1:, 1:] = rows
    del rows
    return accumulator


def _window_sum(table, row0, col0, size):
    return int(table[row0 + size, col0 + size] - table[row0, col0 + size]
               - table[row0 + size, col0] + table[row0, col0])


def category(window):
    if window["reviewed"] == 0:
        return "empty"
    if window["change"] > 0 and window["no_change"] > 0:
        return "mixed"
    if window["change"] > 0:
        return "positive_only"
    return "negative_only"


def select_balanced(pool, threshold):
    """Positive-containing windows paired 1:1 with negative_only windows."""
    usable = [w for w in pool if w["reviewed"] > 0
              and w["reviewed_fraction"] >= threshold]
    positives = sorted([w for w in usable if w["change"] > 0],
                       key=lambda w: (-w["reviewed"], w["pair"], w["row0"], w["col0"]))
    negatives = sorted([w for w in usable if w["change"] == 0],
                       key=lambda w: (-w["reviewed"], w["pair"], w["row0"], w["col0"]))
    take = min(len(positives), len(negatives))
    return positives[:take] + negatives[:take]


def _union_stats(pairs_data, selection, size):
    """Unique geographic pixels and coverage for a selection of windows."""
    per_pair = {}
    for pair, data in pairs_data.items():
        windows = [w for w in selection if w["pair"] == pair]
        if not windows:
            per_pair[pair] = {"windows": 0, "covered_px": 0, "multi_fraction": 0.0,
                              "unique_change_px": 0, "unique_no_change_px": 0}
            continue
        counts = np.zeros(data.shape, dtype=np.uint8)
        for w in windows:
            counts[w["row0"]:w["row0"] + size, w["col0"]:w["col0"] + size] += 1
        covered = counts > 0
        multi = float((counts > 1).sum()) / max(int(covered.sum()), 1)
        per_pair[pair] = {
            "windows": len(windows),
            "covered_px": int(covered.sum()),
            "multi_fraction": round(multi, 4),
            "unique_change_px": int((covered & data.change).sum()),
            "unique_no_change_px": int((covered & data.no_change).sum()),
            "mean_multiplicity_on_labelled": round(
                float(counts[data.reviewed].mean()), 3) if data.reviewed.any() else 0.0,
        }
        del counts
    return per_pair


def summarise(name, selection, pairs_data, size, pool, threshold=None):
    by_split = {split: sum(1 for w in selection if w["split"] == split)
                for split in ("train", "validation")}
    categories = {name_: sum(1 for w in selection if w["category"] == name_)
                  for name_ in ("positive_only", "negative_only", "mixed")}
    fractions = np.array([w["reviewed_fraction"] for w in selection]) if selection else np.zeros(1)
    union = _union_stats(pairs_data, selection, size)
    return {
        "strategy": name,
        "patch_size": size,
        "pool_size": len(pool),
        "unique_patches": len(selection),
        "per_split": by_split,
        "categories": categories,
        "empty_in_pool_never_selected": sum(1 for w in pool if w["category"] == "empty"),
        "below_threshold_not_selected": (0 if threshold is None else
                                          sum(1 for w in pool if w["category"] != "empty"
                                              and w["reviewed_fraction"] < threshold)),
        "patch_instance_change_px": int(sum(w["change"] for w in selection)),
        "patch_instance_no_change_px": int(sum(w["no_change"] for w in selection)),
        "unique_change_px": int(sum(v["unique_change_px"] for v in union.values())),
        "unique_no_change_px": int(sum(v["unique_no_change_px"] for v in union.values())),
        "labelled_fraction": {
            "min": round(float(fractions.min()), 5),
            "median": round(float(np.median(fractions)), 5),
            "mean": round(float(fractions.mean()), 5),
            "max": round(float(fractions.max()), 5),
        },
        "coverage": {
            "covered_px": int(sum(v["covered_px"] for v in union.values())),
            "multi_fraction": round(
                float(np.mean([v["multi_fraction"] for v in union.values()
                               if v["windows"]])) if any(v["windows"] for v in union.values()) else 0.0,
                4),
        },
        "per_pair": union,
    }


def validation_audit(data, size, stride):
    """Deterministic coverage-complete validation selection, and its overlap.

    Every split-contained window holding at least one reviewed validation pixel
    is selected: no balancing, no threshold. Pixels within ``size - 1`` of the
    validation stripe edge cannot be covered by any split-contained window,
    because every window containing them would reach into the neighbouring
    split. That band is measured rather than glossed over.
    """
    rows, cols = data.shape
    stripe = data.column_code == 2
    reviewed_validation = data.reviewed & stripe[None, :]
    windows = []
    for row0 in patch_origins(rows, size, stride):
        for col0 in patch_origins(cols, size, stride):
            if data.split_of(col0, size) != "validation":
                continue
            window = data.window(row0, col0, size)
            if window["reviewed"] > 0:
                windows.append(window)
    if not windows:
        return {"windows": 0}

    counts = np.zeros(data.shape, dtype=np.uint16)
    for w in windows:
        counts[w["row0"]:w["row0"] + size, w["col0"]:w["col0"] + size] += 1

    covered = counts > 0
    covered_reviewed = reviewed_validation & covered
    uncovered_reviewed = reviewed_validation & ~covered

    # Which columns are reachable at all by a split-contained window?
    reachable_columns = np.zeros(cols, dtype=bool)
    for col0 in range(0, cols - size + 1, stride):
        if data.split_of(col0, size) == "validation":
            reachable_columns[col0:col0 + size] = True

    multiplicity = counts[covered_reviewed]
    change_covered = covered_reviewed & data.change
    no_change_covered = covered_reviewed & data.no_change
    result = {
        "windows": len(windows),
        "reviewed_validation_px": int(reviewed_validation.sum()),
        "reviewed_validation_px_covered": int(covered_reviewed.sum()),
        "coverage_pct": round(100.0 * covered_reviewed.sum()
                              / max(int(reviewed_validation.sum()), 1), 4),
        "uncovered_px": int(uncovered_reviewed.sum()),
        "uncovered_px_in_unreachable_band": int(
            (uncovered_reviewed & ~reachable_columns[None, :]).sum()),
        "multiplicity_on_covered": {
            "mean": round(float(multiplicity.mean()), 3) if multiplicity.size else 0.0,
            "median": float(np.median(multiplicity)) if multiplicity.size else 0.0,
            "max": int(multiplicity.max()) if multiplicity.size else 0,
            "mean_change": (round(float(counts[change_covered].mean()), 3)
                            if change_covered.any() else 0.0),
            "mean_no_change": (round(float(counts[no_change_covered].mean()), 3)
                               if no_change_covered.any() else 0.0),
            "change_px": int(change_covered.sum()),
            "no_change_px": int(no_change_covered.sum()),
            "patch_instance_labelled_px": int(counts[covered_reviewed].sum()),
        },
    }
    del counts
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir-a", required=True)
    parser.add_argument("--labels-dir-b", required=True)
    parser.add_argument("--partition-dir", default=os.path.join(
        ROOT, "outputs", "partitions_v2_observation_valid"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    sources = {}
    for pair, directory in (("A", args.labels_dir_a), ("B", args.labels_dir_b)):
        sources[pair] = PairData(
            pair,
            os.path.join(directory, "manual_labels_%s.tif" % pair),
            os.path.join(args.partition_dir, "pair_%s_partition_10m.tif" % pair))

    report = {"strategy_definitions": {
        "positive_only": "change>0 and no_change==0",
        "negative_only": "no_change>0 and change==0",
        "mixed": "change>0 and no_change>0",
        "empty": "no reviewed pixels; never selected",
    }, "thresholds": list(THRESHOLDS), "patch_sizes": list(SIZES),
        "test_windows_used": False, "results": {}, "validation_audit": {}}

    for pair, data in sources.items():
        for size in SIZES:
            stride = size // 2
            pool = build_pool(data, size, stride)
            categories = {name: sum(1 for w in pool if w["category"] == name)
                          for name in ("positive_only", "negative_only", "mixed", "empty")}
            print("pair %s size %d: pool %d %s" % (pair, size, len(pool), categories))
            for threshold in THRESHOLDS:
                key = "%s_%d_plain_%.3f" % (pair, size, threshold)
                report["results"][key] = summarise(
                    key, select_plain(pool, threshold), {pair: data}, size, pool, threshold)
                key = "%s_%d_balanced_%.3f" % (pair, size, threshold)
                report["results"][key] = summarise(
                    key, select_balanced(pool, threshold), {pair: data}, size, pool, threshold)
            report["validation_audit"]["%s_%d" % (pair, size)] = validation_audit(data, size, stride)

    # pooled: one pool over both pairs, balanced across the pool rather than per pair
    for size in SIZES:
        stride = size // 2
        pooled = []
        for data in sources.values():
            pooled.extend(build_pool(data, size, stride))
        print("pooled size %d: pool %d" % (size, len(pooled)))
        for threshold in THRESHOLDS:
            key = "pooled_%d_plain_%.3f" % (size, threshold)
            report["results"][key] = summarise(
                key, select_plain(pooled, threshold), sources, size, pooled, threshold)
            key = "pooled_%d_balanced_%.3f" % (size, threshold)
            report["results"][key] = summarise(
                key, select_balanced(pooled, threshold), sources, size, pooled, threshold)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("\nwrote", args.output)

    header = ("%-26s %6s %5s %5s | %4s %4s %4s | %8s %8s | %7s %7s | %6s"
              % ("strategy", "patch", "trn", "val", "pos", "neg", "mix",
                 "unique_c", "inst_c", "uniq_n", "inst_n", "med.fr"))
    print("\n" + header)
    for key, result in report["results"].items():
        print("%-26s %6d %5d %5d | %4d %4d %4d | %8d %8d | %7d %7d | %6.4f"
              % (key, result["unique_patches"], result["per_split"]["train"],
                 result["per_split"]["validation"], result["categories"]["positive_only"],
                 result["categories"]["negative_only"], result["categories"]["mixed"],
                 result["unique_change_px"], result["patch_instance_change_px"],
                 result["unique_no_change_px"], result["patch_instance_no_change_px"],
                 result["labelled_fraction"]["median"]))


if __name__ == "__main__":
    main()
