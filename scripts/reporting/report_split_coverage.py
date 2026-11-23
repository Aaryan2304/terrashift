"""Per-split label coverage under the two selection policies.

Training and validation are selected differently on purpose. Training thresholds the labelled
fraction with split containment, drops empty windows and excludes test. Validation is
coverage-complete: every split-contained window holding a reviewed validation pixel, no threshold and
no balancing. Mixing them would either drop validation pixels in low-density windows or leak train
pixels into validation. Only the JSON is written.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.patch_sampling import (PairData, build_pool, patch_origins,
                                              select_plain)

SPLIT_CODES = {"train": 1, "validation": 2, "test": 3}


def split_columns(data, split):
    """First and last-plus-one column of a split stripe.

    Every split-contained window lives inside these columns, so all accounting is
    done on the stripe slice rather than the full tile. On this machine the full
    tile costs ~0.5 GiB per temporary array and the job was being OOM-killed.
    """
    code = SPLIT_CODES[split]
    columns = np.flatnonzero(data.column_code == code)
    if columns.size == 0:
        raise SystemExit("split %s has no columns" % split)
    return int(columns[0]), int(columns[-1]) + 1


def available(data, split):
    """Reviewed pixels available in a split, by class, counted on the stripe."""
    col0, col1 = split_columns(data, split)
    labels = data.label[:, col0:col1]
    change = int((labels == 1).sum())
    no_change = int((labels == 0).sum())
    return {"change_px": change, "no_change_px": no_change,
            "reviewed_px": change + no_change}


def covered(data, split, selection, size):
    """Unique and patch-instance pixels covered by a selection, inside one split."""
    windows = [w for w in selection if w["split"] == split]
    categories = {"positive_only": 0, "negative_only": 0, "mixed": 0, "empty": 0}
    for w in windows:
        categories[w["category"]] += 1
    if not windows:
        return {"windows": 0, "unique_change_px": 0, "unique_no_change_px": 0,
                "patch_instance_change_px": 0, "patch_instance_no_change_px": 0,
                "covered_px": 0, "multi_covered_fraction": 0.0,
                "categories": categories}
    col0, col1 = split_columns(data, split)
    width = col1 - col0
    counts = np.zeros((data.shape[0], width), dtype=np.uint8)
    for w in windows:
        left = w["col0"] - col0
        counts[w["row0"]:w["row0"] + size, left:left + size] += 1
    labels = data.label[:, col0:col1]
    covered_px = counts > 0
    reviewed = labels != 255
    result = {
        "windows": len(windows),
        "unique_change_px": int((covered_px & (labels == 1)).sum()),
        "unique_no_change_px": int((covered_px & (labels == 0)).sum()),
        "patch_instance_change_px": int(sum(w["change"] for w in windows)),
        "patch_instance_no_change_px": int(sum(w["no_change"] for w in windows)),
        "covered_px": int(covered_px.sum()),
        "multi_covered_fraction": round(
            float((counts > 1).sum()) / max(int(covered_px.sum()), 1), 4),
        "mean_multiplicity_on_reviewed": round(float(counts[reviewed].mean()), 3),
        "categories": categories,
    }
    del counts
    return result


def pct(part, whole):
    return round(100.0 * part / whole, 2) if whole else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir-a", required=True)
    parser.add_argument("--labels-dir-b", required=True)
    parser.add_argument("--partition-dir", default=os.path.join(
        ROOT, "outputs", "partitions_v2_observation_valid"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--min-fraction", type=float, default=0.002)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    size = args.size
    stride = size // 2
    report = {"patch_size": size, "stride": stride,
              "training_policy": "plain, min labelled fraction %.3f, empty and test excluded"
                                 % args.min_fraction,
              "validation_policy": ("coverage-complete: every split-contained window "
                                    "with >= 1 reviewed validation pixel, no threshold, "
                                    "no balancing"),
              "test_used": False, "pairs": {}}

    for pair, directory in (("A", args.labels_dir_a), ("B", args.labels_dir_b)):
        data = PairData(pair,
                        os.path.join(directory, "manual_labels_%s.tif" % pair),
                        os.path.join(args.partition_dir, "pair_%s_partition_10m.tif" % pair))
        pool = build_pool(data, size, stride)
        train_selection = [w for w in select_plain(pool, args.min_fraction)
                           if w["split"] == "train"]
        validation_selection = [w for w in pool
                                if w["split"] == "validation" and w["reviewed"] > 0]

        entry = {"available": {}, "training": {}, "validation": {}}
        for split in ("train", "validation"):
            entry["available"][split] = available(data, split)
        # The same threshold applied to validation, only to quantify what it
        # would cost. It is not the recommended policy.
        thresholded_validation = [w for w in select_plain(pool, args.min_fraction)
                                  if w["split"] == "validation"]

        entry["training"] = covered(data, "train", train_selection, size)
        entry["validation"] = covered(data, "validation", validation_selection, size)
        entry["validation_if_thresholded"] = covered(data, "validation",
                                                     thresholded_validation, size)
        # Percentages always use the denominator of the split the pixels live in.
        blocks = {
            "training": (entry["training"], entry["available"]["train"]),
            "validation": (entry["validation"], entry["available"]["validation"]),
            "validation_if_thresholded": (entry["validation_if_thresholded"],
                                          entry["available"]["validation"]),
        }
        for block, denominator in blocks.values():
            block["cover_change_pct"] = pct(block["unique_change_px"], denominator["change_px"])
            block["cover_no_change_pct"] = pct(block["unique_no_change_px"],
                                               denominator["no_change_px"])
        report["pairs"][pair] = entry
        del data, pool
        import gc
        gc.collect()
        print("pair %s: train windows %d, validation windows %d"
              % (pair, entry["training"]["windows"], entry["validation"]["windows"]))
        for split in ("training", "validation", "validation_if_thresholded"):
            a = entry["available"]["validation" if split.startswith("validation") else "train"]
            c = entry[split]
            print("  %-11s available change %6d no_change %6d | covered change %6d (%.1f%%) no_change %6d (%.1f%%)"
                  % (split, a["change_px"], a["no_change_px"], c["unique_change_px"],
                     c["cover_change_pct"], c["unique_no_change_px"], c["cover_no_change_pct"]))

    totals = {}
    for split, block in (("train", "training"), ("validation", "validation")):
        totals[split] = {
            "available_change_px": sum(report["pairs"][p]["available"][split]["change_px"]
                                       for p in report["pairs"]),
            "available_no_change_px": sum(report["pairs"][p]["available"][split]["no_change_px"]
                                          for p in report["pairs"]),
            "covered_unique_change_px": sum(report["pairs"][p][block]["unique_change_px"]
                                            for p in report["pairs"]),
            "covered_unique_no_change_px": sum(report["pairs"][p][block]["unique_no_change_px"]
                                               for p in report["pairs"]),
            "patch_instance_change_px": sum(report["pairs"][p][block]["patch_instance_change_px"]
                                            for p in report["pairs"]),
            "patch_instance_no_change_px": sum(report["pairs"][p][block]["patch_instance_no_change_px"]
                                               for p in report["pairs"]),
            "windows": sum(report["pairs"][p][block]["windows"] for p in report["pairs"]),
        }
        totals[split]["cover_change_pct"] = pct(totals[split]["covered_unique_change_px"],
                                                totals[split]["available_change_px"])
        totals[split]["cover_no_change_pct"] = pct(totals[split]["covered_unique_no_change_px"],
                                                   totals[split]["available_no_change_px"])
    report["totals"] = totals

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("\ntotals: train windows %d, validation windows %d"
          % (totals["train"]["windows"], totals["validation"]["windows"]))
    for split in ("train", "validation"):
        t = totals[split]
        print("  %-11s change %6d/%6d (%.1f%%) | no_change %6d/%6d (%.1f%%) | patch-instance change %d no_change %d"
              % (split, t["covered_unique_change_px"], t["available_change_px"],
                 t["cover_change_pct"], t["covered_unique_no_change_px"],
                 t["available_no_change_px"], t["cover_no_change_pct"],
                 t["patch_instance_change_px"], t["patch_instance_no_change_px"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
