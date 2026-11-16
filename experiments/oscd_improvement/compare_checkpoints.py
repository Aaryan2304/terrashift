"""Tabulate the experiment results into one comparison table.

Reads the per-checkpoint analysis outputs rather than recomputing anything, so
the numbers in the report are the numbers on disk.

    python experiments/oscd_improvement/compare_checkpoints.py \
        --root experiments/oscd_improvement \
        --output experiments/oscd_improvement/results/comparison.csv
"""

import argparse
import csv
import json
import os

FIXED_THRESHOLD = "0.90"      # chosen on validation from the baseline sweep


def read_threshold_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_postprocess(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_dir = os.path.join(args.root, "results")
    checkpoints_dir = os.path.join(args.root, "checkpoints")

    runs = []
    for name in sorted(os.listdir(results_dir)):
        directory = os.path.join(results_dir, name)
        analysis = os.path.join(directory, "analysis.json")
        thresholds = os.path.join(directory, "thresholds.csv")
        if not (os.path.isfile(analysis) and os.path.isfile(thresholds)):
            continue
        with open(analysis) as handle:
            summary = json.load(handle)
        rows = read_threshold_rows(thresholds)
        fixed = next((row for row in rows if row["threshold"] == FIXED_THRESHOLD), None)
        best = next(row for row in rows
                    if row["threshold"] == str(summary["best_pixel_f1_threshold"]))
        post = read_postprocess(os.path.join(directory, "postprocess.csv"))
        best_post = max(post, key=lambda row: float(row["object_f1"]))
        runs.append({
            "run": name,
            "ap_scene": round(float(summary["average_precision"]), 4),
            "auc": round(float(summary["roc_auc"]), 4),
            "best_threshold": summary["best_pixel_f1_threshold"],
            "pixel_f1_at_best": best["f1"],
            "pixel_precision_at_best": best["precision"],
            "pixel_recall_at_best": best["recall"],
            "object_f1_at_best": best["object_f1"],
            "pixel_f1_at_0.90": fixed["f1"] if fixed else "",
            "object_f1_at_0.90": fixed["object_f1"] if fixed else "",
            "object_fp_at_0.90": fixed["object_fp"] if fixed else "",
            "best_post": "open %s close %s min %s"
                         % (best_post["open"], best_post["close"],
                            best_post["min_area_px"]),
            "post_pixel_f1": best_post["f1"],
            "post_object_f1": best_post["object_f1"],
            "post_object_fp": best_post["object_fp"],
        })

    if not runs:
        raise SystemExit("no results found under %s" % results_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)

    header = ("%-28s %7s %7s %6s %7s %7s %8s %8s" %
              ("run", "AP", "AUC", "thr", "pixF1", "objF1", "postF1", "postObj"))
    print(header)
    print("-" * len(header))
    for run in runs:
        print("%-28s %7.4f %7.4f %6s %7.3f %7.4f %8.3f %8.4f"
              % (run["run"], run["ap_scene"], run["auc"], run["best_threshold"],
                 float(run["pixel_f1_at_best"]), float(run["object_f1_at_best"]),
                 float(run["post_pixel_f1"]), float(run["post_object_f1"])))
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()
