"""List review candidates that still have no annotation.

Answers "what is left to review" from the same association logic the audit uses,
so the two cannot drift apart. A candidate counts as annotated when some record
in the GeoPackage has it as its strongest containment match, which is exactly the
rule `scripts/audit/audit_annotations_vs_candidates.py` applies.

Nothing is written except the CSV. All inputs are opened read-only.

    python scripts/review/list_unannotated_candidates.py --pair B \
        --labels data/manual_annotations_B.gpkg \
        --candidates outputs/labels/v2_observation_valid/review_candidates_B_p99_9.geojson \
        --panel-index outputs/figures/pair_b_candidate_evidence/candidate_panel_index.csv \
        --output outputs/labels/pair_b_annotation_audit_preliminary/unannotated_holdout_candidates.csv
"""

from __future__ import annotations
from pathlib import Path

import argparse
import csv
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.evaluation.matching import (CANDIDATE_COVERED_WEAK,
                                                    match_candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="B")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--panel-index", required=True)
    parser.add_argument("--partition-dir",
                        default=os.path.join(ROOT, "outputs", "partitions_v2_observation_valid"))
    parser.add_argument("--partitions", default=None,
                        help="defaults to pair_<PAIR>_partitions.geojson in --partition-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    import geopandas as gpd

    partitions_path = args.partitions or os.path.join(
        args.partition_dir, "pair_%s_partitions.geojson" % args.pair)
    annotations = gpd.read_file(args.labels, layer=args.layer)
    candidates = gpd.read_file(args.candidates).to_crs(annotations.crs)
    gpd.read_file(partitions_path)          # existence and readability check

    with open(args.panel_index, newline="") as handle:
        panels = list(csv.DictReader(handle))
    panel_partition = {row["candidate_id"]: row for row in panels}

    per_annotation = match_candidates(annotations, candidates)
    annotated = set()
    for _, scored in per_annotation.items():
        if not scored:
            continue
        best = max(scored, key=lambda item: item["candidate_covered"])
        if best["candidate_covered"] >= CANDIDATE_COVERED_WEAK:
            annotated.add(best["candidate_id"])

    known = {row["candidate_id"] for row in panels}
    missing_panel = annotated - known
    if missing_panel:
        print("WARNING: annotated candidates with no panel row: %s"
              % sorted(missing_panel))

    rows = []
    for row in panels:
        if row["candidate_id"] in annotated:
            continue
        panel_path = os.path.join(os.path.dirname(os.path.abspath(args.panel_index)),
                                  row["png"])
        rows.append({
            "candidate_id": row["candidate_id"],
            "partition": row["partition"],
            "partition_placement": row["partition_placement"],
            "area_px_20m": row["area_px_20m"],
            "mean_score": row["mean_score"],
            "panel_png": panel_path,
            "panel_json": os.path.join(os.path.dirname(panel_path), row["json"]),
        })

    order = {"test": 0, "validation": 1, "train": 2}
    rows.sort(key=lambda r: (order.get(r["partition"], 9), r["candidate_id"]))

    fields = ["candidate_id", "partition", "partition_placement", "area_px_20m",
              "mean_score", "panel_png", "panel_json"]
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    totals = {}
    for row in panels:
        totals[row["partition"]] = totals.get(row["partition"], 0) + 1
    remaining = {}
    for row in rows:
        remaining[row["partition"]] = remaining.get(row["partition"], 0) + 1

    print("candidates %d | annotated %d | unannotated %d"
          % (len(panels), len(annotated), len(rows)))
    for partition in sorted(totals):
        print("  %-11s %3d total | %3d annotated | %3d remaining"
              % (partition, totals[partition],
                 totals[partition] - remaining.get(partition, 0),
                 remaining.get(partition, 0)))
    print()
    for row in rows:
        print("  %-6s %-11s %8s px  mean %6.3f  %s"
              % (row["candidate_id"], row["partition"], row["area_px_20m"],
                 float(row["mean_score"]), row["panel_png"]))
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()
