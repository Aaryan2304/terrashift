"""Build the record-to-candidate crosswalk for one pair.

One row per annotation, including records that match no candidate. Matching is geometric and the
record id is only compared afterwards, so a naming convention is checked instead of assumed.
Candidates that no annotation covers go to a second file, which is the review backlog. Inputs are
read-only and nothing is overwritten.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import csv
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.evaluation.matching import (CANDIDATE_COVERED_CONFIDENT,
                                                    CANDIDATE_COVERED_WEAK,
                                                    match_candidates)


def partition_lookup(partitions):
    """Split name to geometry, for classifying an annotation's location."""
    return {row["split"]: row.geometry for _, row in partitions.iterrows()}


def partition_of(geometry, pieces):
    """Name of the split holding most of a geometry, or a stated fallback."""
    overlaps = {name: geometry.intersection(piece).area
                for name, piece in pieces.items()}
    overlaps = {name: area for name, area in overlaps.items() if area > 0}
    return max(overlaps, key=overlaps.get) if overlaps else "outside_partitions"


def build_rows(annotations, candidates, pieces):
    """Crosswalk rows for every annotation, matched by geometry only."""
    per_annotation = match_candidates(annotations, candidates)

    rows = []
    for _, annotation in annotations.iterrows():
        record_id = annotation["record_id"]
        scored = per_annotation.get(record_id, [])
        top = None
        if scored:
            scored = sorted(scored, key=lambda item: -item["candidate_covered"])
            top = scored[0]
        row = {
            "annotation_record_id": record_id,
            "candidate_id": "",
            "pair": annotation["pair"],
            "partition": partition_of(annotation.geometry, pieces),
            "label_class": annotation["label_class"],
            "match_type": "unmatched_no_overlap",
            "association_metric": "candidate_covered",
            "association_value": 0.0,
            "candidate_covered": 0.0,
            "annotation_covered": 0.0,
            "iou": 0.0,
            "candidate_shared_by": 0,
        }
        if top is not None:
            row.update({
                "candidate_id": top["candidate_id"],
                "candidate_covered": round(top["candidate_covered"], 4),
                "annotation_covered": round(top["annotation_covered"], 4),
                "iou": round(top["iou"], 4),
            })
            if top["candidate_covered"] >= CANDIDATE_COVERED_CONFIDENT:
                row["match_type"] = "confident_one_to_one"
                row["association_value"] = row["candidate_covered"]
            elif top["candidate_covered"] >= CANDIDATE_COVERED_WEAK:
                row["match_type"] = "one_to_one_partial_coverage"
                row["association_value"] = row["candidate_covered"]
            else:
                row["match_type"] = "unmatched_no_overlap"
                row["association_metric"] = "iou"
                row["association_value"] = row["iou"]
        rows.append(row)

    # A candidate claimed by more than one annotation is allowed but must be
    # visible: two reviewers can legitimately split one blob into two features.
    usage = {}
    for row in rows:
        if row["candidate_id"]:
            usage[row["candidate_id"]] = usage.get(row["candidate_id"], 0) + 1
    for row in rows:
        if row["candidate_id"]:
            row["candidate_shared_by"] = usage[row["candidate_id"]]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="B")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--partitions", required=True)
    parser.add_argument("--panel-index", default=None,
                        help="optional; used to point each unreviewed candidate "
                             "at its evidence panel")
    parser.add_argument("--mask", default=None,
                        help="optional label mask; when given, the value under each "
                             "unreviewed candidate is measured rather than assumed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--unreviewed-output", default=None)
    args = parser.parse_args()

    for path in (args.output, args.unreviewed_output):
        if path and os.path.exists(path):
            raise SystemExit("refusing to overwrite %s" % path)

    import geopandas as gpd

    annotations = gpd.read_file(args.labels, layer=args.layer)
    candidates = gpd.read_file(args.candidates).to_crs(annotations.crs)
    partitions = gpd.read_file(args.partitions).to_crs(annotations.crs)
    pieces = partition_lookup(partitions)

    rows = build_rows(annotations, candidates, pieces)

    fields = ["annotation_record_id", "candidate_id", "pair", "partition",
              "label_class", "match_type", "association_metric",
              "association_value", "candidate_covered", "annotation_covered", "iou",
              "candidate_shared_by"]
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    shared = {row["candidate_id"]: row["candidate_shared_by"] for row in rows
              if row["candidate_shared_by"] > 1}
    if shared:
        print("WARNING: %d candidate(s) claimed by more than one annotation: %s"
              % (len(shared), shared))

    matched = {row["candidate_id"] for row in rows if row["candidate_id"]}
    counts = {}
    for row in rows:
        counts[row["match_type"]] = counts.get(row["match_type"], 0) + 1
    print("crosswalk rows: %d | matched candidates: %d of %d"
          % (len(rows), len(matched), len(candidates)))
    print("match types:", counts)

    if args.unreviewed_output:
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
        from rasterio.windows import from_bounds

        panels = {}
        if args.panel_index and os.path.exists(args.panel_index):
            panels = {row["candidate_id"]: row
                      for row in csv.DictReader(open(args.panel_index))}

        mask_dataset = rasterio.open(args.mask) if args.mask else None
        unreviewed = []
        for _, candidate in candidates.iterrows():
            candidate_id = candidate["candidate_id"]
            if candidate_id in matched:
                continue
            panel = panels.get(candidate_id, {})
            observed = "not_measured"
            if mask_dataset is not None:
                geometry = candidate.geometry
                dataset = mask_dataset
                window = from_bounds(*geometry.bounds, transform=dataset.transform)
                row0 = max(0, int(np.floor(window.row_off)))
                col0 = max(0, int(np.floor(window.col_off)))
                row1 = min(dataset.height, int(np.ceil(window.row_off + window.height)))
                col1 = min(dataset.width, int(np.ceil(window.col_off + window.width)))
                local = rasterio.windows.transform(
                    rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0),
                    dataset.transform)
                burn = rasterize([(geometry.__geo_interface__, 1)],
                                 out_shape=(row1 - row0, col1 - col0), transform=local,
                                 fill=0, dtype="uint8").astype(bool)
                crop = dataset.read(1, window=rasterio.windows.Window(
                    col0, row0, col1 - col0, row1 - row0))
                values = sorted({int(v) for v in np.unique(crop[burn])})
                observed = "|".join(str(v) for v in values)
            unreviewed.append({
                "candidate_id": candidate_id,
                "partition": panel.get("partition", partition_of(candidate.geometry, pieces)),
                "area_px_20m": candidate.get("area_px_20m"),
                "mean_score": candidate.get("mean_score"),
                "mask_value_under_candidate": observed,
                "usable_as_negative": "no" if observed in ("255", "not_measured") else "see report",
                "panel_png": panel.get("png", ""),
            })
        if mask_dataset is not None:
            mask_dataset.close()
        unreviewed.sort(key=lambda item: item["candidate_id"])
        with open(args.unreviewed_output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(unreviewed[0]))
            writer.writeheader()
            writer.writerows(unreviewed)
        by_partition = {}
        for item in unreviewed:
            by_partition[item["partition"]] = by_partition.get(item["partition"], 0) + 1
        print("unreviewed candidates: %d | by partition: %s"
              % (len(unreviewed), by_partition))
        print("wrote", args.unreviewed_output)

    print("wrote", args.output)


if __name__ == "__main__":
    main()
