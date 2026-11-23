"""Read-only audit of a manual-annotation GeoPackage against candidates.

Reports how polygons distribute over the partitions, whether any crosses a boundary, lands on
invalid pixels or falls outside the imagery extent, plus geometry and attribute validity and the
candidate each annotation belongs to.

Matching is strict: a confident match needs IoU >= 0.5 and mutual best agreement. The rest is
reported as ambiguous or unmatched rather than guessed, and the record id is compared only after
the match.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import csv
import json
import os
import re

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import box
from terrashift.evaluation.matching import (
    CANDIDATE_COVERED_CONFIDENT,
    CANDIDATE_COVERED_WEAK,
    match_candidates,
    overlap_area,
)

ROOT = Path(__file__).resolve().parents[2]

LABEL_CLASSES = ("change", "no_change", "uncertain")
CONFIDENCES = ("high", "medium", "low")
EVIDENCE_SOURCES = ("before_after_visual", "field_verified", "other_manual")
REQUIRED = ("record_id", "pair", "label_class", "reviewer", "reviewed_at",
            "evidence_source", "evidence", "confidence")

# Association is judged by how much of the *candidate* the annotation covers.
# The candidates come from the score tail and are small; the reviewer draws the
# footprint of the feature, which is usually larger. Judging by IoU alone marks
# a correct association as weak purely because the annotation is generous: a
# candidate of 4,800 m2 inside an annotation of 17,317 m2 has an IoU near 0.28
# while being 99 % covered. IoU is still reported, but it decides nothing.


def load_layers(args):
    import geopandas as gpd

    annotations = gpd.read_file(args.labels, layer=args.layer)
    candidates = gpd.read_file(args.candidates)
    partitions = gpd.read_file(args.partitions)
    if annotations.crs is None:
        raise SystemExit("annotations have no CRS")
    target = annotations.crs
    candidates = candidates.to_crs(target)
    partitions = partitions.to_crs(target)
    return annotations, candidates, partitions


def partition_geometry(partitions):
    """Split polygons plus the guard bands between them, in the label CRS."""
    pieces = {row["split"]: row.geometry for _, row in partitions.iterrows()}
    order = sorted(pieces, key=lambda name: pieces[name].bounds[0])
    guards = []
    for left, right in zip(order, order[1:]):
        gap_start = pieces[left].bounds[2]
        gap_stop = pieces[right].bounds[0]
        if gap_stop > gap_start:
            low = min(pieces[left].bounds[1], pieces[right].bounds[1])
            high = max(pieces[left].bounds[3], pieces[right].bounds[3])
            guards.append((box(gap_start, low, gap_stop, high), "%s|%s" % (left, right)))
    return pieces, order, guards


def composition(geometry, pieces, guards):
    """Area of a polygon in each split and guard band, plus area outside."""
    parts = {}
    total = float(geometry.area)
    covered = 0.0
    for name, piece in pieces.items():
        overlap = geometry.intersection(piece).area
        if overlap > 0:
            parts[name] = float(overlap)
            covered += float(overlap)
    for guard, label in guards:
        overlap = geometry.intersection(guard).area
        if overlap > 0:
            parts["guard(%s)" % label] = float(overlap)
            covered += float(overlap)
    parts["outside_partitions"] = max(0.0, total - covered)
    return parts


def invalid_pixels(geometry, dataset, validity):
    """Count invalid observation pixels under a polygon, on the validity grid."""
    window = from_bounds(*geometry.bounds, transform=dataset.transform)
    row0 = max(0, int(np.floor(window.row_off)))
    col0 = max(0, int(np.floor(window.col_off)))
    row1 = min(dataset.height, int(np.ceil(window.row_off + window.height)))
    col1 = min(dataset.width, int(np.ceil(window.col_off + window.width)))
    if row1 <= row0 or col1 <= col0:
        return 0, 0
    local = rasterio.windows.transform(
        rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0), dataset.transform)
    burn = rasterize([(geometry.__geo_interface__, 1)],
                     out_shape=(row1 - row0, col1 - col0), transform=local,
                     fill=0, dtype="uint8").astype(bool)
    inside = validity[row0:row1, col0:col1][burn]
    return int(inside.size), int((~inside).sum())


def attribute_errors(row, pair):
    """Every rule from the label schema that this record breaks."""
    errors = []
    for field in REQUIRED:
        value = row.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)) \
                or (isinstance(value, str) and not value.strip()):
            errors.append("%s missing" % field)
    record_id = str(row.get("record_id", ""))
    if record_id and not re.fullmatch(r"%s_\d+" % pair, record_id):
        errors.append("record_id %r does not match %s_<n>" % (record_id, pair))
    if row.get("pair") not in (None, pair):
        errors.append("pair is %r, expected %r" % (row.get("pair"), pair))
    if row.get("label_class") not in LABEL_CLASSES:
        errors.append("label_class %r not in %s" % (row.get("label_class"), LABEL_CLASSES))
    if row.get("confidence") not in CONFIDENCES:
        errors.append("confidence %r not in %s" % (row.get("confidence"), CONFIDENCES))
    if row.get("evidence_source") not in EVIDENCE_SOURCES:
        errors.append("evidence_source %r not in %s"
                      % (row.get("evidence_source"), EVIDENCE_SOURCES))
    return errors


def summarise_matches(annotations, per_annotation):
    """Classify every annotation as one-to-one, ambiguous or unmatched.

    A one-to-one association needs exactly one candidate to sit substantially
    inside the annotation. Two or more means the annotation spans separate score
    blobs, which is a real ambiguity and is reported rather than resolved by
    picking the largest. A single candidate that is only partly covered is
    reported separately: the pairing is still unique, but the polygons do not
    nest, and that is worth a look before the final QA.
    """
    rows = []
    for _, row in annotations.iterrows():
        record_id = row["record_id"]
        scored = per_annotation.get(record_id, [])
        entry = {"record_id": record_id, "label_class": row["label_class"],
                 "candidate_id": "", "candidate_covered": "", "annotation_covered": "",
                 "iou": "", "status": "", "candidates_overlapping": len(scored),
                 "id_agrees": ""}
        if not scored:
            entry["status"] = "unmatched_no_overlap"
            rows.append(entry)
            continue

        scored.sort(key=lambda item: -item["candidate_covered"])
        strong = [item for item in scored
                  if item["candidate_covered"] >= CANDIDATE_COVERED_CONFIDENT]
        top = scored[0]
        entry["candidate_id"] = top["candidate_id"]
        entry["candidate_covered"] = round(top["candidate_covered"], 4)
        entry["annotation_covered"] = round(top["annotation_covered"], 4)
        entry["iou"] = round(top["iou"], 4)

        if top["candidate_covered"] < CANDIDATE_COVERED_WEAK:
            entry["status"] = "unmatched_below_threshold"
        elif len(strong) > 1:
            entry["status"] = "ambiguous_multiple_candidates"
        elif len(strong) == 1:
            entry["status"] = "confident_one_to_one"
        else:
            entry["status"] = "one_to_one_partial_coverage"
        entry["id_agrees"] = str(entry["candidate_id"] == record_id)
        rows.append(entry)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="B")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--partitions", required=True)
    parser.add_argument("--validity", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    per_record_csv = os.path.join(args.output_dir, "per_record.csv")
    matches_csv = os.path.join(args.output_dir, "candidate_matches.csv")
    summary_json = os.path.join(args.output_dir, "summary.json")
    for path in (per_record_csv, matches_csv, summary_json):
        if os.path.exists(path):
            raise SystemExit("refusing to overwrite %s" % path)

    annotations, candidates, partitions = load_layers(args)
    pieces, order, guards = partition_geometry(partitions)
    print("annotations %d | candidates %d | partitions %s | guards %d"
          % (len(annotations), len(candidates), order, len(guards)))
    for name in order:
        low, _, high, _ = pieces[name].bounds
        print("  %-11s x %10.1f .. %10.1f" % (name, low, high))
    for guard, label in guards:
        low, _, high, _ = guard.bounds
        print("  guard %-14s x %10.1f .. %10.1f" % (label, low, high))

    with rasterio.open(args.validity) as dataset:
        validity = dataset.read(1).astype(bool)
        raster_bounds = dataset.bounds
        raster_crs = dataset.crs
        if str(raster_crs) != str(annotations.crs):
            raise SystemExit("validity raster CRS %s does not match annotations %s"
                             % (raster_crs, annotations.crs))

        tile = box(raster_bounds.left, raster_bounds.bottom,
                   raster_bounds.right, raster_bounds.top)

        records = []
        for _, row in annotations.iterrows():
            geometry = row.geometry
            errors = attribute_errors(row, args.pair)
            parts = composition(geometry, pieces, guards)
            total = float(geometry.area)
            dominant = max((name for name in parts if not name.startswith("guard")),
                           key=lambda name: parts[name], default="none")
            guard_area = sum(value for name, value in parts.items()
                             if name.startswith("guard"))
            outside = parts.get("outside_partitions", 0.0)
            covered_px, invalid_px = invalid_pixels(geometry, dataset, validity)
            record = {
                "record_id": row["record_id"],
                "label_class": row["label_class"],
                "confidence": row["confidence"],
                "geometry_valid": bool(geometry.is_valid),
                "geometry_type": geometry.geom_type,
                "vertices": int(len(geometry.exterior.coords)),
                "area_m2": round(total, 1),
                "split": dominant,
                "split_area_m2": round(parts.get(dominant, 0.0), 1),
                "split_area_fraction": round(parts.get(dominant, 0.0) / total, 4) if total else None,
                "guard_area_m2": round(guard_area, 1),
                "outside_partition_m2": round(outside, 1),
                "crosses_partition": int(len([n for n in parts
                                              if n in pieces and parts[n] > 0]) > 1),
                "touches_guard": int(guard_area > 0),
                "pixels_20m": covered_px,
                "invalid_pixels_20m": invalid_px,
                "outside_image_extent": int(not tile.covers(geometry)),
                "attribute_errors": "; ".join(errors),
            }
            records.append(record)

    per_annotation = match_candidates(annotations, candidates)
    match_rows = summarise_matches(annotations, per_annotation)

    os.makedirs(args.output_dir, exist_ok=True)
    record_fields = list(records[0])
    with open(per_record_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=record_fields)
        writer.writeheader()
        writer.writerows(records)

    match_fields = ["record_id", "label_class", "candidate_id", "candidate_covered",
                    "annotation_covered", "iou", "status", "candidates_overlapping",
                    "id_agrees"]
    with open(matches_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=match_fields)
        writer.writeheader()
        writer.writerows(match_rows)

    def tally(key, predicate=lambda r: True):
        counts = {}
        for record in records:
            if not predicate(record):
                continue
            bucket = record[key]
            counts[bucket] = counts.get(bucket, 0) + 1
        return dict(sorted(counts.items()))

    by_split_class = {}
    for record in records:
        bucket = by_split_class.setdefault(record["split"], {})
        bucket[record["label_class"]] = bucket.get(record["label_class"], 0) + 1
    by_split_confidence = {}
    for record in records:
        bucket = by_split_confidence.setdefault(record["split"], {})
        bucket[record["confidence"]] = bucket.get(record["confidence"], 0) + 1

    status_counts = {}
    for match in match_rows:
        status_counts[match["status"]] = status_counts.get(match["status"], 0) + 1

    matched = [m for m in match_rows if m["candidate_id"]]
    id_agree = [m for m in matched if m["id_agrees"] == "True"]
    summary = {
        "pair": args.pair,
        "status": "PRELIMINARY_INTERIM_ANNOTATIONS_NOT_FINAL",
        "inputs": {"labels": os.path.abspath(args.labels),
                   "candidates": os.path.abspath(args.candidates),
                   "partitions": os.path.abspath(args.partitions),
                   "validity": os.path.abspath(args.validity)},
        "counts": {"features": len(records),
                   "by_class": tally("label_class"),
                   "by_confidence": tally("confidence"),
                   "by_split": tally("split"),
                   "by_split_class": by_split_class,
                   "by_split_confidence": by_split_confidence},
        "partition_checks": {
            "crossing_more_than_one_partition": sum(r["crosses_partition"] for r in records),
            "touching_guard_band": sum(r["touches_guard"] for r in records),
            "outside_image_extent": sum(r["outside_image_extent"] for r in records),
            "records_touching_guard": [r["record_id"] for r in records if r["touches_guard"]],
            "records_crossing": [r["record_id"] for r in records if r["crosses_partition"]],
        },
        "validity_checks": {
            "records_with_invalid_pixels": sum(1 for r in records
                                               if r["invalid_pixels_20m"] > 0),
            # Inside annotated polygons only. The tile-wide invalid count is a
            # different number: see docs/pair_b_validity_audit.md.
            "total_invalid_pixels_under_annotations": sum(r["invalid_pixels_20m"]
                                                          for r in records),
            "records": [{"record_id": r["record_id"],
                         "invalid_pixels_20m": r["invalid_pixels_20m"],
                         "pixels_20m": r["pixels_20m"]}
                        for r in records if r["invalid_pixels_20m"] > 0],
        },
        "geometry_checks": {
            "invalid_geometry": sum(1 for r in records if not r["geometry_valid"]),
            "attribute_failures": sum(1 for r in records if r["attribute_errors"]),
            "failures": {r["record_id"]: r["attribute_errors"] for r in records
                         if r["attribute_errors"]},
            "non_polygon": sum(1 for r in records if r["geometry_type"] != "Polygon"),
        },
        "candidate_matching": {
            "method": ("association by the share of the candidate inside the "
                       "annotation; IoU is reported but does not decide, because a "
                       "generous annotation around a small score blob has a low "
                       "IoU while containing the candidate entirely"),
            "confident_candidate_covered": CANDIDATE_COVERED_CONFIDENT,
            "weak_candidate_covered": CANDIDATE_COVERED_WEAK,
            "status_counts": status_counts,
            "with_candidate": len(matched),
            "annotations_overlapping_more_than_one_candidate":
                sum(1 for m in match_rows if int(m["candidates_overlapping"] or 0) > 1),
            "iou_range": [min(float(m["iou"]) for m in matched),
                          max(float(m["iou"]) for m in matched)] if matched else None,
            "id_convention_agrees": len(id_agree),
            "id_convention_disagrees": [{"record_id": m["record_id"],
                                         "candidate_id": m["candidate_id"],
                                         "candidate_covered": m["candidate_covered"]}
                                        for m in matched if m["id_agrees"] != "True"][:20],
        },
    }
    with open(summary_json, "w") as handle:
        json.dump(summary, handle, indent=1)

    print("\nclass by split:", json.dumps(by_split_class))
    print("confidence by split:", json.dumps(by_split_confidence))
    print("crossing: %d | guard: %d | outside extent: %d | invalid-pixel records: %d"
          % (summary["partition_checks"]["crossing_more_than_one_partition"],
             summary["partition_checks"]["touching_guard_band"],
             summary["partition_checks"]["outside_image_extent"],
             summary["validity_checks"]["records_with_invalid_pixels"]))
    print("matches:", json.dumps(status_counts))
    print("id convention agrees %d of %d matched" % (len(id_agree), len(matched)))
    print("\nwrote", per_record_csv)
    print("wrote", matches_csv)
    print("wrote", summary_json)


if __name__ == "__main__":
    main()
