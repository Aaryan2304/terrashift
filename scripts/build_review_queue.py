"""Build a QGIS review queue from the candidate layers and the partitions.

Output is one CSV row per candidate, ordered by pair and partition, with a
review order inside each group. No decision is made here: the human fields are
written empty and every row starts as ``unreviewed``.

The partition assignment is read from the 20 m partition raster, window by
window, so a candidate that spans more than one split is detected rather than
guessed from its centroid. Codes follow ``scripts/prepare_local_partitions.py``:
0 is the 500 m guard band, 1 train, 2 validation, 3 test.

Usage:

    python scripts/build_review_queue.py \
        --output-dir review/manual_review_v1
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
import shapely
from shapely.geometry import shape as shapely_shape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CANDIDATE_DIR = os.path.join(ROOT, "outputs", "labels", "v2_observation_valid")
PARTITION_DIR = os.path.join(ROOT, "outputs", "partitions_v2_observation_valid")
DEFAULT_OUT = os.path.join(ROOT, "review", "manual_review_v1")

SPLIT_CODES = {1: "train", 2: "validation", 3: "test"}
GUARD_CODE = 0
PIXEL_AREA_M2 = 20.0 * 20.0
PARTITION_ORDER = {"train": 0, "validation": 1, "test": 2, "guard": 3, "mixed": 4}

FIELDS = [
    "candidate_id", "pair", "component_id", "partition", "n_partitions",
    "partition_codes", "crosses_partition_boundary", "touches_guard_band",
    "guard_fraction", "bbox_width_px_20m", "bbox_height_px_20m",
    "bbox_width_m", "bbox_height_m", "bbox_area_m2",
    "component_area_px_20m", "component_area_m2",
    "mean_score", "max_score", "score_threshold", "priority_percentile",
    "centroid_lon", "centroid_lat", "centroid_easting", "centroid_northing",
    "review_order", "review_status", "human_decision", "evidence", "confidence",
    "note",
]


def partition_window(dataset, bounds):
    """The raster window covering a geometry's bounds, clipped to the raster."""
    window = from_bounds(*bounds, transform=dataset.transform)
    window = window.round_offsets().round_lengths()
    col0 = max(0, int(window.col_off))
    row0 = max(0, int(window.row_off))
    col1 = min(dataset.width, int(window.col_off + window.width))
    row1 = min(dataset.height, int(window.row_off + window.height))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def classify_partition(codes, total):
    """Partition label and flags from the codes under one polygon.

    Returns ``(partition, crosses_boundary, touches_guard, guard_fraction)``.
    ``mixed`` means the polygon covers pixels from more than one of train,
    validation and test, which the rasterizer refuses.
    """
    present_splits = sorted({SPLIT_CODES[c] for c in codes if c in SPLIT_CODES})
    guard_pixels = int((codes == GUARD_CODE).sum())
    guard_fraction = guard_pixels / total if total else 0.0
    touches_guard = guard_pixels > 0
    crosses = len(present_splits) > 1

    if not present_splits:
        return "guard", False, touches_guard, guard_fraction
    if crosses:
        return "mixed", True, touches_guard, guard_fraction
    return present_splits[0], False, touches_guard, guard_fraction


def candidate_rows(pair, candidates, partition_path):
    """Rows for one pair, without the review order (assigned later)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    rows = []
    with rasterio.open(partition_path) as dataset:
        for feature in candidates["features"]:
            props = feature["properties"]
            geometry = shapely_shape(feature["geometry"])
            geometry_m = shapely.transform(
                geometry,
                lambda coords: np.column_stack(transformer.transform(coords[:, 0],
                                                                     coords[:, 1])))

            window = partition_window(dataset, geometry_m.bounds)
            if window is None:
                partition, crosses, touches_guard, guard_fraction = (
                    "outside", False, False, 0.0)
                codes = np.zeros(0, dtype=np.uint8)
            else:
                data = dataset.read(1, window=window)
                transform = dataset.window_transform(window)
                inside = geometry_mask([geometry_m], out_shape=data.shape,
                                       transform=transform, invert=True)
                codes = data[inside]
                partition, crosses, touches_guard, guard_fraction = classify_partition(
                    codes, codes.size)

            minx, miny, maxx, maxy = geometry_m.bounds
            centroid = geometry_m.centroid
            lon, lat = Transformer.from_crs("EPSG:32643", "EPSG:4326",
                                            always_xy=True).transform(centroid.x, centroid.y)
            pixels = props.get("area_px_20m")
            rows.append({
                "candidate_id": props["candidate_id"],
                "pair": props["pair"],
                "component_id": props["component_id"],
                "partition": partition,
                "n_partitions": len({c for c in codes if c in SPLIT_CODES}),
                "partition_codes": "|".join(str(int(c)) for c in np.unique(codes))
                                   if codes.size else "",
                "crosses_partition_boundary": crosses,
                "touches_guard_band": touches_guard,
                "guard_fraction": round(guard_fraction, 4),
                "bbox_width_px_20m": int(round((maxx - minx) / 20.0)),
                "bbox_height_px_20m": int(round((maxy - miny) / 20.0)),
                "bbox_width_m": round(maxx - minx, 1),
                "bbox_height_m": round(maxy - miny, 1),
                "bbox_area_m2": round((maxx - minx) * (maxy - miny), 1),
                "component_area_px_20m": int(pixels) if pixels is not None else "",
                "component_area_m2": round(pixels * PIXEL_AREA_M2, 1) if pixels else "",
                "mean_score": round(float(props["mean_score"]), 4),
                "max_score": round(float(props["max_score"]), 4),
                "score_threshold": props.get("score_threshold"),
                "priority_percentile": props.get("priority_percentile"),
                "centroid_lon": round(lon, 7),
                "centroid_lat": round(lat, 7),
                "centroid_easting": round(centroid.x, 2),
                "centroid_northing": round(centroid.y, 2),
                "review_order": "",
                "review_status": "unreviewed",
                "human_decision": "",
                "evidence": "",
                "confidence": "",
                "note": "",
            })
    return rows


def order_rows(rows):
    """Review order inside each (pair, partition) group.

    Flags first, because a polygon that straddles a boundary has to be split or
    dropped before it can be labelled at all. Then largest component first, so
    the reviewer starts with clearest evidence, then highest score.
    """
    rows.sort(key=lambda row: (row["pair"], PARTITION_ORDER.get(row["partition"], 9),
                               not (row["crosses_partition_boundary"]
                                    or row["touches_guard_band"]),
                               -int(row["component_area_px_20m"] or 0),
                               -float(row["max_score"])))
    counter = {}
    for row in rows:
        key = (row["pair"], row["partition"])
        counter[key] = counter.get(key, 0) + 1
        row["review_order"] = counter[key]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", default=CANDIDATE_DIR)
    parser.add_argument("--partition-dir", default=PARTITION_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    output_csv = os.path.join(args.output_dir, "review_queue.csv")
    output_json = os.path.join(args.output_dir, "review_queue_summary.json")
    for path in (output_csv, output_json):
        if os.path.exists(path):
            parser.error("refusing to overwrite %s" % path)

    rows = []
    for pair in ("A", "B"):
        candidate_path = os.path.join(
            args.candidate_dir, "review_candidates_%s_p99_9.geojson" % pair)
        partition_path = os.path.join(
            args.partition_dir, "pair_%s_partition_20m.tif" % pair)
        for path in (candidate_path, partition_path):
            if not os.path.isfile(path):
                parser.error("missing input: %s" % path)
        with open(candidate_path) as handle:
            candidates = json.load(handle)
        rows.extend(candidate_rows(pair, candidates, partition_path))
        print("%s: %d candidates" % (pair, len(candidates["features"])))

    order_rows(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        key = "%s/%s" % (row["pair"], row["partition"])
        counts[key] = counts.get(key, 0) + 1
    flagged_cross = [r["candidate_id"] for r in rows if r["crosses_partition_boundary"]]
    flagged_guard = [r["candidate_id"] for r in rows if r["touches_guard_band"]]
    summary = {
        "rows": len(rows),
        "counts_by_pair_and_partition": dict(sorted(counts.items())),
        "crosses_partition_boundary": flagged_cross,
        "touches_guard_band": flagged_guard,
        "fields": FIELDS,
        "status": "review_queue_not_labels",
        "note": ("Every row starts unreviewed with empty human fields. Nothing "
                 "here is ground truth, and the rasterizer still has the final say."),
    }
    with open(output_json, "w") as handle:
        json.dump(summary, handle, indent=1)

    print("\nrows written:", len(rows))
    for key, value in sorted(counts.items()):
        print("  %-18s %d" % (key, value))
    print("crosses a partition boundary:", len(flagged_cross), flagged_cross)
    print("touches the guard band:", len(flagged_guard), flagged_guard)
    print("\nwrote", output_csv)
    print("wrote", output_json)


if __name__ == "__main__":
    main()
