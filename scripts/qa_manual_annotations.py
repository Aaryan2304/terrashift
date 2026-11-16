"""Per-record quality assurance for a manual annotation file.

The rasterizer stops at the first bad feature. This runs the same rules over
every feature and reports each one, so a review pass sees all problems at once
instead of fixing them one exception at a time.

Rules are imported from ``scripts/rasterize_manual_labels.py`` where possible, so
the QA and the rasterizer cannot drift apart. The spatial checks replicate its
order: tile containment, split and buffer crossing, observation validity, and
inter-polygon overlap.

Read-only. Nothing is modified, discarded or relabelled.

    python scripts/qa_manual_annotations.py \
        --labels data/manual_annotations.gpkg --pair A \
        --partition-dir outputs/partitions_v2_observation_valid \
        --validity-dir outputs/baseline_v2_observation_valid \
        --output outputs/labels/pair_a_qa/qa_report.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from rasterize_manual_labels import (CLASSES, EVIDENCE_SOURCES, PARTITION_NAMES,
                                     validate_feature)


def read_features(path, layer=None):
    """GeoJSON or GeoPackage, returned as (properties, shapely geometry) pairs."""
    if path.lower().endswith((".gpkg", ".sqlite")):
        import geopandas as gpd
        frame = gpd.read_file(path, layer=layer, engine="pyogrio")
        return [(dict(row.drop(labels="geometry")), row.geometry)
                for _, row in frame.iterrows()], frame.crs
    with open(path) as handle:
        payload = json.load(handle)
    return ([(feature["properties"], shape(feature["geometry"]))
             for feature in payload["features"]], None)


def feature_window(geom, transform, width, height):
    from rasterio.windows import from_bounds
    window = from_bounds(*geom.bounds, transform=transform)
    row0 = max(0, int(np.floor(window.row_off)))
    col0 = max(0, int(np.floor(window.col_off)))
    row1 = min(height, int(np.ceil(window.row_off + window.height)))
    col1 = min(width, int(np.ceil(window.col_off + window.width)))
    return row0, col0, row1, col1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--pair", choices=["A", "B"], required=True)
    parser.add_argument("--partition-dir", required=True)
    parser.add_argument("--validity-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    for path in (args.output, args.csv):
        if path and os.path.exists(path):
            raise SystemExit("refusing to overwrite %s" % path)

    features, source_crs = read_features(args.labels, args.layer)
    print("features:", len(features), "| source CRS:", source_crs)

    partition_path = os.path.join(args.partition_dir, "pair_%s_partition_10m.tif" % args.pair)
    validity_path = os.path.join(args.validity_dir, "observation_valid_%s.tif" % args.pair)

    with rasterio.open(partition_path) as part_src, \
            rasterio.open(validity_path) as valid_src:
        partition = part_src.read(1)
        tile = box(part_src.bounds.left, part_src.bounds.bottom,
                   part_src.bounds.right, part_src.bounds.top)
        valid_10m = np.zeros((part_src.height, part_src.width), dtype=np.uint8)
        reproject(source=valid_src.read(1), destination=valid_10m,
                  src_transform=valid_src.transform, src_crs=valid_src.crs,
                  src_nodata=valid_src.nodata, dst_transform=part_src.transform,
                  dst_crs=part_src.crs, dst_nodata=0, resampling=Resampling.nearest)
        valid_10m = valid_10m > 0
        transform = part_src.transform
        part_crs = part_src.crs.to_string()
        width, height = part_src.width, part_src.height
        bounds = (part_src.bounds.left, part_src.bounds.bottom,
                  part_src.bounds.right, part_src.bounds.top)

    assigned = np.zeros((height, width), dtype=bool)
    rows = []
    for props, geometry in features:
        record = {"record_id": props.get("record_id"), "label_class": props.get("label_class"),
                  "errors": [], "warnings": []}
        try:
            props_checked, geom_wgs84 = validate_feature(
                {"type": "Feature", "properties": props,
                 "geometry": geometry.__geo_interface__}, args.pair)
            record["attributes_valid"] = True
        except Exception as error:                            # noqa: BLE001
            record["attributes_valid"] = False
            record["errors"].append("attribute: %s" % error)
            props_checked, geom_wgs84 = props, geometry

        record["geometry_valid"] = bool(geometry.is_valid)
        record["geometry_type"] = geometry.geom_type
        record["area_m2"] = round(float(geometry.area), 1)

        if not geometry.is_valid:
            record["errors"].append("geometry: invalid polygon")

        # Transform from the file's own CRS, not from an assumed WGS84. A
        # GeoPackage may already be in the pair CRS, in which case this is the
        # identity and must not be applied twice.
        from rasterio.warp import transform_geom
        src_crs = str(source_crs) if source_crs is not None else "EPSG:4326"
        geom_target = shape(transform_geom(src_crs, part_crs,
                                           geometry.__geo_interface__, precision=8))

        if not tile.covers(geom_target):
            record["errors"].append("outside the pair tile")
            rows.append(record)
            continue

        row0, col0, row1, col1 = feature_window(geom_target, transform, width, height)
        if row1 <= row0 or col1 <= col0:
            record["errors"].append("covers no pixel centres")
            rows.append(record)
            continue
        window_transform = rasterio.windows.transform(
            rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0), transform)
        burn = rasterize([(geom_target.__geo_interface__, 1)],
                         out_shape=(row1 - row0, col1 - col0),
                         transform=window_transform, fill=0, dtype="uint8",
                         all_touched=False).astype(bool)

        if not burn.any():
            record["errors"].append("covers no pixel centres")
            rows.append(record)
            continue

        codes = partition[row0:row1, col0:col1][burn]
        unique_codes = np.unique(codes)
        splits = [int(c) for c in unique_codes if c != 0]
        record["partition_codes"] = [int(c) for c in unique_codes]
        record["partitions"] = [PARTITION_NAMES.get(c, str(c)) for c in splits]
        record["guard_pixels"] = int((codes == 0).sum())
        record["burned_pixels"] = int(burn.sum())
        record["guard_fraction"] = round(float((codes == 0).mean()), 4)
        record["pixels"] = int(burn.sum())

        if len(splits) != 1 or np.any(codes == 0):
            reasons = []
            if np.any(codes == 0):
                reasons.append("touches the guard band (%d of %d burned pixels)"
                               % (int((codes == 0).sum()), int(burn.sum())))
            if len(splits) != 1:
                reasons.append("covers %d partitions (%s)"
                               % (len(splits), ", ".join(
                                   PARTITION_NAMES.get(c, str(c)) for c in splits)))
            record["errors"].append("partition: " + "; ".join(reasons))
        else:
            record["split"] = PARTITION_NAMES.get(splits[0], str(splits[0]))

        inside = valid_10m[row0:row1, col0:col1][burn]
        record["invalid_pixels"] = int((~inside).sum())
        if not inside.all():
            record["errors"].append(
                "observation validity: %d of %d burned pixels are not valid"
                % (int((~inside).sum()), int(burn.sum())))

        current = assigned[row0:row1, col0:col1]
        overlap = int((current & burn).sum())
        record["overlap_pixels"] = overlap
        if overlap:
            record["errors"].append("overlap: %d pixels already claimed by an "
                                    "earlier annotation" % overlap)
        current[burn] = True

        record["status"] = "fail" if record["errors"] else "pass"
        rows.append(record)

    summary = {
        "features": len(features),
        "class_counts": {},
        "status_counts": {"pass": 0, "fail": 0},
        "failing_records": [],
        "source_crs": str(source_crs),
        "partition_raster": os.path.abspath(partition_path),
        "validity_raster": os.path.abspath(validity_path),
        "class_values": CLASSES,
    }
    for row in rows:
        summary["class_counts"][row["label_class"]] = \
            summary["class_counts"].get(row["label_class"], 0) + 1
        summary["status_counts"][row.get("status", "fail")] += 1
        if row.get("status") == "fail":
            summary["failing_records"].append({"record_id": row["record_id"],
                                               "reason": "; ".join(row["errors"])})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump({"summary": summary, "records": rows}, handle, indent=1)
    if args.csv:
        fields = ["record_id", "label_class", "status", "split", "pixels",
                  "guard_pixels", "invalid_pixels", "overlap_pixels", "area_m2",
                  "geometry_valid", "attributes_valid", "errors"]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["errors"] = " | ".join(row["errors"])
                writer.writerow(row)

    print("\nclasses:", summary["class_counts"])
    print("pass:", summary["status_counts"]["pass"], "| fail:", summary["status_counts"]["fail"])
    for item in summary["failing_records"]:
        print("  FAIL %-6s %s" % (item["record_id"], item["reason"]))
    print("\nwrote", args.output)
    if args.csv:
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
