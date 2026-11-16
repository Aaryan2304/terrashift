"""Export unreviewed baseline components as a GIS review layer, not labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from rasterio.warp import transform_geom


def export_candidates(csv_path: Path, output_path: Path, source_crs: str,
                     expected_pair: str | None = None) -> int:
    features = []
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("candidate CSV is empty: %s" % csv_path)
    for row in rows:
        if expected_pair is not None and row.get("pair") != expected_pair:
            raise ValueError("candidate row has pair %r, expected %r" %
                             (row.get("pair"), expected_pair))
        xmin = float(row["bbox_min_easting"])
        ymin = float(row["bbox_min_northing"])
        xmax = float(row["bbox_max_easting"])
        ymax = float(row["bbox_max_northing"])
        if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
            raise ValueError("candidate bbox must contain finite coordinates")
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("invalid candidate bounding box: %s" % row.get("component_id"))
        geom = {"type": "Polygon", "coordinates": [[
            [xmin, ymin], [xmax, ymin], [xmax, ymax],
            [xmin, ymax], [xmin, ymin],
        ]]}
        geom = transform_geom(source_crs, "EPSG:4326", geom, precision=8)
        props = {
            "candidate_id": "%s_%s" % (row["pair"], row["component_id"]),
            "pair": row["pair"],
            "component_id": int(row["component_id"]),
            "priority_percentile": float(row["priority_percentile"]),
            "score_threshold": float(row["score_threshold"]),
            "area_px_20m": int(row["area_px"]),
            "bbox_area_m2": int(round((xmax - xmin) * (ymax - ymin))),
            "mean_score": float(row["mean_score"]),
            "max_score": float(row["max_score"]),
            "review_status": "unreviewed_candidate_not_ground_truth",
        }
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    if output_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing candidate GeoJSON: %s" % output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collection = {
        "type": "FeatureCollection",
        "name": "unreviewed_candidate_components",
        "features": features,
    }
    output_path.write_text(json.dumps(collection, indent=2) + "\n", encoding="utf-8")
    return len(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=("A", "B"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-crs", required=True,
                        help="projected CRS of the component CSV coordinates")
    args = parser.parse_args()
    count = export_candidates(args.input, args.output, args.source_crs,
                              expected_pair=args.pair)
    print("exported", count, "unreviewed components to", args.output)
    print("These polygons are candidate bounding boxes, not analyst labels.")


if __name__ == "__main__":
    main()
