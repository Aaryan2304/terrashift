"""Turn a binary change mask into vector components with area and length.

Each connected component of the mask becomes one feature with:

* geometry, polygonised on pixel edges (``rasterio.features.shapes``);
* area in square metres, with an erosion/dilation range so the boundary
  sensitivity is visible;
* skeleton length, for structures that are linear rather than blob-shaped;
* the pair, CRS and the mask that produced it, so the record is traceable.

This produces *candidate* geometry. It does not decide whether a component is a
real change; that is the reviewer's job, recorded in the detection register.

Usage:
    python scripts/components_to_geojson.py --mask prediction_mask_A.tif \
        --pair A --output-geojson ... --output-csv ...
Nothing is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import transform_geom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.geometry import boundary_area_range, components, skeleton_length


def extract(mask, transform, pixel_size, min_area_px=4):
    """Vectorise a boolean mask into component records."""
    mask = np.asarray(mask, dtype=bool)
    labels, areas = components(mask)
    records = []
    for label, area in enumerate(areas, start=1):
        if area < min_area_px:
            continue
        component = labels == label
        geometry = None
        for shape, value in features.shapes(component.astype(np.uint8),
                                            mask=component, transform=transform):
            if int(value) == 1:
                geometry = shape
                break
        if geometry is None:
            continue
        area_info = boundary_area_range(component, pixel_size * pixel_size)
        length_info = skeleton_length(component, pixel_size)
        rows, cols = np.nonzero(component)
        records.append({
            "component_id": int(label),
            "area_px": int(area),
            "area_m2": area_info["nominal_m2"],
            "area_min_m2": area_info["min_m2"],
            "area_max_m2": area_info["max_m2"],
            "skeleton_length_m": round(length_info["length_m"], 2),
            "skeleton_px": length_info["skeleton_px"],
            "centroid_row": float(rows.mean()),
            "centroid_col": float(cols.mean()),
            "geometry": geometry,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True,
                        help="raster where 1 means candidate change")
    parser.add_argument("--pair", choices=["A", "B"], required=True)
    parser.add_argument("--output-geojson", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-area-px", type=int, default=4)
    args = parser.parse_args()

    existing = [str(p) for p in (args.output_geojson, args.output_csv) if p.exists()]
    if existing:
        raise SystemExit("refusing to overwrite existing component outputs: %s"
                         % ", ".join(existing))

    with rasterio.open(args.mask) as ds:
        if ds.crs is None:
            raise SystemExit("mask has no CRS; refusing to write ungeoreferenced geometry")
        data = ds.read(1)
        transform = ds.transform
        crs = ds.crs
        pixel_size = abs(float(ds.transform.a))

    mask = (data == 1)
    records = extract(mask, transform, pixel_size, args.min_area_px)
    if not records:
        raise SystemExit("no component reached the %d pixel minimum" % args.min_area_px)

    features_out = []
    for record in records:
        wgs84 = transform_geom(str(crs), "EPSG:4326",
                               record["geometry"], precision=8)
        properties = {key: value for key, value in record.items() if key != "geometry"}
        properties.update({"pair": args.pair, "source_mask": args.mask.name,
                           "review_status": "unreviewed_candidate_not_ground_truth"})
        features_out.append({"type": "Feature", "geometry": wgs84,
                             "properties": properties})

    args.output_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.output_geojson.write_text(json.dumps({
        "type": "FeatureCollection",
        "name": "change_components_%s" % args.pair,
        "features": features_out}, indent=2) + "\n", encoding="utf-8")

    fields = ["component_id", "area_px", "area_m2", "area_min_m2", "area_max_m2",
              "skeleton_length_m", "skeleton_px", "centroid_row", "centroid_col"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fields})

    print("wrote", args.output_geojson, "(%d components)" % len(records))
    print("wrote", args.output_csv)
    total_area = sum(r["area_m2"] for r in records)
    print("total nominal area %.0f m2; total skeleton length %.0f m"
          % (total_area, sum(r["skeleton_length_m"] for r in records)))


if __name__ == "__main__":
    main()
