"""Rank high-score areas for analyst review; do not treat them as detections.

The percentile cut is a prioritisation rule only. It is not calibrated against
labels and it does not assign a change class.

``--score-dir`` must point at a baseline run that applied the paired
observation-valid mask, so the candidates never sit on an invalid pixel. The
score raster's nodata value is the validity boundary.
"""

import argparse
import csv
import os
import sys

import numpy as np
import rasterio
from pyproj import Transformer
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.geometry import clean_binary_mask

DEFAULT_SCORE_DIR = os.path.join(ROOT, "outputs", "baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["A", "B"], required=True)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--score-dir", default=DEFAULT_SCORE_DIR,
                        help="directory holding change_score_<PAIR>.tif")
    parser.add_argument("--output", default=None,
                        help="CSV path; defaults to outputs/audit next to the "
                             "existing review files")
    parser.add_argument("--min-component-px", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.percentile < 100:
        parser.error("percentile must be between 0 and 100")
    if args.min_component_px < 1:
        parser.error("min-component-px must be positive")

    score_path = os.path.join(args.score_dir, "change_score_%s.tif" % args.pair)
    if not os.path.isfile(score_path):
        parser.error("score raster not found: %s" % score_path)
    out_path = args.output or os.path.join(
        ROOT, "outputs", "audit",
        "review_candidates_%s_p%s.csv" % (args.pair, str(args.percentile).replace('.', '_')))
    if os.path.exists(out_path):
        parser.error("refusing to overwrite existing candidate CSV: %s "
                     "(pass --output with a new path)" % out_path)

    with rasterio.open(score_path) as ds:
        score = ds.read(1, masked=True).filled(np.nan)
        transform = ds.transform
        crs = ds.crs
    if crs is None:
        raise SystemExit("score raster has no CRS: %s" % score_path)
    valid = np.isfinite(score)
    threshold = float(np.percentile(score[valid], args.percentile))
    mask = clean_binary_mask((score > threshold) & valid, valid=valid,
                             min_area_px=args.min_component_px)
    labels, count = ndi.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    ids = np.arange(1, count + 1)
    centroids = ndi.center_of_mass(mask, labels=labels, index=ids)
    means = ndi.mean(score, labels=labels, index=ids)
    maxima = ndi.maximum(score, labels=labels, index=ids)
    boxes = ndi.find_objects(labels)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    rows = []
    for component, size, center, mean_score, max_score in zip(
            ids, sizes[ids], centroids, means, maxima):
        row, col = center
        easting, northing = rasterio.transform.xy(transform, row, col)
        lon, lat = transformer.transform(easting, northing)
        sl = boxes[int(component) - 1]
        x0, y0 = rasterio.transform.xy(transform, sl[0].start, sl[1].start,
                                       offset="ul")
        x1, y1 = rasterio.transform.xy(transform, sl[0].stop, sl[1].stop,
                                       offset="ul")
        rows.append({
            "pair": args.pair,
            "priority_percentile": args.percentile,
            "score_threshold": threshold,
            "component_id": int(component),
            "area_px": int(size),
            "area_m2": int(size) * 20 * 20,
            "centroid_easting": float(easting),
            "centroid_northing": float(northing),
            "longitude": float(lon),
            "latitude": float(lat),
            "bbox_min_easting": min(x0, x1),
            "bbox_min_northing": min(y0, y1),
            "bbox_max_easting": max(x0, x1),
            "bbox_max_northing": max(y0, y1),
            "mean_score": float(mean_score),
            "max_score": float(max_score),
            "status": "unreviewed_candidate_for_analyst",
        })
    rows.sort(key=lambda r: (-r["area_px"], -r["mean_score"]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = list(rows[0]) if rows else [
        "pair", "priority_percentile", "score_threshold", "component_id",
        "area_px", "area_m2", "centroid_easting", "centroid_northing",
        "longitude", "latitude", "bbox_min_easting", "bbox_min_northing",
        "bbox_max_easting", "bbox_max_northing", "mean_score", "max_score",
        "status",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out_path)
    print("score threshold %.6f; valid pixels %d; review candidates %d" %
          (threshold, int(valid.sum()), len(rows)))
    print("This percentile list is for review only, not validated detections.")


if __name__ == "__main__":
    main()
