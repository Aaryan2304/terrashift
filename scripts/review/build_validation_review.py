from pathlib import Path
"""Build the Pair A validation-split review package.

Three layers for reviewing the validation stripe in QGIS:

1. the candidate rectangles that fall in the validation partition, with the
   class already assigned to them, if any;
2. the annotations that live in the validation partition;
3. proposal areas: places inside the validation partition where the classical
   change score passes the 99th percentile but which are **not** covered by any
   existing annotation or candidate rectangle. These are suggestions for a human
   to look at, not detections and not labels.

    python scripts/review/build_validation_review.py --pair A \
        --output-dir outputs/labels/pair_a_qa/validation_review
"""

import argparse
import csv
import json
import os

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import shapes as raster_shapes
from rasterio.features import rasterize
from scipy import ndimage as ndi
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

ROOT = Path(__file__).resolve().parents[2]

PARTITION_NAMES = {1: "train", 2: "validation", 3: "test"}
MIN_PROPOSAL_PX = 4


def load_queue(path, pair):
    with open(path, newline="") as handle:
        return [row for row in csv.DictReader(handle)
                if row["pair"] == pair and row["partition"] == "validation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="A")
    parser.add_argument("--labels", default="labels/manual_labels_A.geojson")
    parser.add_argument("--queue", default="review/manual_review_v1/review_queue.csv")
    parser.add_argument("--candidates",
                        default="outputs/labels/v2_observation_valid/review_candidates_A_p99_9.geojson")
    parser.add_argument("--audit", default="outputs/labels/pair_a_provisional/manual_labels_A_audit.csv")
    parser.add_argument("--score", default="outputs/baseline_v2_observation_valid/change_score_A.tif")
    parser.add_argument("--partition", default="outputs/partitions_v2_observation_valid/pair_A_partition_20m.tif")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise SystemExit("refusing to write into a non-empty %s" % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. candidates in the validation partition
    queue = load_queue(args.queue, args.pair)
    annotated = {}
    with open(args.audit, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "validation":
                annotated[row["record_id"]] = row

    with open(args.candidates) as handle:
        candidates = json.load(handle)
    by_id = {f["properties"]["candidate_id"]: f for f in candidates["features"]}
    validation_candidates = []
    for row in queue:
        feature = by_id.get(row["candidate_id"])
        if feature is None:
            continue
        props = dict(feature["properties"])
        props.update({"queue_partition": "validation",
                      "annotated_as": annotated.get(row["candidate_id"], {}).get("label_class", ""),
                      "review_order": row["review_order"]})
        validation_candidates.append({"type": "Feature", "properties": props,
                                      "geometry": feature["geometry"]})

    candidate_path = os.path.join(args.output_dir, "validation_candidates.geojson")
    with open(candidate_path, "w") as handle:
        json.dump({"type": "FeatureCollection", "name": "validation_candidates",
                   "features": validation_candidates}, handle, indent=1)

    # --- 2. annotations already in the validation partition
    with open(args.labels) as handle:
        labels = json.load(handle)
    annotation_features = [f for f in labels["features"]
                           if f["properties"]["record_id"] in annotated]
    annotation_path = os.path.join(args.output_dir, "validation_annotations.geojson")
    with open(annotation_path, "w") as handle:
        json.dump({"type": "FeatureCollection", "name": "validation_annotations",
                   "features": annotation_features}, handle, indent=1)

    # --- 3. proposal areas: score tail inside validation, not already reviewed
    with rasterio.open(args.score) as score_src, rasterio.open(args.partition) as part_src:
        score = score_src.read(1, masked=True).filled(np.nan)
        partition = part_src.read(1)
        transform, crs = score_src.transform, score_src.crs
        if score.shape != partition.shape:
            raise SystemExit("score and partition rasters are not on the same grid")

        valid = np.isfinite(score)
        threshold = float(np.percentile(score[valid], args.percentile))
        in_validation = partition == 2
        proposal = (score > threshold) & valid & in_validation

        # remove anything already annotated or already offered as a candidate
        reviewed = np.zeros(score.shape, dtype=bool)
        if annotation_features:
            geoms = [shape(f["geometry"]) for f in annotation_features]
            to_metric = Transformer.from_crs("EPSG:4326", crs.to_string(), always_xy=True)
            geoms = [shapely_transform(to_metric.transform, g) for g in geoms]
            reviewed |= rasterize([(g.__geo_interface__, 1) for g in geoms],
                                  out_shape=score.shape, transform=transform,
                                  fill=0, dtype="uint8").astype(bool)
        if validation_candidates:
            reviewed |= rasterize([(f["geometry"], 1) for f in validation_candidates],
                                  out_shape=score.shape, transform=transform,
                                  fill=0, dtype="uint8").astype(bool)
        proposal &= ~reviewed

        labels_map, count = ndi.label(proposal)
        sizes = np.bincount(labels_map.ravel(), minlength=count + 1)
        keep = sizes >= MIN_PROPOSAL_PX
        keep[0] = False
        cleaned = keep[labels_map]

        to_wgs84 = Transformer.from_crs(crs.to_string(), "EPSG:4326", always_xy=True)
        features = []
        for geom, value in raster_shapes(cleaned.astype("uint8"), mask=cleaned,
                                         transform=transform):
            polygon = shape(geom)
            area_m2 = float(polygon.area)
            inside = score[cleaned & rasterize(
                [(geom, 1)], out_shape=score.shape, transform=transform,
                fill=0, dtype="uint8").astype(bool)]
            inside = inside[np.isfinite(inside)]
            features.append({
                "type": "Feature",
                "properties": {
                    "pair": args.pair,
                    "area_px_20m": int(round(area_m2 / 400.0)),
                    "area_m2": round(area_m2, 1),
                    "max_score": round(float(inside.max()), 3) if inside.size else None,
                    "mean_score": round(float(inside.mean()), 3) if inside.size else None,
                    "status": "proposal_only_not_a_label",
                },
                "geometry": json.loads(json.dumps(
                    shapely_transform(to_wgs84.transform, polygon).__geo_interface__)),
            })
        proposal_path = os.path.join(args.output_dir, "validation_proposal_areas.geojson")
        with open(proposal_path, "w") as handle:
            json.dump({"type": "FeatureCollection", "name": "validation_proposal_areas",
                       "features": features}, handle, indent=1)

    summary = {
        "pair": args.pair,
        "validation_candidates": len(validation_candidates),
        "validation_annotations": len(annotation_features),
        "proposal_areas": len(features),
        "proposal_threshold_percentile": args.percentile,
        "proposal_threshold_value": threshold,
        "proposal_pixels": int(cleaned.sum()),
        "annotation_classes": {},
        "layers": {"candidates": os.path.abspath(candidate_path),
                   "annotations": os.path.abspath(annotation_path),
                   "proposals": os.path.abspath(proposal_path)},
    }
    for row in annotated.values():
        summary["annotation_classes"][row["label_class"]] = \
            summary["annotation_classes"].get(row["label_class"], 0) + 1
    with open(os.path.join(args.output_dir, "validation_review.json"), "w") as handle:
        json.dump(summary, handle, indent=1)

    print("validation candidates:", len(validation_candidates))
    print("validation annotations:", len(annotation_features), summary["annotation_classes"])
    print("proposal areas (score > p%.1f, not already reviewed): %d covering %d px"
          % (args.percentile, len(features), int(cleaned.sum())))
    print("\nwrote", candidate_path)
    print("wrote", annotation_path)
    print("wrote", proposal_path)


if __name__ == "__main__":
    main()
