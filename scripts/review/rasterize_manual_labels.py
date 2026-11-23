"""Validate human GeoJSON labels and rasterize them onto each pair's 10 m grid.

Nothing is inferred from OSM, model output or candidate scores. The candidate mask serves only as
a validity mask; a human supplies every feature and its decision in the input GeoJSON.

Water is not part of observation validity. Validity asks whether there is a trustworthy
measurement, meaning NoData, MSK_CLASSI, MSK_QUALIT or a registration gap. Water asks what is on
the ground. This tool consults only the first, so no water raster is read and no polygon is
refused for touching water: water turning into built or filled land is a target class, and
rejecting those pixels would delete the evidence. The reviewer decides, marking stable water as
``no_change`` or ``uncertain``, and built-over water with before/after evidence as ``change``.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import yaml
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_geom
from rasterio.enums import Resampling
from shapely.geometry import box, shape

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.grids import (grid_from_dataset, require_expected_crs,
                        require_same_grid, resolution)
from terrashift.data.manifest import load_manifest, pair_config

CLASSES = {"no_change": 0, "change": 1, "uncertain": 255}
EVIDENCE_SOURCES = {"before_after_visual", "field_verified", "other_manual"}
PARTITION_NAMES = {1: "train", 2: "validation", 3: "test"}


def _valid_date(value: str) -> bool:
    try:
        if "T" in value:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            date.fromisoformat(value)
        return True
    except (TypeError, ValueError):
        return False


def validate_feature(feature: dict, expected_pair: str) -> tuple[dict, dict]:
    if feature.get("type") != "Feature" or not isinstance(feature.get("geometry"), dict):
        raise ValueError("each annotation must be a GeoJSON Feature with geometry")
    props = feature.get("properties") or {}
    required = ("record_id", "pair", "label_class", "reviewer", "reviewed_at",
                "evidence_source", "evidence", "confidence")
    missing = [key for key in required if key not in props]
    if missing:
        raise ValueError("feature is missing required properties: %s" % missing)
    if not isinstance(props["record_id"], str) or not props["record_id"].strip():
        raise ValueError("record_id must be a non-empty string")
    if props["pair"] != expected_pair:
        raise ValueError("feature %s has pair %s, expected %s" % (
            props["record_id"], props["pair"], expected_pair))
    if props["label_class"] not in CLASSES:
        raise ValueError("unsupported label_class in %s" % props["record_id"])
    if not isinstance(props["reviewer"], str) or not props["reviewer"].strip():
        raise ValueError("reviewer must be a non-empty string")
    if not _valid_date(props["reviewed_at"]):
        raise ValueError("reviewed_at must be an ISO-8601 date or datetime")
    if props["evidence_source"] not in EVIDENCE_SOURCES:
        raise ValueError("evidence_source must be a manual source; automated sources are rejected")
    if not isinstance(props["evidence"], str) or len(props["evidence"].strip()) < 8:
        raise ValueError("evidence must include a short human-written explanation")
    if props["confidence"] not in ("high", "medium", "low"):
        raise ValueError("confidence must be high, medium, or low")
    geom = shape(feature["geometry"])
    if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty or not geom.is_valid:
        raise ValueError("annotation geometry must be a valid non-empty Polygon or MultiPolygon")
    return props, geom


def _feature_window(geom, transform, width: int, height: int):
    from rasterio.features import geometry_window
    from shapely.geometry import mapping

    class _DatasetShape:
        def __init__(self):
            self.transform = transform
            self.width = width
            self.height = height

    try:
        return geometry_window(_DatasetShape(), [mapping(geom)], pad_x=0, pad_y=0)
    except Exception as exc:
        raise ValueError("annotation lies outside the pair grid: %s" % exc) from exc


def rasterize_annotations(pair: str, geojson_path: Path, output_dir: Path,
                          partition_dir: Path, score_dir: Path,
                          validity_dir: Optional[Path] = None,
                          expected_crs: Optional[str] = None) -> dict:
    vector = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    if vector.get("type") != "FeatureCollection" or not isinstance(vector.get("features"), list):
        raise ValueError("input must be a GeoJSON FeatureCollection")
    if vector.get("crs"):
        crs_name = vector["crs"].get("properties", {}).get("name", "")
        if "4326" not in crs_name and "CRS84" not in crs_name.upper():
            raise ValueError("manual label GeoJSON must use WGS84 longitude/latitude")
    props_and_geoms = [validate_feature(feature, pair) for feature in vector["features"]]
    ids = [props["record_id"] for props, _ in props_and_geoms]
    if len(ids) != len(set(ids)):
        raise ValueError("record_id values must be unique")
    if not props_and_geoms:
        raise ValueError("no human annotations found; refusing to create an empty label raster")
    mask_path = output_dir / ("manual_labels_%s.tif" % pair)
    audit_path = output_dir / ("manual_labels_%s_audit.csv" % pair)
    existing_outputs = [path for path in (mask_path, audit_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "refusing to overwrite existing manual-label outputs; use a new --output-dir: %s"
            % ", ".join(str(path) for path in existing_outputs))

    partition_path = partition_dir / ("pair_%s_partition_10m.tif" % pair)
    score_path = score_dir / ("change_score_%s.tif" % pair)
    validity_dir = Path(validity_dir) if validity_dir is not None else Path(score_dir)
    validity_path = validity_dir / ("observation_valid_%s.tif" % pair)
    if not validity_path.is_file():
        raise FileNotFoundError(
            "observation-validity raster not found: %s\n"
            "annotations must be checked against the quality-valid footprint, "
            "not against the change-score raster alone." % validity_path)

    with rasterio.open(partition_path) as part_src, \
            rasterio.open(score_path) as score_src, \
            rasterio.open(validity_path) as validity_src:
        part_grid = grid_from_dataset(part_src)
        score_grid = grid_from_dataset(score_src)
        validity_grid = grid_from_dataset(validity_src)
        if expected_crs:
            require_expected_crs(part_grid, expected_crs,
                                 label="partition raster %s" % partition_path)
            require_expected_crs(score_grid, expected_crs,
                                 label="score raster %s" % score_path)
        require_same_grid(validity_grid, score_grid,
                          "validity raster %s" % validity_path,
                          "score raster %s" % score_path)
        if resolution(part_grid) != (10.0, 10.0):
            raise ValueError("expected 10 m local partition grid, found %g m"
                             % resolution(part_grid)[0])
        profile = part_src.profile.copy()
        partition = part_src.read(1)

        # Explicit, recorded reprojection: 20 m validity raster -> 10 m
        # partition grid, nearest-neighbour so a 10 m pixel inherits the
        # validity of the 20 m pixel that contains it.
        valid_10m = np.zeros((part_grid.height, part_grid.width), dtype=np.uint8)
        reproject(
            source=validity_src.read(1), destination=valid_10m,
            src_transform=validity_src.transform, src_crs=validity_src.crs,
            src_nodata=validity_src.nodata, dst_transform=part_src.transform,
            dst_crs=part_src.crs, dst_nodata=0, resampling=Resampling.nearest)
        valid_10m = valid_10m > 0
        out = np.full((part_grid.height, part_grid.width), 255, dtype=np.uint8)
        assigned = np.zeros(out.shape, dtype=bool)
        feature_rows = []

        for props, geom_wgs84 in props_and_geoms:
            geom_target = shape(transform_geom(
                "EPSG:4326", part_src.crs, geom_wgs84.__geo_interface__, precision=8))
            tile_area = box(part_src.bounds.left, part_src.bounds.bottom,
                            part_src.bounds.right, part_src.bounds.top)
            if not tile_area.covers(geom_target):
                raise ValueError("annotation %s extends outside the pair tile" % props["record_id"])
            window = _feature_window(geom_target, part_src.transform,
                                     part_src.width, part_src.height)
            row0 = int(window.row_off)
            col0 = int(window.col_off)
            h = int(window.height)
            w = int(window.width)
            transform = part_src.window_transform(window)
            burn = rasterize(
                [(geom_target.__geo_interface__, 1)], out_shape=(h, w),
                transform=transform, fill=0, dtype="uint8", all_touched=False)
            local = burn.astype(bool)
            if not local.any():
                raise ValueError("annotation %s covers no pixel centres" % props["record_id"])
            split_values = np.unique(partition[row0:row0 + h, col0:col0 + w][local])
            split_values = split_values[split_values != 0]
            raw_values = np.unique(partition[row0:row0 + h, col0:col0 + w][local])
            if len(split_values) != 1 or np.any(raw_values == 0):
                raise ValueError("annotation %s crosses a split boundary or buffer" % props["record_id"])
            if not np.all(valid_10m[row0:row0 + h, col0:col0 + w][local]):
                # Validity only: NoData, quality flags, registration gap.
                # Water is not part of this test and never rejects a polygon.
                raise ValueError(
                    "annotation %s covers pixels that are not valid observations"
                    % props["record_id"])
            current = assigned[row0:row0 + h, col0:col0 + w]
            if np.any(current & local):
                raise ValueError("annotation %s overlaps another reviewed polygon" % props["record_id"])
            target = out[row0:row0 + h, col0:col0 + w]
            target[local] = CLASSES[props["label_class"]]
            current[local] = True
            code = int(split_values[0])
            feature_rows.append({
                "record_id": props["record_id"], "pair": pair,
                "split": PARTITION_NAMES[code], "label_class": props["label_class"],
                "pixels": int(local.sum()), "area_m2": int(local.sum()) * 100,
                "reviewer": props["reviewer"],
                "reviewed_at": props["reviewed_at"],
                "evidence_source": props["evidence_source"],
                "evidence": props["evidence"],
                "confidence": props["confidence"],
            })

        output_dir.mkdir(parents=True, exist_ok=True)
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255,
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(out, 1)
            dst.update_tags(
                label_source="human_manual_geojson",
                label_schema="manual-labels-v1",
                unlabeled_and_uncertain_value="255",
                no_change_value="0", change_value="1",
                candidate_score_used_as_ground_truth="false",
                osm_used_as_ground_truth="false",
                validity_raster=validity_path.name,
                validity_definition="nodata+msk_classi+msk_qualit+registration_coverage")

    with audit_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    return {"mask": str(mask_path), "audit": str(audit_path),
            "feature_count": len(feature_rows),
            "pixels": int(sum(row["pixels"] for row in feature_rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--pair", choices=("A", "B"), required=True)
    parser.add_argument("--labels", type=Path, required=True,
                        help="human-edited WGS84 GeoJSON FeatureCollection")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--partition-dir", type=Path, default=None)
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--validity-dir", type=Path, default=None,
                        help="directory holding observation_valid_<PAIR>.tif; "
                             "defaults to the score directory")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg["data"]
    output_dir = args.output_dir or Path(data_cfg["manual_label_dir"])
    partition_dir = args.partition_dir or Path(data_cfg["local_partition_dir"])
    score_dir = args.score_dir or Path(data_cfg["local_score_dir"])
    validity_dir = args.validity_dir or score_dir
    manifest = load_manifest(args.manifest or Path(data_cfg["manifest"]))
    expected_crs = pair_config(manifest, args.pair)["crs"]
    result = rasterize_annotations(args.pair, args.labels, output_dir,
                                   partition_dir, score_dir,
                                   validity_dir=validity_dir,
                                   expected_crs=expected_crs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
