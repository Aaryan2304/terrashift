"""Validate a detection register and export it as CSV and GeoJSON.

The register is written by a human, one row per reviewed place. This only enforces
``configs/detection_register_schema.yaml`` (v1) or ``..._v2.yaml`` when the row
carries ``pair_id``, and converts WKT to RFC 7946 GeoJSON in WGS84. A
``confirmed`` row needs a reviewer, a date and an evidence note; a ``candidate`` row keeps
``change_type = unknown`` and is never counted as a detection in the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date, datetime
from pathlib import Path

import yaml
from rasterio.warp import transform_geom
from shapely import wkt
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_V2 = ROOT / "configs" / "detection_register_schema_v2.yaml"

from terrashift.data.manifest import load_manifest, pair_config

REQUIRED = ["register_id", "pair", "tile", "date1", "date2", "geometry_wkt", "crs",
            "longitude", "latitude", "area_m2", "area_ha", "change_type",
            "confidence", "status", "detection_source", "evidence",
            "before_image", "after_image", "reviewer", "reviewed_at"]
CHANGE_TYPES = {"construction", "demolition", "reclamation", "other_structural",
                "not_change", "unknown"}
CONFIDENCE = {"high", "medium", "low"}
STATUSES = {"candidate", "reviewed", "confirmed", "rejected"}
SOURCES = {"model", "classical_baseline", "shoreline_sweep", "manual_only"}


def _valid_date(value):
    try:
        if "T" in value:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            date.fromisoformat(value)
        return True
    except (TypeError, ValueError):
        return False


def _schema_columns(path):
    """Column rules from a schema file, so validation follows the declared contract."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload.get("columns", [])


def validate_row_v2(row, manifest, seen_ids):
    """Validate one row against detection-register-v2.

    v2 keeps the reviewer's own vocabulary (``accepted_man_made``, ``uncertain``,
    ``building_construction``) instead of translating it into v1's enums, and it
    identifies a row by ``location_id`` rather than ``register_id`` alone.
    """
    problems = []
    for column in _schema_columns(SCHEMA_V2):
        name = column["name"]
        if column.get("required") and not str(row.get(name, "")).strip():
            problems.append("missing required column %r" % name)
            continue
        value = str(row.get(name, "")).strip()
        if not value:
            continue
        if column.get("values") and value not in column["values"]:
            problems.append("%s=%r is not one of %s" % (name, value, column["values"]))
        pattern = column.get("pattern")
        if pattern and not re.match(pattern, value):
            problems.append("%s=%r does not match %s" % (name, value, pattern))
        if column.get("type") == "date" and not _valid_date(value):
            problems.append("%s=%r is not a date" % (name, value))
    if problems:
        raise ValueError("; ".join(problems))

    identifier = row["location_id"]
    if identifier in seen_ids:
        raise ValueError("duplicate location_id %r" % identifier)
    seen_ids.add(identifier)

    pair = row["pair_id"]
    pair_cfg = pair_config(manifest, pair)
    if row["tile"] != pair_cfg["tile"]:
        raise ValueError("%s: tile %r does not match %s"
                         % (identifier, row["tile"], pair_cfg["tile"]))
    if row["crs"] != pair_cfg["crs"]:
        raise ValueError("%s: crs %r does not match %s"
                         % (identifier, row["crs"], pair_cfg["crs"]))
    geometry = wkt.loads(row["geometry_wkt"])
    if not geometry.is_valid or geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("%s: geometry is not a valid polygon" % identifier)
    if geometry.area <= 0:
        raise ValueError("%s: geometry has zero area" % identifier)
    return geometry


def validate_row(row, manifest, seen_ids):
    """Raise ``ValueError`` if one register row breaks a documented rule."""
    if "pair_id" in row and "pair" not in row:
        return validate_row_v2(row, manifest, seen_ids)
    for column in REQUIRED:
        if column not in row or str(row[column]).strip() == "":
            raise ValueError("row is missing required column %r: %r" % (column, row))
    identifier = row["register_id"]
    if identifier in seen_ids:
        raise ValueError("duplicate register_id: %s" % identifier)
    seen_ids.add(identifier)

    if row["pair"] not in ("A", "B"):
        raise ValueError("register %s has pair %r" % (identifier, row["pair"]))
    pair_cfg = pair_config(manifest, row["pair"])
    if row["tile"] != pair_cfg["tile"]:
        raise ValueError("register %s names tile %r, pair %s is %s"
                         % (identifier, row["tile"], row["pair"], pair_cfg["tile"]))
    if row["crs"] != pair_cfg["crs"]:
        raise ValueError("register %s has CRS %r, pair %s uses %s"
                         % (identifier, row["crs"], row["pair"], pair_cfg["crs"]))
    if row["change_type"] not in CHANGE_TYPES:
        raise ValueError("register %s has change_type %r" % (identifier, row["change_type"]))
    if row["confidence"] not in CONFIDENCE:
        raise ValueError("register %s has confidence %r" % (identifier, row["confidence"]))
    if row["status"] not in STATUSES:
        raise ValueError("register %s has status %r" % (identifier, row["status"]))
    if row["detection_source"] not in SOURCES:
        raise ValueError("register %s has detection_source %r"
                         % (identifier, row["detection_source"]))
    if not _valid_date(row["reviewed_at"]):
        raise ValueError("register %s has an invalid reviewed_at %r"
                         % (identifier, row["reviewed_at"]))

    if row["status"] == "confirmed":
        if not row["reviewer"].strip():
            raise ValueError("confirmed register %s has no reviewer" % identifier)
        if len(row["evidence"].strip()) < 10:
            raise ValueError("confirmed register %s has no usable evidence note"
                             % identifier)
    if row["status"] == "candidate" and row["change_type"] != "unknown":
        raise ValueError("candidate register %s must not carry change_type %r"
                         % (identifier, row["change_type"]))

    geometry = wkt.loads(row["geometry_wkt"])
    if geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("register %s geometry is %r" % (identifier, geometry.geom_type))
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("register %s geometry is empty or invalid" % identifier)
    for key in ("longitude", "latitude", "area_m2", "area_ha"):
        try:
            float(row[key])
        except (TypeError, ValueError) as exc:
            raise ValueError("register %s has a non-numeric %s" % (identifier, key)) from exc
    return geometry


def to_feature(row, geometry, pair_cfg):
    wgs84 = transform_geom(row["crs"], "EPSG:4326", mapping(geometry), precision=8)
    properties = {key: value for key, value in row.items()
                  if key not in ("geometry_wkt", "crs")}
    properties["register_status"] = row.get("status") or row.get("decision_status")
    return {"type": "Feature", "geometry": wgs84, "properties": properties}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-geojson", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/data_manifest.yaml")
    args = parser.parse_args()

    existing = [str(p) for p in (args.output_csv, args.output_geojson) if p.exists()]
    if existing:
        raise SystemExit("refusing to overwrite existing register outputs: %s"
                         % ", ".join(existing))

    manifest = load_manifest(args.manifest)
    with args.input.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("the register is empty; nothing to export")

    seen = set()
    features = []
    for row in rows:
        geometry = validate_row(row, manifest, seen)
        pair_key = row.get("pair") or row.get("pair_id")
        features.append(to_feature(row, geometry, pair_config(manifest, pair_key)))

    summary = {}
    for row in rows:
        key = row.get("status") or row.get("decision_status")
        summary[key] = summary.get(key, 0) + 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.output_geojson.write_text(json.dumps({
        "type": "FeatureCollection",
        "name": "terrashift_detection_register",
        "features": features}, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.output_csv)
    print("wrote", args.output_geojson)
    print("rows by status:", summary)


if __name__ == "__main__":
    main()
