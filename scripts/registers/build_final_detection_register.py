"""Build the final detection register from the completed 26-location review.

One register row per unique geographical location. The reviewer's decision vocabulary
(`accepted_man_made`, `uncertain`, `building_construction`) is preserved verbatim; the
v1 schema could not hold it, so `configs/detection_register_schema_v2.yaml` supersedes v1
for this register. v1 and its builder are left untouched.

The script reads the worksheet, the two review packages and the locked-test artefacts, and
writes only under outputs/register/.
"""

from __future__ import annotations

import argparse

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

import geopandas as gpd
from rasterio.warp import transform_geom
from shapely.geometry import mapping

V1 = ROOT / "outputs/submission_review_v1"
V2 = ROOT / "outputs/submission_review_v2"
OUT = ROOT / "outputs/register"
MANIFEST = ROOT / "outputs/evaluation/final_locked_test_v1/evaluation_manifest.json"
WORKSHEET = V2 / "review_worksheet_26.csv"
EXPECTED = {"accepted_man_made": 23, "uncertain": 3}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hashes(path):
    return {str(item.relative_to(path)): sha256(item)
            for item in sorted(path.rglob("*")) if item.is_file()}


def main() -> int:
    # Parsed first so --help prints usage and exits 0 without inspecting or writing outputs.
    # The overwrite guard below stays active for every real execution.
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if OUT.exists() and any(OUT.iterdir()):
        print("outputs/register/ already populated, refusing to overwrite")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    v1_before, v2_before = directory_hashes(V1), directory_hashes(V2)
    worksheet_hash_before = sha256(WORKSHEET)

    rows = list(csv.DictReader(open(WORKSHEET)))
    counts = {}
    for row in rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    if counts != EXPECTED:
        print("WORKSHEET DISCREPANCY: decisions", counts, "expected", EXPECTED)
        return 3
    if any(row["decision"] == "awaiting_human_review" for row in rows):
        print("WORKSHEET DISCREPANCY: unresolved rows present")
        return 3

    locations = gpd.read_file(V2 / "review_locations.gpkg", layer="review_locations")
    candidates = gpd.read_file(V1 / "candidates.gpkg", layer="candidates")
    summary = json.loads((V2 / "package_summary.json").read_text())
    mapping_by_location = summary["mapping"]
    manifest = json.loads(MANIFEST.read_text())
    data_manifest = yaml.safe_load(open(ROOT / "configs/data_manifest.yaml", encoding="utf-8"))
    pairs = data_manifest["pairs"]
    product_dates = {p["id"]: p["date"] for p in data_manifest["products"]}

    # Reviewed_at: the worksheet carries no per-row timestamp, so the completion time of the
    # artifact is used. Documented in the report; not invented per row.
    reviewed_at = datetime.utcfromtimestamp(WORKSHEET.stat().st_mtime).date().isoformat()

    register = []
    for location in locations.sort_values("location_id").itertuples():
        sheet = next(r for r in rows if r["location_id"] == location.location_id)
        ids = mapping_by_location[location.location_id]
        sources = candidates[candidates["candidate_id"].isin(ids)]
        if len(sources) != 2:
            print("PROVENANCE DISCREPANCY at", location.location_id, "records", len(sources))
            return 3
        thresholds = sorted(sources["threshold"].tolist())
        if thresholds != ["0.05", "0.50"]:
            print("THRESHOLD DISCREPANCY at", location.location_id, thresholds)
            return 3
        centroid = location.geometry.centroid
        pair = location.pair_id
        register.append({
            "detection_id": "TS-%s" % location.location_id,
            "register_id": "%s-%04d" % (pair, int(location.location_id.split("_L")[1])),
            "location_id": location.location_id,
            "pair_id": pair,
            "tile": pairs[pair]["tile"],
            "source_candidate_ids": ";".join(ids),
            "source_thresholds": ";".join(thresholds),
            "acquisition_date_before": product_dates[pairs[pair]["date1"]],
            "acquisition_date_after": product_dates[pairs[pair]["date2"]],
            "change_type": sheet["change_type"],
            "decision_status": sheet["decision"],
            "confidence": sheet["confidence"],
            "geometry_wkt": location.geometry.wkt,
            "crs": "EPSG:32643",
            "longitude": round(centroid.x, 6),
            "latitude": round(centroid.y, 6),
            "area_pixels": int(location.area_pixels),
            "area_m2": float(location.area_m2),
            "area_ha": round(float(location.area_m2) / 10000.0, 6),
            "mean_score": float(location.mean_score_005),
            "max_score": float(location.max_score_005),
            "evidence_before": str(V2 / "chips" / ("%s_before.png" % location.location_id)),
            "evidence_after": str(V2 / "chips" / ("%s_after.png" % location.location_id)),
            "overlay_path": str(V2 / "overlays" / ("%s_panel.png" % location.location_id)),
            "decision_reason": sheet["decision_reason"],
            "reviewer_notes": sheet["reviewer_notes"],
            "reviewer": "Aaryan",
            "reviewed_at": reviewed_at,
            "feature_geometry_class": "unclassified",
            "detection_source": "model",
            "notes": ("reviewed candidate area from reviewed positive test pixels; "
                      "not an independently measured building footprint"),
            "geometry": location.geometry,
        })

    frame = gpd.GeoDataFrame(register, geometry="geometry", crs="EPSG:32643")
    csv_fields = [c for c in frame.columns]
    frame.drop(columns=["geometry"]).to_csv(OUT / "detections.csv", index=False)
    frame.to_file(OUT / "detections.gpkg", layer="detections", driver="GPKG")

    features = []
    for row in register:
        wgs84 = transform_geom("EPSG:32643", "EPSG:4326", mapping(row["geometry"]),
                               precision=8)
        properties = {k: v for k, v in row.items() if k != "geometry"}
        features.append({"type": "Feature", "geometry": wgs84, "properties": properties})
    (OUT / "detections.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "name": "detections",
         "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
         "features": features}, indent=1) + "\n")

    # Candidate provenance crosswalk, 52 rows.
    crosswalk = []
    for row in register:
        for candidate_id in row["source_candidate_ids"].split(";"):
            source = candidates[candidates["candidate_id"] == candidate_id].iloc[0]
            crosswalk.append({
                "source_candidate_id": candidate_id,
                "location_id": row["location_id"],
                "pair_id": row["pair_id"],
                "threshold": source["threshold"],
                "component_id": int(source["component_id"]),
                "linked_detection_id": row["detection_id"],
                "review_decision": row["decision_status"],
                "change_type": row["change_type"],
                "confidence": row["confidence"],
                "source_geometry_reference": ("outputs/submission_review_v1/"
                                              "candidates.gpkg#%s" % candidate_id),
            })
    crosswalk.sort(key=lambda item: (item["pair_id"], item["source_candidate_id"]))
    with open(OUT / "candidate_provenance.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crosswalk[0]))
        writer.writeheader()
        writer.writerows(crosswalk)

    # ---- validation ---------------------------------------------------------------
    checks = {}
    checks["1_register_rows"] = {"value": len(register), "pass": len(register) == 26}
    by_status = {}
    for row in register:
        by_status[row["decision_status"]] = by_status.get(row["decision_status"], 0) + 1
    checks["2_status_counts"] = {"value": by_status, "expected": EXPECTED,
                                 "pass": by_status == EXPECTED}
    by_pair = {}
    for row in register:
        key = (row["pair_id"], row["decision_status"])
        by_pair["%s_%s" % key] = by_pair.get("%s_%s" % key, 0) + 1
    sheet_pair = {}
    for row in rows:
        key = "%s_%s" % (row["pair_id"], row["decision"])
        sheet_pair[key] = sheet_pair.get(key, 0) + 1
    checks["3_pair_counts_match_worksheet"] = {"value": by_pair, "worksheet": sheet_pair,
                                               "pass": by_pair == sheet_pair}
    checks["4_unique_location_ids"] = {
        "value": int(len({r["location_id"] for r in register})),
        "pass": len({r["location_id"] for r in register}) == 26}
    checks["4_unique_detection_ids"] = {
        "value": int(len({r["detection_id"] for r in register})),
        "pass": len({r["detection_id"] for r in register}) == 26}

    link_problems = []
    for row in register:
        expected_ids = set(mapping_by_location[row["location_id"]])
        if set(row["source_candidate_ids"].split(";")) != expected_ids:
            link_problems.append(row["location_id"])
    checks["5_source_ids_match_mapping"] = {"problems": link_problems,
                                            "pass": not link_problems}
    checks["6_crosswalk_rows"] = {"value": len(crosswalk), "pass": len(crosswalk) == 52}
    crosswalk_ids = [item["source_candidate_id"] for item in crosswalk]
    checks["7_each_candidate_once"] = {
        "value": len(crosswalk_ids), "unique": len(set(crosswalk_ids)),
        "all_52_present": set(crosswalk_ids) == set(candidates["candidate_id"]),
        "pass": len(crosswalk_ids) == 52 and len(set(crosswalk_ids)) == 52
        and set(crosswalk_ids) == set(candidates["candidate_id"])}
    thresholds_ok = True
    for location_id in {r["location_id"] for r in crosswalk}:
        subset = [c["threshold"] for c in crosswalk if c["location_id"] == location_id]
        if sorted(subset) != ["0.05", "0.50"]:
            thresholds_ok = False
    checks["8_one_record_per_threshold"] = {"pass": thresholds_ok}

    field_problems = []
    for row in register:
        sheet = next(r for r in rows if r["location_id"] == row["location_id"])
        for register_field, sheet_field in (("decision_status", "decision"),
                                            ("change_type", "change_type"),
                                            ("confidence", "confidence"),
                                            ("decision_reason", "decision_reason"),
                                            ("reviewer_notes", "reviewer_notes")):
            if str(row[register_field]) != str(sheet[sheet_field]):
                field_problems.append({"location": row["location_id"],
                                       "field": register_field})
    checks["9_fields_match_worksheet_exactly"] = {"problems": field_problems,
                                                 "pass": not field_problems}
    valid = frame.geometry.is_valid.all() and not frame.geometry.is_empty.any()
    checks["10_geometries_valid"] = {"pass": bool(valid)}

    area_attr = frame["area_m2"].to_numpy(dtype=float)
    area_geom = frame.geometry.area.to_numpy(dtype=float)
    checks["11_areas_match_geometry"] = {
        "max_relative_difference": float(np.max(np.abs(area_geom - area_attr) / area_attr)),
        "pass": bool(np.allclose(area_geom, area_attr, rtol=1e-6))}

    dates_problems = []
    for row in register:
        pair = pairs[row["pair_id"]]
        if row["acquisition_date_before"] != product_dates[pair["date1"]]:
            dates_problems.append(row["detection_id"])
        if row["acquisition_date_after"] != product_dates[pair["date2"]]:
            dates_problems.append(row["detection_id"])
    checks["12_acquisition_dates_match_manifest"] = {"problems": dates_problems,
                                                    "pass": not dates_problems}

    geojson = json.loads((OUT / "detections.geojson").read_text())
    lon_ok = True
    for feature in geojson["features"]:
        coords = feature["geometry"]["coordinates"]
        polygons = coords if feature["geometry"]["type"] == "MultiPolygon" else [coords]
        for polygon in polygons:
            for ring in polygon:
                for x, y in ring:
                    if not (-180.0 <= x <= 180.0 and -90.0 <= y <= 90.0):
                        lon_ok = False
    checks["13_geojson_epsg4326_order"] = {
        "features": len(geojson["features"]),
        "coordinates_in_degree_range": lon_ok,
        "pass": len(geojson["features"]) == 26 and lon_ok}

    v1_after, v2_after = directory_hashes(V1), directory_hashes(V2)
    worksheet_hash_after = sha256(WORKSHEET)
    checks["14_review_packages_unchanged"] = {
        "v1_files": len(v1_before), "v2_files": len(v2_before),
        "v1_identical": v1_before == v1_after, "v2_identical": v2_before == v2_after,
        "worksheet_identical": worksheet_hash_before == worksheet_hash_after,
        "pass": v1_before == v1_after and v2_before == v2_after
        and worksheet_hash_before == worksheet_hash_after}

    immutable = {}
    for pair, entry in manifest["splits"].items():
        for kind in ("labels", "partition"):
            current = sha256(Path(entry[kind]["path"]))
            immutable["%s_%s" % (pair, kind)] = {"pass": current == entry[kind]["sha256"]}
    for name, entry in manifest["artifacts"].items():
        immutable[name] = {"pass": sha256(ROOT / entry["path"]) == entry["sha256"]}
    checks["15_locked_test_artifacts_unchanged"] = {
        "artifacts": immutable, "all_identical": all(v["pass"] for v in immutable.values()),
        "pass": all(v["pass"] for v in immutable.values())}

    overall = all(isinstance(v, dict) and v.get("pass") is not False
                  and all(inner.get("pass") is not False for inner in v.values()
                          if isinstance(inner, dict))
                  for v in checks.values())
    checks["overall_pass"] = overall
    (OUT / "register_validation.json").write_text(json.dumps(checks, indent=1) + "\n")

    print("register rows:", len(register), "| crosswalk rows:", len(crosswalk))
    print("by status:", by_status, "| by pair+status:", by_pair)
    print("VALIDATION:", "PASS" if overall else "FAIL")
    for name, value in checks.items():
        if name == "overall_pass":
            continue
        print("  %-40s %s" % (name, value.get("pass")))
    return 0 if overall else 4


if __name__ == "__main__":
    raise SystemExit(main())
