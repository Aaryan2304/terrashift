"""Consolidate the positive and negative reviews into one reviewed-case register.

38 cases: 26 from the positive review, 12 from the negative review. Reviewer decisions,
change types, confidence, reasons and notes are copied verbatim. Nothing is reclassified,
reinterpreted or repaired.

The original detection register (outputs/register/*) and both review packages are read-only
here and are hashed before and after.
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

POS_DIR = ROOT / "outputs/submission_review_v2"
NEG_DIR = ROOT / "outputs/negative_review_v1"
OUT = ROOT / "outputs/register/reviewed_cases_v2"
MANIFEST = ROOT / "outputs/evaluation/final_locked_test_v1/evaluation_manifest.json"
EXPECTED = {"accepted_man_made": 23, "rejected_natural": 5, "rejected_non_target": 5,
            "uncertain": 5}
QUALITY_FLAGS = {
    "NB_L08": ("resolution_limited",
               "single-pixel component; decision preserved but not usable as a strong "
               "example of confirmed natural change"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hashes(path):
    return {str(i.relative_to(path)): sha256(i) for i in sorted(path.rglob("*")) if i.is_file()}


def main() -> int:
    # Parsed first so --help prints usage and exits 0 without inspecting or writing outputs.
    # The overwrite guard below stays active for every real execution.
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if OUT.exists() and any(OUT.iterdir()):
        print("output directory already populated, refusing to overwrite")
        return 2

    # The consolidated register writes INSIDE outputs/register/, so guarding that directory
    # wholesale would flag its own output. Guard the original register's files by name.
    REGISTER_FILES = ["detections.csv", "detections.geojson", "detections.gpkg",
                      "candidate_provenance.csv", "register_validation.json",
                      "detection_register_report.md", "rejected_cases_note.md"]

    def register_hashes():
        return {name: sha256(ROOT / "outputs/register" / name) for name in REGISTER_FILES}

    guarded = {
        "positive_v2": POS_DIR,
        "negative_v1": NEG_DIR,
        "locked_test": ROOT / "outputs/evaluation/final_locked_test_v1",
    }
    before = {name: directory_hashes(path) for name, path in guarded.items()}
    register_before = register_hashes()

    pos_sheet = list(csv.DictReader(open(POS_DIR / "review_worksheet_26.csv")))
    neg_sheet = list(csv.DictReader(open(NEG_DIR / "negative_review_worksheet.csv")))
    if len(pos_sheet) != 26 or len(neg_sheet) != 12:
        print("INPUT COUNT MISMATCH: positive %d, negative %d" % (len(pos_sheet), len(neg_sheet)))
        return 3

    pos_locs = gpd.read_file(POS_DIR / "review_locations.gpkg", layer="review_locations")
    neg_locs = gpd.read_file(NEG_DIR / "negative_review.gpkg", layer="negative_review")
    pos_map = json.loads((POS_DIR / "package_summary.json").read_text())["mapping"]
    data_manifest = yaml.safe_load(open(ROOT / "configs/data_manifest.yaml", encoding="utf-8"))
    pairs = data_manifest["pairs"]
    dates = {p["id"]: p["date"] for p in data_manifest["products"]}
    reviewed_at_pos = datetime.utcfromtimestamp(
        (POS_DIR / "review_worksheet_26.csv").stat().st_mtime).date().isoformat()
    reviewed_at_neg = datetime.utcfromtimestamp(
        (NEG_DIR / "negative_review_worksheet.csv").stat().st_mtime).date().isoformat()

    cases = []
    for sheet in sorted(pos_sheet, key=lambda r: r["location_id"]):
        location = pos_locs[pos_locs["location_id"] == sheet["location_id"]].iloc[0]
        pair = sheet["pair_id"]
        cases.append({
            "case_id": "RC-%s" % sheet["location_id"],
            "source_review_id": sheet["location_id"],
            "case_source": "positive_region_review",
            "pair_id": pair,
            "acquisition_date_before": dates[pairs[pair]["date1"]],
            "acquisition_date_after": dates[pairs[pair]["date2"]],
            "source_component_id": int(location["component_id"]),
            "source_candidate_ids": ";".join(pos_map[sheet["location_id"]]),
            "source_thresholds": sheet["source_thresholds"],
            "model_threshold": 0.05,
            "model_mean_score": round(float(location["mean_score_005"]), 6),
            "model_max_score": round(float(location["max_score_005"]), 6),
            "decision_status": sheet["decision"],
            "change_type": sheet["change_type"],
            "confidence": sheet["confidence"],
            "area_pixels": int(location["area_pixels"]),
            "area_m2": float(location["area_m2"]),
            "flagged_area_m2": None,
            "area_semantics": "annotated_change_area_of_reviewed_component",
            "visual_change_visible": sheet["visual_change_visible"],
            "evidence_before": str(POS_DIR / sheet["before_evidence"]),
            "evidence_after": str(POS_DIR / sheet["after_evidence"]),
            "overlay_path": str(POS_DIR / "overlays" / ("%s_panel.png"
                                                        % sheet["location_id"])),
            "decision_reason": sheet["decision_reason"],
            "reviewer_notes": sheet["reviewer_notes"],
            "reviewer": "Aaryan",
            "reviewed_at": reviewed_at_pos,
            "data_quality_flag": None,
            "geometry": location.geometry,
        })

    for sheet in sorted(neg_sheet, key=lambda r: r["review_id"]):
        location = neg_locs[neg_locs["review_id"] == sheet["review_id"]].iloc[0]
        pair = sheet["pair_id"]
        flag = QUALITY_FLAGS.get(sheet["review_id"])
        cases.append({
            "case_id": "RC-%s" % sheet["review_id"],
            "source_review_id": sheet["review_id"],
            "case_source": "negative_region_review",
            "pair_id": pair,
            "acquisition_date_before": dates[pairs[pair]["date1"]],
            "acquisition_date_after": dates[pairs[pair]["date2"]],
            "source_component_id": int(sheet["source_component_id"]),
            "source_candidate_ids": None,
            "source_thresholds": sheet["threshold"],
            "model_threshold": 0.05,
            "model_mean_score": round(float(sheet["mean_score"]), 6),
            "model_max_score": round(float(sheet["max_score"]), 6),
            "decision_status": sheet["decision"],
            "change_type": sheet["change_type"],
            "confidence": sheet["confidence"],
            "area_pixels": int(location["component_pixels"]),
            "area_m2": None,
            "flagged_area_m2": float(location["area_m2"]),
            "area_semantics": "model_flagged_area_within_reviewed_no_change",
            "visual_change_visible": sheet["visual_change_visible"],
            "evidence_before": str(NEG_DIR / sheet["before_evidence"]),
            "evidence_after": str(NEG_DIR / sheet["after_evidence"]),
            "overlay_path": str(NEG_DIR / sheet["overlay_path"]),
            "decision_reason": sheet["decision_reason"],
            "reviewer_notes": sheet["reviewer_notes"],
            "reviewer": "Aaryan",
            "reviewed_at": reviewed_at_neg,
            "data_quality_flag": flag[0] if flag else None,
            "geometry": location.geometry,
        })

    if len(cases) != 38:
        print("CONSOLIDATED COUNT MISMATCH:", len(cases))
        return 3

    OUT.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame(cases, geometry="geometry", crs="EPSG:32643")
    frame.drop(columns=["geometry"]).to_csv(OUT / "reviewed_cases.csv", index=False)
    frame.to_file(OUT / "reviewed_cases.gpkg", layer="reviewed_cases", driver="GPKG")

    features = []
    for case in cases:
        wgs = transform_geom("EPSG:32643", "EPSG:4326", mapping(case["geometry"]),
                             precision=8)
        props = {k: (None if v is None else v)
                 for k, v in case.items() if k != "geometry"}
        props["area_m2"] = case["area_m2"]
        features.append({"type": "Feature", "geometry": wgs, "properties": props})
    (OUT / "reviewed_cases.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, indent=1) + "\n")

    # Provenance crosswalk: every source reference, once.
    provenance = []
    for case in cases:
        if case["case_source"] == "positive_region_review":
            for candidate_id in case["source_candidate_ids"].split(";"):
                threshold = "0.05" if "_005_" in candidate_id else "0.50"
                provenance.append({
                    "case_id": case["case_id"],
                    "case_source": case["case_source"],
                    "source_reference_type": "candidate_record",
                    "source_reference_id": candidate_id,
                    "threshold": threshold,
                    "component_id": case["source_component_id"],
                    "source_package": "outputs/submission_review_v1/candidates.gpkg",
                    "linked_case_id": case["case_id"],
                })
        else:
            provenance.append({
                "case_id": case["case_id"],
                "case_source": case["case_source"],
                "source_reference_type": "review_record",
                "source_reference_id": case["source_review_id"],
                "threshold": case["source_thresholds"],
                "component_id": case["source_component_id"],
                "source_package": "outputs/negative_review_v1/negative_review.gpkg",
                "linked_case_id": case["case_id"],
            })
    with open(OUT / "reviewed_case_provenance.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)

    # ---- validation ---------------------------------------------------------------
    checks = {}
    checks["1_positive_input_rows"] = {"value": len(pos_sheet), "pass": len(pos_sheet) == 26}
    checks["2_negative_input_rows"] = {"value": len(neg_sheet), "pass": len(neg_sheet) == 12}
    checks["3_consolidated_records"] = {"value": len(cases), "pass": len(cases) == 38}
    status_counts = {}
    for case in cases:
        status_counts[case["decision_status"]] = status_counts.get(case["decision_status"], 0) + 1
    checks["4_accepted_preserved"] = {"value": status_counts.get("accepted_man_made", 0),
                                      "pass": status_counts.get("accepted_man_made", 0) == 23}
    checks["5_rejected_preserved"] = {
        "value": status_counts.get("rejected_natural", 0) + status_counts.get("rejected_non_target", 0),
        "pass": status_counts.get("rejected_natural", 0) == 5
        and status_counts.get("rejected_non_target", 0) == 5}
    checks["6_uncertain_preserved"] = {"value": status_counts.get("uncertain", 0),
                                       "pass": status_counts.get("uncertain", 0) == 5}
    checks["6_status_counts_match_expected"] = {"value": status_counts, "expected": EXPECTED,
                                                "pass": status_counts == EXPECTED}
    ids = [case["source_review_id"] for case in cases]
    checks["7_each_review_id_once"] = {"count": len(ids), "unique": len(set(ids)),
                                       "pass": len(ids) == len(set(ids)) == 38}
    case_ids = [case["case_id"] for case in cases]
    checks["8_no_duplicate_case_ids"] = {"pass": len(set(case_ids)) == len(case_ids) == 38}

    resolve_problems = []
    candidates = gpd.read_file(ROOT / "outputs/submission_review_v1/candidates.gpkg",
                               layer="candidates")
    known_candidates = set(candidates["candidate_id"])
    known_neg = set(neg_locs["review_id"])
    for case in cases:
        if case["case_source"] == "positive_region_review":
            for candidate_id in case["source_candidate_ids"].split(";"):
                if candidate_id not in known_candidates:
                    resolve_problems.append(candidate_id)
        else:
            if case["source_review_id"] not in known_neg:
                resolve_problems.append(case["source_review_id"])
    checks["9_references_resolve"] = {"problems": resolve_problems,
                                      "pass": not resolve_problems}

    field_problems = []
    for case in cases:
        if case["case_source"] == "positive_region_review":
            sheet = next(r for r in pos_sheet if r["location_id"] == case["source_review_id"])
            pairs_to_check = (("decision_status", "decision"), ("change_type", "change_type"),
                              ("confidence", "confidence"),
                              ("decision_reason", "decision_reason"),
                              ("reviewer_notes", "reviewer_notes"))
        else:
            sheet = next(r for r in neg_sheet if r["review_id"] == case["source_review_id"])
            pairs_to_check = (("decision_status", "decision"), ("change_type", "change_type"),
                              ("confidence", "confidence"),
                              ("decision_reason", "decision_reason"),
                              ("reviewer_notes", "reviewer_notes"))
        for case_field, sheet_field in pairs_to_check:
            if str(case[case_field] or "") != str(sheet[sheet_field] or ""):
                field_problems.append({"case": case["case_id"], "field": case_field})
    checks["10_decisions_match_worksheets"] = {"problems": field_problems,
                                               "pass": not field_problems}
    checks["11_geometries_valid"] = {
        "pass": bool(frame.geometry.is_valid.all() and not frame.geometry.is_empty.any())}
    geojson = json.loads((OUT / "reviewed_cases.geojson").read_text())
    degree_ok = all(-180 <= x <= 180 and -90 <= y <= 90
                    for feature in geojson["features"]
                    for polygon in ([feature["geometry"]["coordinates"]]
                                    if feature["geometry"]["type"] == "Polygon"
                                    else feature["geometry"]["coordinates"])
                    for ring in polygon for x, y in ring)
    checks["12_crs_and_order"] = {"gpkg_crs": str(frame.crs).upper(),
                                  "geojson_features": len(geojson["features"]),
                                  "coords_in_degrees": degree_ok,
                                  "pass": str(frame.crs).upper().endswith("32643") and degree_ok}

    semantics_ok = True
    for case in cases:
        if case["case_source"] == "positive_region_review":
            if case["area_m2"] is None or case["flagged_area_m2"] is not None:
                semantics_ok = False
            if abs(case["area_m2"] - case["area_pixels"] * 100.0) > 1e-6:
                semantics_ok = False
            if case["area_semantics"] != "annotated_change_area_of_reviewed_component":
                semantics_ok = False
        else:
            if case["flagged_area_m2"] is None or case["area_m2"] is not None:
                semantics_ok = False
            if abs(case["flagged_area_m2"] - case["area_pixels"] * 100.0) > 1e-6:
                semantics_ok = False
            if case["area_semantics"] != "model_flagged_area_within_reviewed_no_change":
                semantics_ok = False
    checks["13_area_semantics"] = {"pass": semantics_ok}

    missing = [case["case_id"] for case in cases
               if not (Path(case["evidence_before"]).exists()
                       and Path(case["evidence_after"]).exists()
                       and Path(case["overlay_path"]).exists())]
    checks["14_evidence_exists"] = {"missing": missing, "pass": not missing}

    after = {name: directory_hashes(path) for name, path in guarded.items()}
    checks["15_review_packages_unchanged"] = {
        "positive_v2": before["positive_v2"] == after["positive_v2"],
        "negative_v1": before["negative_v1"] == after["negative_v1"],
        "files": {k: len(before[k]) for k in ("positive_v2", "negative_v1")},
        "pass": before["positive_v2"] == after["positive_v2"]
        and before["negative_v1"] == after["negative_v1"]}
    register_after = register_hashes()
    register_problems = [name for name in REGISTER_FILES
                         if register_before[name] != register_after[name]]
    checks["16_original_register_unchanged"] = {
        "files_checked": REGISTER_FILES,
        "changed": register_problems,
        "identical": not register_problems,
        "pass": not register_problems}
    manifest = json.loads(MANIFEST.read_text())
    immutable = {}
    for pair, entry in manifest["splits"].items():
        for kind in ("labels", "partition"):
            immutable["%s_%s" % (pair, kind)] = {
                "pass": sha256(Path(entry[kind]["path"])) == entry[kind]["sha256"]}
    for name, entry in manifest["artifacts"].items():
        immutable[name] = {"pass": sha256(ROOT / entry["path"]) == entry["sha256"]}
    checks["17_locked_test_artifacts_unchanged"] = {
        "artifacts": immutable,
        "tree_identical": before["locked_test"] == after["locked_test"],
        "pass": all(v["pass"] for v in immutable.values())
        and before["locked_test"] == after["locked_test"]}

    overall = all(isinstance(v, dict) and v.get("pass") is not False
                  and all(inner.get("pass") is not False for inner in v.values()
                          if isinstance(inner, dict))
                  for v in checks.values())
    checks["overall_pass"] = overall
    (OUT / "validation.json").write_text(json.dumps(checks, indent=1) + "\n")

    by_source = {}
    for case in cases:
        key = (case["pair_id"], case["case_source"], case["decision_status"])
        by_source[str(key)] = by_source.get(str(key), 0) + 1
    print("cases:", len(cases), "| status counts:", status_counts)
    print("crosswalk rows:", len(provenance))
    print("VALIDATION:", "PASS" if overall else "FAIL")
    for name, value in checks.items():
        if name != "overall_pass":
            print("  %-38s %s" % (name, value.get("pass")))
    print("by pair/source/status:", json.dumps(by_source, indent=1))
    return 0 if overall else 4


if __name__ == "__main__":
    raise SystemExit(main())
