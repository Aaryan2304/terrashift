"""Validate the deduplicated 26-location package against the 52-record package.

Checks grouping correctness, linkage back to the original candidate records, geometry and
score preservation, file counts, worksheet state, and the immutability of both the source
evaluation artifacts and the original review package.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

import geopandas as gpd

V1 = ROOT / "outputs/submission_review_v1"
V2 = ROOT / "outputs/submission_review_v2"
MANIFEST = ROOT / "outputs/evaluation/final_locked_test_v1/evaluation_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print((__doc__ or "usage: %s [--help]" % __file__).strip())
        return 0
    checks = {}
    v1 = gpd.read_file(V1 / "candidates.gpkg", layer="candidates")
    v2 = gpd.read_file(V2 / "review_locations.gpkg", layer="review_locations")

    checks["1_unique_locations"] = {"value": len(v2), "expected": 26,
                                    "pass": len(v2) == 26}
    checks["1_unique_ids"] = {"value": int(v2["location_id"].nunique()),
                              "pass": int(v2["location_id"].nunique()) == len(v2)}

    counts = v2.groupby("pair_id").size().to_dict()
    checks["2_by_pair"] = {"value": {k: int(v) for k, v in counts.items()},
                           "expected": {"A": 16, "B": 10},
                           "pass": counts == {"A": 16, "B": 10}}

    # 3. linkage: every source id exists in v1, and the group's geometry is that of its
    #    source records. Also confirms each group holds exactly two records, one per
    #    threshold, with a single pair and one component.
    linkage_problems = []
    geometry_problems = []
    score_problems = []
    for _, row in v2.iterrows():
        ids = row["source_candidate_ids"].split(";")
        thresholds = row["source_thresholds"].split(";")
        sources = v1[v1["candidate_id"].isin(ids)]
        if len(sources) != 2 or len(ids) != 2:
            linkage_problems.append({"location": row["location_id"],
                                     "reason": "expected 2 source records",
                                     "found": len(sources)})
            continue
        if set(sources["pair_id"]) != {row["pair_id"]}:
            linkage_problems.append({"location": row["location_id"],
                                     "reason": "pair mismatch"})
        if set(sources["component_id"].astype(int)) != {int(row["component_id"])}:
            linkage_problems.append({"location": row["location_id"],
                                     "reason": "component mismatch"})
        if sorted(thresholds) != ["0.05", "0.50"] or set(sources["threshold"]) != set(thresholds):
            linkage_problems.append({"location": row["location_id"],
                                     "reason": "threshold set mismatch",
                                     "value": thresholds})
        # 5. geometry must be identical, not merely equivalent.
        for other in sources.geometry:
            if not other.equals_exact(row.geometry, 0.0):
                geometry_problems.append({"location": row["location_id"],
                                          "reason": "geometry differs from source"})
                break
        # 4. scores must match the per-threshold source records.
        for threshold, mean_key, max_key in (("0.05", "mean_score_005", "max_score_005"),
                                             ("0.50", "mean_score_050", "max_score_050")):
            source = sources[sources["threshold"] == threshold]
            if source.empty:
                score_problems.append({"location": row["location_id"],
                                       "reason": "missing source at " + threshold})
                continue
            if abs(float(source["mean_score"].iloc[0]) - float(row[mean_key])) > 1e-9:
                score_problems.append({"location": row["location_id"],
                                       "reason": "mean mismatch at " + threshold})
            if abs(float(source["max_score"].iloc[0]) - float(row[max_key])) > 1e-9:
                score_problems.append({"location": row["location_id"],
                                       "reason": "max mismatch at " + threshold})

    checks["3_source_linkage"] = {"problems": linkage_problems,
                                  "checked": len(v2), "pass": not linkage_problems}
    checks["4_threshold_scores_preserved"] = {"problems": score_problems,
                                              "pass": not score_problems}
    checks["5_geometry_not_altered"] = {"problems": geometry_problems,
                                        "pass": not geometry_problems}
    checks["6_geometries_valid"] = {"all_valid": bool(v2.geometry.is_valid.all()),
                                    "none_empty": bool(not v2.geometry.is_empty.any()),
                                    "pass": bool(v2.geometry.is_valid.all()
                                                 and not v2.geometry.is_empty.any())}
    checks["7_crs"] = {"value": str(v2.crs).upper(),
                       "pass": str(v2.crs).upper().endswith("32643")}

    # 8/9. imagery and overlays.
    chips = sorted((V2 / "chips").glob("*.png"))
    pairs_ok = True
    for path in (V2 / "chips").glob("*_before.png"):
        after = path.with_name(path.name.replace("_before", "_after"))
        if not after.exists():
            pairs_ok = False
            break
        with rasterio.open(path) as a, rasterio.open(after) as b:
            if (a.width, a.height) != (b.width, b.height):
                pairs_ok = False
                break
    checks["8_image_pairs"] = {"files": len(chips), "expected": 52,
                               "before_after_dimensions_match": pairs_ok,
                               "pass": len(chips) == 52 and pairs_ok}
    overlays = sorted((V2 / "overlays").glob("*.png"))
    checks["9_overlays"] = {"files": len(overlays), "expected": 26,
                            "pass": len(overlays) == 26}
    checks["9_every_location_has_overlay"] = {
        "pass": {p.stem.replace("_panel", "") for p in overlays}
        == set(v2["location_id"])}

    worksheet = list(csv.DictReader(open(V2 / "review_worksheet_26.csv")))
    checks["10_worksheet_rows"] = {"value": len(worksheet), "pass": len(worksheet) == 26}
    checks["11_all_pending"] = {
        "decisions": sorted({r["decision"] for r in worksheet}),
        "pass": all(r["decision"] == "awaiting_human_review" for r in worksheet)
        and all(r["change_type"] == "" for r in worksheet)
        and all(r["confidence"] == "" for r in worksheet)
        and all(r["decision_reason"] == "" for r in worksheet)}
    checks["11_gpkg_review_fields_unset"] = {
        "review_status": sorted(v2["review_status"].unique().tolist()),
        "change_type": sorted(v2["change_type"].unique().tolist()),
        "nulls_present": bool(v2["confidence"].isna().all()
                              and v2["decision_reason"].isna().all()),
        "pass": set(v2["review_status"].unique()) == {"awaiting_human_review"}
        and set(v2["change_type"].unique()) == {"unknown"}
        and bool(v2["confidence"].isna().all() and v2["decision_reason"].isna().all())}

    # 12. original package unchanged: structure plus its own recorded verdict.
    v1_worksheet = list(csv.DictReader(open(V1 / "reviewer_worksheet.csv")))
    v1_validation = json.loads((V1 / "validation.json").read_text())
    checks["12_v1_package_intact"] = {
        "records": int(len(v1)),
        "chips": len(list((V1 / "chips").glob("*.png"))),
        "overlays": len(list((V1 / "overlays").glob("*.png"))),
        "worksheet_rows": len(v1_worksheet),
        "v1_validation_overall": v1_validation.get("overall_pass"),
        "pass": len(v1) == 52 and len(list((V1 / "chips").glob("*.png"))) == 104
        and len(list((V1 / "overlays").glob("*.png"))) == 52
        and len(v1_worksheet) == 52 and v1_validation.get("overall_pass") is True}

    # 13. source evaluation artifacts.
    manifest = json.loads(MANIFEST.read_text())
    immutable = {}
    for pair, entry in manifest["splits"].items():
        for kind in ("labels", "partition"):
            path = Path(entry[kind]["path"])
            current = sha256(path)
            immutable["%s_%s" % (pair, kind)] = {
                "recorded": entry[kind]["sha256"][:16], "current": current[:16],
                "pass": current == entry[kind]["sha256"]}
    for name, entry in manifest["artifacts"].items():
        current = sha256(ROOT / entry["path"])
        immutable[name] = {"recorded": entry["sha256"][:16], "current": current[:16],
                           "pass": current == entry["sha256"]}
    checks["13_source_artifacts_unchanged"] = immutable

    # Score statistics are threshold-independent by construction; assert the pairs match so
    # the duplicated columns cannot drift.
    checks["4_scores_threshold_independent"] = {
        "pass": bool(np.allclose(v2["mean_score_005"], v2["mean_score_050"])
                     and np.allclose(v2["max_score_005"], v2["max_score_050"])),
        "note": "mean/max do not depend on a threshold; only recall does"}

    overall = True
    for value in checks.values():
        if not isinstance(value, dict):
            overall = False
            continue
        if value.get("pass") is False:
            overall = False
        for inner in value.values():
            if isinstance(inner, dict) and inner.get("pass") is False:
                overall = False
    checks["overall_pass"] = overall

    (V2 / "validation.json").write_text(json.dumps(checks, indent=1) + "\n")
    for name, value in checks.items():
        if name == "overall_pass" or not isinstance(value, dict):
            continue
        status = value.get("pass")
        print("%-36s %s" % (name, "PASS" if status else status))
    print("\nOVERALL:", "PASS" if overall else "FAIL")
    return 0 if overall else 4


if __name__ == "__main__":
    raise SystemExit(main())
