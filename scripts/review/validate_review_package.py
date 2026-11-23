"""Validate the review package and render a contact sheet.

Checks the package against the immutable locked-test artefacts: candidate count, ID
uniqueness, geometry validity, area consistency, CRS, stripe containment, worksheet state,
chip pairing, and, most importantly, that the annotation, partition and metric files still
hash to the values recorded in the evaluation manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

import geopandas as gpd

PKG = ROOT / "outputs/submission_review_v1"
MANIFEST = ROOT / "outputs/evaluation/final_locked_test_v1/evaluation_manifest.json"
METRICS = ROOT / "outputs/evaluation/final_locked_test_v1/test_metrics.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checks = {}

    frame = gpd.read_file(PKG / "candidates.gpkg", layer="candidates")
    checks["record_count"] = {"value": len(frame), "expected": 52,
                             "pass": len(frame) == 52}
    checks["unique_ids"] = {"value": int(frame["candidate_id"].nunique()),
                            "expected": 52,
                            "pass": int(frame["candidate_id"].nunique()) == 52}
    checks["crs"] = {"value": str(frame.crs).upper(),
                     "pass": str(frame.crs).upper().endswith("32643")}
    valid = frame.geometry.is_valid.all()
    checks["geometries_valid"] = {"value": bool(valid), "pass": bool(valid)}
    empty = frame.geometry.is_empty.any()
    checks["no_empty_geometries"] = {"value": not bool(empty), "pass": not bool(empty)}

    area_attr = frame["area_m2"].to_numpy(dtype=float)
    area_geom = frame.geometry.area.to_numpy(dtype=float)
    ratio = area_geom / area_attr
    checks["geometry_area_matches_attribute"] = {
        "min_ratio": round(float(ratio.min()), 6),
        "max_ratio": round(float(ratio.max()), 6),
        "pass": bool(np.allclose(ratio, 1.0, atol=0.01))}
    checks["area_equals_pixels_x100"] = {
        "pass": bool(np.allclose(area_attr, frame["area_pixels"].to_numpy() * 100.0))}

    counts = frame.groupby("pair_id").size().to_dict()
    checks["by_pair"] = {"value": {k: int(v) for k, v in counts.items()},
                         "expected": {"A": 32, "B": 20},
                         "pass": counts == {"A": 32, "B": 20}}

    # Candidate pixels at 0.05 must reconcile with the evaluated positives.
    metrics = json.loads(METRICS.read_text())
    recon = {}
    for pair in ("A", "B"):
        at05 = frame[(frame["pair_id"] == pair) & (frame["threshold"] == "0.05")]
        total = int(at05["area_pixels"].sum())
        expected = metrics["counts"][pair]["evaluated_unique_positive"]
        recon[pair] = {"package_pixels": total, "evaluated_positive": expected,
                       "pass": total == expected}
    checks["positive_pixel_reconciliation"] = recon

    # Every candidate must sit inside its pair's test stripe.
    stripe = {}
    for pair in ("A", "B"):
        partition_path = ROOT / ("outputs/partitions_v2_observation_valid/"
                                 "pair_%s_partition_10m.tif" % pair)
        with rasterio.open(partition_path) as handle:
            partition = handle.read(1)
            transform = handle.transform
        columns = np.flatnonzero((partition == 3).any(axis=0))
        col0, col1 = int(columns.min()), int(columns.max()) + 1
        subset = frame[frame["pair_id"] == pair]
        inverse = ~transform
        cols = [int((inverse * (geom.centroid.x, geom.centroid.y))[0])
                for geom in subset.geometry]
        inside = all(col0 <= c < col1 for c in cols)
        stripe[pair] = {"stripe_columns": [col0, col1],
                        "min_centroid_col": min(cols), "max_centroid_col": max(cols),
                        "pass": inside}
    checks["inside_test_stripe"] = stripe

    worksheet = list(csv.DictReader(open(PKG / "reviewer_worksheet.csv")))
    checks["worksheet_rows"] = {"value": len(worksheet), "pass": len(worksheet) == 52}
    checks["worksheet_all_pending"] = {
        "value": sorted({row["decision"] for row in worksheet}),
        "pass": all(row["decision"] == "awaiting_human_review" for row in worksheet)}
    checks["worksheet_no_preselected_types"] = {
        "pass": all(row["change_type"] == "" for row in worksheet)}

    # Chip pairing: 104 files, and each before/after pair shares pixel dimensions.
    chips = sorted((PKG / "chips").glob("*.png"))
    pairs_ok = True
    for path in PKG.joinpath("chips").glob("*_before.png"):
        after = path.with_name(path.name.replace("_before", "_after"))
        if not after.exists():
            pairs_ok = False
            break
        with rasterio.open(path) as a, rasterio.open(after) as b:
            if (a.width, a.height) != (b.width, b.height):
                pairs_ok = False
                break
    checks["chip_pairs"] = {"files": len(chips), "pass": len(chips) == 104 and pairs_ok}
    checks["overlays"] = {"files": len(list((PKG / "overlays").glob("*.png"))),
                          "pass": len(list((PKG / "overlays").glob("*.png"))) == 52}

    # Immutability: annotation, partition and metric files vs the evaluation manifest.
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
        path = ROOT / entry["path"]
        current = sha256(path)
        immutable[name] = {"recorded": entry["sha256"][:16], "current": current[:16],
                          "pass": current == entry["sha256"]}
    checks["source_artifacts_unchanged"] = immutable

    all_pass = True
    for name, value in checks.items():
        if isinstance(value, dict):
            if value.get("pass") is False:
                all_pass = False
            for pair_value in value.values():
                if isinstance(pair_value, dict) and pair_value.get("pass") is False:
                    all_pass = False
    checks["overall_pass"] = all_pass

    (PKG / "validation.json").write_text(json.dumps(checks, indent=1) + "\n")
    for name, value in checks.items():
        if name == "overall_pass":
            continue
        status = value.get("pass") if isinstance(value, dict) else None
        print("%-34s %s" % (name, "PASS" if status else status))
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")

    # Contact sheet
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    panels = sorted((PKG / "overlays").glob("*_panel.png"))
    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3.4, rows * 2.0))
    axes = np.atleast_2d(axes)
    for axis in axes.ravel():
        axis.set_axis_off()
    for index, path in enumerate(panels):
        axis = axes[index // columns][index % columns]
        axis.imshow(mpimg.imread(path))
        axis.set_axis_off()
        axis.set_title(path.stem.replace("_panel", ""), fontsize=7)
    figure.suptitle("TerraShift review package — %d candidates, awaiting human review"
                    % len(panels), fontsize=11)
    figure.tight_layout()
    figure.savefig(PKG / "contact_sheet.png", dpi=95)
    plt.close(figure)
    print("wrote", PKG / "contact_sheet.png")
    return 0 if all_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
