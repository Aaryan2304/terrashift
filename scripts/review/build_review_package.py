"""Build the QGIS human-review package for the 52 test candidates.

Evidence preparation only. This script decides nothing: every record is written as
``awaiting_human_review`` with ``change_type = unknown``. It does not touch the
checkpoint, probabilities, thresholds, metrics, annotations or partitions.

Candidates are recovered from the reviewed test target using the *same* component
labelling the locked-test metrics used (8-connected on ``covered & target == 1``), so
the component IDs here are the ones in ``test_metrics.json`` and
``test_review_list.csv``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import MultiPolygon, Polygon, mapping, shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import quantize
from terrashift.data.local_patches import pair_image_loader
from terrashift.evaluation.testlock import read_test_target
from terrashift.review.chips import MARGIN_PX, STRUCTURE, reviewed_components, component_scores, chip_bounds, stretch, render_chips, save_chip, rings_in_chip_pixels

OUT = ROOT / "outputs/submission_review_v1"
REGISTRATION = ROOT / "outputs/audit/registration_v2_observation_valid/registration.json"
PARTITIONS = ROOT / "outputs/partitions_v2_observation_valid"
LABELS = ROOT / "outputs/labels"
METRICS = ROOT / "outputs/evaluation/final_locked_test_v1/test_metrics.json"
PREDICTIONS = ROOT / "outputs/evaluation/final_locked_test_v1/test_predictions"
RGB_BANDS = ("B04", "B03", "B02")   # true colour, 10 m


def component_geometry(state, component_id):
    full = state["components"] == component_id
    rows, cols = np.nonzero(full)
    if rows.size == 0:
        return None
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    mask = full[r0:r1, c0:c1]
    transform = state["transform"] * Affine.translation(c0, r0)
    parts = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if int(value) == 1:
            parts.append(shape(geom))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def write_png(path, geometry_px, pair, candidate_id, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    low, high = info["stretch"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for axis, image, caption in ((axes[0], info["before"], "before"),
                                 (axes[1], info["after"], "after")):
        rgb = np.dstack([np.clip((image[i] - low) / (high - low), 0, 1) for i in range(3)])
        axis.imshow(rgb, interpolation="nearest")
        if geometry_px is not None:
            for ring in geometry_px:
                axis.add_patch(MplPolygon(ring, closed=True, fill=False,
                                          edgecolor="#00e5ff", linewidth=1.4))
        axis.set_title("%s  %s  (%s)" % (candidate_id, pair, caption), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("%s  pair %s  mean %.3f  max %.3f  %d px"
                    % (candidate_id, pair, info["mean_score"], info["max_score"],
                       info["area_pixels"]), fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> int:
    # Answer --help before touching any output path: the overwrite guard below must
    # never run (or create directories) just because someone asked for usage text.
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__.strip() if __doc__ else "usage: %s [--help]" % Path(__file__).name)
        return 0
    if OUT.exists() and any(OUT.iterdir()):
        print("output directory already populated, refusing to overwrite:", OUT)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(METRICS.read_text())
    review_rows = list(csv.DictReader(open(ROOT / "outputs/submission_audit_v1/"
                                           "test_review_list.csv")))
    print("source candidates:", len(review_rows))

    states = {pair: reviewed_components(pair) for pair in ("A", "B")}
    scores = {pair: np.load(PREDICTIONS / ("pair_%s_torch_fp32_scores.npy" % pair))
              for pair in ("A", "B")}
    labels = {pair: np.load(PREDICTIONS / ("pair_%s_labels.npy" % pair))
              for pair in ("A", "B")}
    for pair, state in states.items():
        covered = state["target"] != 255
        if not np.array_equal(state["target"][covered].astype(np.int64), labels[pair]):
            raise SystemExit("label vector does not match the rebuilt target for pair " + pair)

    manifest = ROOT / "configs/data_manifest.yaml"
    loader = pair_image_loader(manifest, REGISTRATION, bands=RGB_BANDS)

    chips = OUT / "chips"
    over = OUT / "overlays"
    chips.mkdir(exist_ok=True)
    over.mkdir(exist_ok=True)

    records = []
    missing = []
    for row in review_rows:
        pair = row["pair"]
        component_id = int(row["component_id"])
        state = states[pair]
        geometry = component_geometry(state, component_id)
        stats = component_scores(state, component_id, scores[pair])
        if geometry is None or stats is None:
            missing.append(row["review_id"])
            continue
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        candidate_id = row["review_id"]

        info = render_chips(loader, pair, state, component_id, candidate_id)
        low, high = info["stretch"]
        save_chip(chips / ("%s_before.png" % candidate_id), info["before"], low, high)
        save_chip(chips / ("%s_after.png" % candidate_id), info["after"], low, high)
        write_png(over / ("%s_panel.png" % candidate_id),
                  rings_in_chip_pixels(state, component_id, info["bounds"]),
                  pair, candidate_id,
                  {"stretch": info["stretch"], "before": info["before"],
                   "after": info["after"], "mean_score": stats["mean"],
                   "max_score": stats["max"], "area_pixels": stats["pixels"]})
        records.append({
            "candidate_id": candidate_id,
            "pair_id": pair,
            "component_id": component_id,
            "threshold": row["threshold"],
            "area_pixels": stats["pixels"],
            "area_m2": stats["pixels"] * 100,
            "mean_score": stats["mean"],
            "max_score": stats["max"],
            "median_score": stats["median"],
            "model_recall_in_component": row["model_recall_in_component"],
            "rank_by_area": int(row["rank_by_area"]),
            "source_raster": "outputs/evaluation/final_locked_test_v1/"
                             "test_predictions/pair_%s_torch_fp32_scores.npy" % pair,
            "source_labels": str(ROOT / ("outputs/labels/pair_%s_v2/"
                                         "manual_labels_%s.tif" % (pair.lower(), pair))),
            "geometry_area_m2": round(geometry.area, 2),
            "review_status": "awaiting_human_review",
            "change_type": "unknown",
            "reviewer_confidence": None,
            "decision_reason": None,
            "reviewer_notes": None,
            "geometry": geometry,
        })

    # Spatial layer: GeoPackage, original CRS preserved.
    import geopandas as gpd
    frame = gpd.GeoDataFrame(
        [{k: v for k, v in r.items() if k != "geometry"} | {"geometry": r["geometry"]}
         for r in records], crs="EPSG:32643")
    gpkg = OUT / "candidates.gpkg"
    frame.to_file(gpkg, layer="candidates", driver="GPKG")
    geojson = OUT / "candidates.geojson"
    frame.to_file(geojson, driver="GeoJSON")

    # Reviewer worksheet: one row per candidate, nothing pre-decided.
    worksheet = OUT / "reviewer_worksheet.csv"
    fields = ["candidate_id", "pair_id", "visual_change_visible", "decision", "change_type",
              "confidence", "before_evidence", "after_evidence", "natural_change_reason",
              "acquisition_difference_reason", "uncertainty_reason", "reviewer_notes"]
    with open(worksheet, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "candidate_id": record["candidate_id"], "pair_id": record["pair_id"],
                "visual_change_visible": "", "decision": "awaiting_human_review",
                "change_type": "", "confidence": "",
                "before_evidence": "overlays/%s_panel.png" % record["candidate_id"],
                "after_evidence": "overlays/%s_panel.png" % record["candidate_id"],
                "natural_change_reason": "", "acquisition_difference_reason": "",
                "uncertainty_reason": "", "reviewer_notes": "",
            })

    summary = {
        "candidates_source": len(review_rows),
        "candidates_written": len(records),
        "missing": missing,
        "by_pair": {p: sum(1 for r in records if r["pair_id"] == p) for p in "AB"},
        "distinct_geometries": len({(r["pair_id"], r["component_id"]) for r in records}),
        "crs": "EPSG:32643",
        "gpkg": str(gpkg.relative_to(ROOT)),
        "geojson": str(geojson.relative_to(ROOT)),
        "worksheet": str(worksheet.relative_to(ROOT)),
    }
    (OUT / "package_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
