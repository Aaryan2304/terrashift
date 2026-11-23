"""Build the deduplicated 26-location review package from the 52-record package.

Grouping is by spatial identity, not by component number or ID pattern: two records are
the same location only when they share a pair, have demonstrably equivalent geometry, and
carry one record at each of the two thresholds. Anything that fails those conditions is
reported as an inconsistency rather than forced into a group.

Imagery is regenerated with the same functions the 52-record package used (true colour
B04/B03/B02, 10 m, 960 m margin, one shared stretch per location). The original package is
hashed before and after and is never written.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import geopandas as gpd

from terrashift.review.chips import component_scores, render_chips, reviewed_components, rings_in_chip_pixels, save_chip

V1 = ROOT / "outputs/submission_review_v1"
OUT = ROOT / "outputs/submission_review_v2"
# Duplicated from build_review_package on purpose: importing the sibling CLI would
# re-execute its module-level configuration. A test asserts the two stay equal.
PREDICTIONS = ROOT / "outputs/evaluation/final_locked_test_v1/test_predictions"
THRESHOLDS = {"0.05", "0.50"}
import yaml
with open(ROOT / "configs/data_manifest.yaml", encoding="utf-8") as _handle:
    _manifest = yaml.safe_load(_handle)
PRODUCT_DATES = {p["id"]: p["date"] for p in _manifest["products"]}
DATES = {pair: (PRODUCT_DATES[value["date1"]], PRODUCT_DATES[value["date2"]])
         for pair, value in _manifest["pairs"].items()}
REVIEW_FIELDS = ["review_status", "change_type", "confidence", "decision_reason",
                 "reviewer_notes"]


def directory_hashes(path):
    """sha256 of every file, keyed by relative path. Detects any change to the package."""
    out = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest = hashlib.sha256()
            with open(item, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            out[str(item.relative_to(path))] = digest.hexdigest()
    return out


def geometry_key(geometry):
    """Stable spatial key: normalised WKB hex, so vertex order and direction do not matter."""
    return hashlib.sha256(geometry.normalize().wkb_hex.encode("ascii")).hexdigest()


def write_panel(path, rings, pair, location_id, info, candidate_ids, thresholds, dates):
    """Before/after overlay carrying every identifier the reviewer needs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    low, high = info["stretch"]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 5.0))
    for axis, image, caption, date in ((axes[0], info["before"], "before", dates[0]),
                                       (axes[1], info["after"], "after", dates[1])):
        rgb = np.dstack([np.clip((image[i] - low) / (high - low), 0, 1) for i in range(3)])
        axis.imshow(rgb, interpolation="nearest")
        for ring in rings:
            axis.add_patch(MplPolygon(ring, closed=True, fill=False,
                                      edgecolor="#00e5ff", linewidth=1.4))
        axis.set_title("%s  %s  pair %s" % (location_id, caption, pair) + "\n" + date,
                       fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    subtitle = ("candidates %s   thresholds %s   mean %.3f   max %.3f   %d px"
                % (" / ".join(candidate_ids), ", ".join(thresholds),
                   info["mean_score"], info["max_score"], info["area_pixels"]))
    figure.suptitle("%s   %s" % (location_id, subtitle), fontsize=9.5)
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

    v1_before = directory_hashes(V1)

    frame = gpd.read_file(V1 / "candidates.gpkg", layer="candidates")
    print("v1 records:", len(frame))

    # ---- 1. group by spatial identity ------------------------------------------------
    groups = {}
    for index, row in frame.iterrows():
        identity = (row["pair_id"], geometry_key(row.geometry))
        groups.setdefault(identity, []).append(index)

    inconsistencies = []
    accepted = []
    for (pair, key), members in groups.items():
        subset = frame.loc[members]
        problems = []
        if subset["pair_id"].nunique() != 1:
            problems.append("mixed pair ids")
        thresholds = list(subset["threshold"])
        if sorted(thresholds) != ["0.05", "0.50"]:
            problems.append("thresholds %s" % thresholds)
        if len(subset) != 2:
            problems.append("%d records" % len(subset))
        # Geometries must be equivalent, not merely sharing a hash.
        first = subset.geometry.iloc[0]
        for other in subset.geometry.iloc[1:]:
            if not first.equals_exact(other, 0.0) and not first.equals(other):
                problems.append("geometry mismatch within group")
            if abs(first.area - other.area) > 1e-6:
                problems.append("area mismatch %.6f vs %.6f" % (first.area, other.area))
        if problems:
            inconsistencies.append({"pair": pair, "key": key[:16],
                                    "candidate_ids": list(subset["candidate_id"]),
                                    "problems": problems})
        else:
            accepted.append((pair, sorted(members)))

    print("groups:", len(groups), "accepted:", len(accepted),
          "inconsistencies:", len(inconsistencies))
    if inconsistencies:
        for item in inconsistencies:
            print("  INCONSISTENT", item)
        return 3
    if len(accepted) != 26:
        print("expected 26 unique locations, found", len(accepted))
        return 3

    # ---- 2. stable ids, ordered by geometry so record order cannot matter -------------
    states = {pair: reviewed_components(pair) for pair in ("A", "B")}
    scores = {pair: np.load(PREDICTIONS / ("pair_%s_torch_fp32_scores.npy" % pair))
              for pair in ("A", "B")}

    locations = []
    for pair, members in accepted:
        subset = frame.loc[members]
        geometry = subset.geometry.iloc[0]
        centroid = geometry.centroid
        locations.append({"pair": pair, "members": members, "geometry": geometry,
                          "centroid": (centroid.y, centroid.x),
                          "component_id": int(subset["component_id"].iloc[0])})
    locations.sort(key=lambda item: (item["pair"], round(item["centroid"][0], 3),
                                     round(item["centroid"][1], 3)))
    counters = {"A": 0, "B": 0}
    for item in locations:
        counters[item["pair"]] += 1
        item["location_id"] = "%s_L%02d" % (item["pair"], counters[item["pair"]])

    # ---- 3. imagery, one before/after pair and one overlay per location --------------
    chips = OUT / "chips"
    over = OUT / "overlays"
    chips.mkdir(parents=True, exist_ok=True)
    over.mkdir(parents=True, exist_ok=True)

    from terrashift.data.local_patches import pair_image_loader
    loader = pair_image_loader(ROOT / "configs/data_manifest.yaml",
                               ROOT / "outputs/audit/registration_v2_observation_valid/"
                               "registration.json", bands=("B04", "B03", "B02"))

    records = []
    for item in locations:
        pair = item["pair"]
        subset = frame.loc[item["members"]].sort_values("threshold")
        component_id = item["component_id"]
        state = states[pair]
        stats = component_scores(state, component_id, scores[pair])
        if stats is None:
            raise SystemExit("no scores for component %d pair %s" % (component_id, pair))

        info = render_chips(loader, pair, state, component_id, item["location_id"])
        low, high = info["stretch"]
        save_chip(chips / ("%s_before.png" % item["location_id"]), info["before"], low, high)
        save_chip(chips / ("%s_after.png" % item["location_id"]), info["after"], low, high)

        ids = sorted(subset["candidate_id"])
        thresholds = sorted(subset["threshold"])
        write_panel(over / ("%s_panel.png" % item["location_id"]),
                    rings_in_chip_pixels(state, component_id, info["bounds"]),
                    pair, item["location_id"],
                    {"stretch": info["stretch"], "before": info["before"],
                     "after": info["after"], "mean_score": stats["mean"],
                     "max_score": stats["max"], "area_pixels": stats["pixels"]},
                    ids, thresholds, DATES[pair])

        records.append({
            "location_id": item["location_id"],
            "pair_id": pair,
            "source_candidate_ids": ";".join(ids),
            "source_thresholds": ";".join(sorted(subset["threshold"])),
            "component_id": component_id,
            "area_pixels": stats["pixels"],
            "area_m2": stats["pixels"] * 100,
            # Score statistics do not depend on the threshold; only recall does. Both
            # columns are populated with the same measured values, and the report says so.
            "mean_score_005": stats["mean"],
            "max_score_005": stats["max"],
            "mean_score_050": stats["mean"],
            "max_score_050": stats["max"],
            "median_score": stats["median"],
            "recall_005": float(subset.loc[subset["threshold"] == "0.05",
                                           "model_recall_in_component"].iloc[0]),
            "recall_050": float(subset.loc[subset["threshold"] == "0.50",
                                           "model_recall_in_component"].iloc[0]),
            "review_status": "awaiting_human_review",
            "change_type": "unknown",
            "confidence": None,
            "decision_reason": None,
            "reviewer_notes": None,
            "evidence_overlay": "overlays/%s_panel.png" % item["location_id"],
            "geometry": item["geometry"],
        })

    # ---- 4. layers -------------------------------------------------------------------
    frame_out = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:32643")
    frame_out.to_file(OUT / "review_locations.gpkg", layer="review_locations", driver="GPKG")
    frame_out.to_file(OUT / "review_locations.geojson", driver="GeoJSON")

    # ---- 5. worksheet, 26 rows, nothing pre-decided -----------------------------------
    fields = ["location_id", "pair_id", "source_candidate_ids", "source_thresholds",
              "before_evidence", "after_evidence", "visual_change_visible", "decision",
              "change_type", "confidence", "decision_reason", "reviewer_notes"]
    with open(OUT / "review_worksheet_26.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "location_id": record["location_id"], "pair_id": record["pair_id"],
                "source_candidate_ids": record["source_candidate_ids"],
                "source_thresholds": record["source_thresholds"],
                "before_evidence": "chips/%s_before.png" % record["location_id"],
                "after_evidence": "chips/%s_after.png" % record["location_id"],
                "visual_change_visible": "", "decision": "awaiting_human_review",
                "change_type": "", "confidence": "",
                "decision_reason": "", "reviewer_notes": "",
            })

    # ---- 6. contact sheet, pairs separated -------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    blocks = [(pair, [r for r in records if r["pair_id"] == pair]) for pair in ("A", "B")]
    columns = 4
    row_counts = [int(np.ceil(len(rows) / columns)) for _pair, rows in blocks]
    figure, axes = plt.subplots(sum(row_counts), columns,
                                figsize=(columns * 3.6, sum(row_counts) * 2.3))
    axes = np.atleast_2d(axes)
    for axis in axes.ravel():
        axis.set_axis_off()
    offset = 0
    for (pair, rows), row_count in zip(blocks, row_counts):
        figure.text(0.01, 1 - (offset / sum(row_counts)) - 0.005,
                    "Pair %s — %d unique locations" % (pair, len(rows)),
                    fontsize=12, weight="bold", va="top")
        for position, record in enumerate(rows):
            row, column = divmod(position, columns)
            axis = axes[offset + row][column]
            axis.imshow(mpimg.imread(OUT / record["evidence_overlay"]))
            axis.set_axis_off()
            axis.set_title("%s  (pair %s)" % (record["location_id"], record["pair_id"]),
                           fontsize=8)
        offset += row_count
    figure.suptitle("TerraShift — 26 unique locations, awaiting human review",
                    fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(OUT / "contact_sheet_26.png", dpi=100)
    plt.close(figure)

    v1_after = directory_hashes(V1)
    summary = {
        "v1_records": int(len(frame)),
        "unique_locations": len(records),
        "by_pair": {p: sum(1 for r in records if r["pair_id"] == p) for p in "AB"},
        "grouping": "pair_id + normalised-geometry sha256; each group must hold one record "
                    "at 0.05 and one at 0.50",
        "inconsistencies": inconsistencies,
        "v1_unchanged": v1_before == v1_after,
        "v1_files_hashed": len(v1_before),
        "crs": "EPSG:32643",
        "mapping": {r["location_id"]: r["source_candidate_ids"].split(";") for r in records},
    }
    (OUT / "package_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "mapping"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
