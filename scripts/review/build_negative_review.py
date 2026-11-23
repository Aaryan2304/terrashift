"""Build the negative-region (apparent false-alarm) review package.

Candidates are 8-connected components of predicted-positive pixels *inside manually
reviewed no_change pixels* of the locked test stripe, at the fixed diagnostic threshold
0.05. Nothing is classified: every record starts `awaiting_human_review`.

Unreviewed pixels are never used as evidence. A component is built only from reviewed
no-change pixels, so no unreviewed pixel is presented as a confirmed false positive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import MultiPolygon, Polygon, shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml

from terrashift.data.local_patches import pair_image_loader
from terrashift.evaluation.testlock import read_test_target

STRUCTURE = np.ones((3, 3), dtype=np.uint8)
THRESHOLD = 0.05
MARGIN_PX = 96
OUT = ROOT / "outputs/negative_review_v1"
PRED = ROOT / "outputs/evaluation/final_locked_test_v1/test_predictions"
PARTITIONS = ROOT / "outputs/partitions_v2_observation_valid"
LABELS = ROOT / "outputs/labels"
REGISTRATION = ROOT / "outputs/audit/registration_v2_observation_valid/registration.json"
TARGET_PER_PAIR = 8
BANDS = [(0.05, 0.10), (0.10, 0.50), (0.50, 0.90), (0.90, 1.0000001)]
MIN_DISTANCES = [150, 100, 60, 30]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hashes(path):
    return {str(i.relative_to(path)): sha256(i) for i in sorted(path.rglob("*")) if i.is_file()}


def band_of(value):
    for low, high in BANDS:
        if low <= value < high:
            return "%.2f-%.2f" % (low, min(high, 1.0))
    return None


def size_class(pixels):
    if pixels < 50:
        return "lt_50_px"
    if pixels < 200:
        return "50_200_px"
    if pixels < 1000:
        return "200_1000_px"
    return "ge_1000_px"


def pair_state(pair):
    labels_path = LABELS / ("pair_%s_v2" % pair.lower()) / ("manual_labels_%s.tif" % pair)
    partition_path = PARTITIONS / ("pair_%s_partition_10m.tif" % pair)
    target, col0, _ = read_test_target(labels_path, partition_path)
    with rasterio.open(labels_path) as handle:
        base, height = handle.transform, handle.height
    covered = target != 255
    rows, cols = np.nonzero(covered)
    scores = np.load(PRED / ("pair_%s_torch_fp32_scores.npy" % pair))
    labels = np.load(PRED / ("pair_%s_labels.npy" % pair))
    if not np.array_equal(target[covered].astype(np.int64), labels):
        raise SystemExit("score vector does not map to the stripe pixels for pair " + pair)
    grid = np.full(target.shape, np.nan, dtype=np.float32)
    grid[rows, cols] = scores
    negative = (target == 0) & (grid >= THRESHOLD)
    components, count = ndi.label(negative, structure=STRUCTURE)
    return {
        "target": target, "grid": grid, "components": components, "count": count,
        "transform": base * Affine.translation(col0, 0), "col0": col0,
        "shape": (height, target.shape[1]), "reviewed_negative": target == 0,
        "scores": scores, "rows": rows, "cols": cols,
        "reviewed_negative_pixels": int((target == 0).sum()),
        "flagged_negative_pixels": int(negative.sum()),
    }


def component_geometry(state, component_id):
    full = state["components"] == component_id
    rows, cols = np.nonzero(full)
    if rows.size == 0:
        return None
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    mask = full[r0:r1, c0:c1]
    transform = state["transform"] * Affine.translation(c0, r0)
    parts = [shape(geom) for geom, value in shapes(mask.astype(np.uint8), mask=mask,
                                                   transform=transform) if int(value) == 1]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def component_records(state):
    out = []
    for component_id in range(1, state["count"] + 1):
        mask = state["components"] == component_id
        values = state["grid"][mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows, cols = np.nonzero(mask)
        out.append({
            "component_id": component_id,
            "pixels": int(mask.sum()),
            "mean": float(values.mean()), "median": float(np.median(values)),
            "max": float(values.max()),
            "centroid": (float(rows.mean()), float(cols.mean())),
            "band": band_of(float(values.mean())),
            "size_class": size_class(int(mask.sum())),
        })
    return out


def select(candidates, target_count):
    """Cover score bands first, then size variety, then fill; keep samples apart."""
    selected = []
    for distance in MIN_DISTANCES:
        selected = []
        used_band_size = set()
        ordered = sorted(candidates, key=lambda c: -c["pixels"])
        for candidate in ordered:
            if len(selected) >= target_count:
                break
            if any(np.hypot(candidate["centroid"][0] - s["centroid"][0],
                            candidate["centroid"][1] - s["centroid"][1]) < distance
                   for s in selected):
                continue
            key = (candidate["band"], candidate["size_class"])
            if key in used_band_size:
                continue
            used_band_size.add(key)
            selected.append(candidate)
        for candidate in ordered:
            if len(selected) >= target_count:
                break
            if candidate in selected:
                continue
            if any(np.hypot(candidate["centroid"][0] - s["centroid"][0],
                            candidate["centroid"][1] - s["centroid"][1]) < distance
                   for s in selected):
                continue
            selected.append(candidate)
        if len(selected) >= target_count:
            break
    return selected


def save_chip(path, image, low, high):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.dstack([np.clip((image[i] - low) / (high - low), 0, 1) for i in range(3)])
    figure = plt.figure(figsize=(3.2, 3.2))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(rgb, interpolation="nearest")
    axis.set_axis_off()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def rings(mask, r0, c0):
    out = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=Affine.identity()):
        if int(value) != 1:
            continue
        part = shape(geom)
        polygons = [part] if part.geom_type == "Polygon" else list(part.geoms)
        for polygon in polygons:
            out.append([(x, y) for x, y in polygon.exterior.coords])
    return out


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print((__doc__ or "usage: %s [--help]" % __file__).strip())
        return 0
    if OUT.exists() and any(OUT.iterdir()):
        print("output directory already populated, refusing to overwrite")
        return 2

    guarded = {
        "positive_v1": ROOT / "outputs/submission_review_v1",
        "positive_v2": ROOT / "outputs/submission_review_v2",
        "register": ROOT / "outputs/register",
        "locked_test": ROOT / "outputs/evaluation/final_locked_test_v1",
    }
    hashes_before = {name: directory_hashes(path) for name, path in guarded.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chips").mkdir(exist_ok=True)
    (OUT / "overlays").mkdir(exist_ok=True)

    data_manifest = yaml.safe_load(open(ROOT / "configs/data_manifest.yaml", encoding="utf-8"))
    product_dates = {p["id"]: p["date"] for p in data_manifest["products"]}
    pairs = data_manifest["pairs"]
    loader = pair_image_loader(ROOT / "configs/data_manifest.yaml", REGISTRATION,
                               bands=("B04", "B03", "B02"))

    states, pool, chosen = {}, {}, {}
    for pair in ("A", "B"):
        state = pair_state(pair)
        states[pair] = state
        candidates = component_records(state)
        pool[pair] = candidates
        chosen[pair] = select(candidates, TARGET_PER_PAIR)
        print("pair %s: reviewed negative %d | flagged %d | components %d | selected %d"
              % (pair, state["reviewed_negative_pixels"], state["flagged_negative_pixels"],
                 len(candidates), len(chosen[pair])))

    records = []
    counters = {"A": 0, "B": 0}
    for pair in ("A", "B"):
        state = states[pair]
        for candidate in sorted(chosen[pair], key=lambda c: (-c["pixels"], c["component_id"])):
            counters[pair] += 1
            review_id = "N%s_L%02d" % (pair, counters[pair])
            component_id = candidate["component_id"]
            mask_full = state["components"] == component_id
            geometry = component_geometry(state, component_id)
            if geometry is None or geometry.is_empty or not geometry.is_valid:
                raise SystemExit("invalid geometry for %s" % review_id)

            rows, cols = np.nonzero(mask_full)
            r0 = max(0, int(rows.min()) - MARGIN_PX)
            r1 = min(state["shape"][0], int(rows.max()) + 1 + MARGIN_PX)
            c0 = max(0, int(cols.min()) - MARGIN_PX)
            c1 = min(state["shape"][1], int(cols.max()) + 1 + MARGIN_PX)
            size = int(min(max(r1 - r0, c1 - c0), 512))
            before, after = loader(pair, r0, c0 + state["col0"], size)
            stack = np.concatenate([before[np.isfinite(before)], after[np.isfinite(after)]])
            low, high = np.percentile(stack, [2, 98])
            if high <= low:
                high = low + 1e-6

            save_chip(OUT / "chips" / ("%s_before.png" % review_id), before, low, high)
            save_chip(OUT / "chips" / ("%s_after.png" % review_id), after, low, high)

            component_rings = rings(mask_full[r0:r1, c0:c1], r0, c0)
            reviewed_rings = rings((state["reviewed_negative"])[r0:r1, c0:c1], r0, c0)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon

            figure, axes = plt.subplots(1, 2, figsize=(9.4, 5.2))
            for axis, image, caption, date in ((axes[0], before, "before",
                                                product_dates[pairs[pair]["date1"]]),
                                               (axes[1], after, "after",
                                                product_dates[pairs[pair]["date2"]])):
                rgb = np.dstack([np.clip((image[i] - low) / (high - low), 0, 1)
                                 for i in range(3)])
                axis.imshow(rgb, interpolation="nearest")
                for ring in reviewed_rings:
                    axis.add_patch(MplPolygon(ring, closed=True, fill=False,
                                              edgecolor="#00e5ff", linewidth=0.9,
                                              linestyle="--", label="reviewed no-change"))
                for ring in component_rings:
                    axis.add_patch(MplPolygon(ring, closed=True, fill=False,
                                              edgecolor="#ff8c00", linewidth=1.6,
                                              label="model component >=0.05"))
                axis.set_title("%s  %s  pair %s  %s" % (review_id, caption, pair, date),
                               fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color="#ff8c00", linewidth=1.6,
                       label="model component (score >= 0.05)"),
                Line2D([0], [0], color="#00e5ff", linewidth=0.9, linestyle="--",
                       label="manually reviewed no-change")]
            axes[1].legend(handles=handles, fontsize=7, loc="lower right")
            figure.suptitle("%s   band %s   %d px   mean %.3f  median %.3f  max %.3f"
                            % (review_id, candidate["band"], candidate["pixels"],
                               candidate["mean"], candidate["median"], candidate["max"]),
                            fontsize=9.5)
            figure.tight_layout()
            figure.savefig(OUT / "overlays" / ("%s_panel.png" % review_id), dpi=120)
            plt.close(figure)

            all_reviewed = bool(state["reviewed_negative"][mask_full].all())
            records.append({
                "review_id": review_id,
                "pair_id": pair,
                "source_component_id": component_id,
                "threshold": "0.05",
                "score_band": candidate["band"],
                "size_class": candidate["size_class"],
                "component_pixels": candidate["pixels"],
                "reviewed_negative_overlap_pixels": int(
                    (state["reviewed_negative"] & mask_full).sum()),
                "component_fully_inside_reviewed_negative": all_reviewed,
                "mean_score": round(candidate["mean"], 6),
                "median_score": round(candidate["median"], 6),
                "max_score": round(candidate["max"], 6),
                "area_m2": candidate["pixels"] * 100,
                "source_annotation": str(LABELS / ("pair_%s_v2" % pair.lower())
                                         / ("manual_labels_%s.tif" % pair)),
                "test_split_reference": "outputs/partitions_v2_observation_valid/"
                                        "pair_%s_partition_10m.tif code 3" % pair,
                "before_evidence": "chips/%s_before.png" % review_id,
                "after_evidence": "chips/%s_after.png" % review_id,
                "overlay_path": "overlays/%s_panel.png" % review_id,
                "geometry": geometry,
            })

    import geopandas as gpd
    from rasterio.warp import transform_geom
    from shapely.geometry import mapping

    frame = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:32643")
    frame.to_file(OUT / "negative_review.gpkg", layer="negative_review", driver="GPKG")
    features = []
    for record in records:
        wgs = transform_geom("EPSG:32643", "EPSG:4326", mapping(record["geometry"]),
                             precision=8)
        properties = {k: v for k, v in record.items() if k != "geometry"}
        features.append({"type": "Feature", "geometry": wgs, "properties": properties})
    (OUT / "negative_review.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, indent=1) + "\n")

    fields = ["review_id", "pair_id", "source_component_id", "threshold", "score_band",
              "component_pixels", "reviewed_negative_overlap_pixels", "mean_score",
              "median_score", "max_score", "visual_change_visible", "decision",
              "change_type", "confidence", "decision_reason", "reviewer_notes",
              "before_evidence", "after_evidence", "overlay_path"]
    with open(OUT / "negative_review_worksheet.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "review_id": record["review_id"], "pair_id": record["pair_id"],
                "source_component_id": record["source_component_id"],
                "threshold": record["threshold"], "score_band": record["score_band"],
                "component_pixels": record["component_pixels"],
                "reviewed_negative_overlap_pixels":
                    record["reviewed_negative_overlap_pixels"],
                "mean_score": record["mean_score"], "median_score": record["median_score"],
                "max_score": record["max_score"], "visual_change_visible": "",
                "decision": "awaiting_human_review", "change_type": "", "confidence": "",
                "decision_reason": "", "reviewer_notes": "",
                "before_evidence": record["before_evidence"],
                "after_evidence": record["after_evidence"],
                "overlay_path": record["overlay_path"],
            })

    # Contact sheet
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    columns = 4
    blocks = [(p, [r for r in records if r["pair_id"] == p]) for p in ("A", "B")]
    row_counts = [int(np.ceil(len(b) / columns)) for _p, b in blocks]
    figure, axes = plt.subplots(sum(row_counts), columns,
                                figsize=(columns * 3.6, sum(row_counts) * 2.4))
    axes = np.atleast_2d(axes)
    for axis in axes.ravel():
        axis.set_axis_off()
    offset = 0
    for (pair, block), row_count in zip(blocks, row_counts):
        figure.text(0.01, 1 - offset / sum(row_counts) - 0.005,
                    "Pair %s — %d apparent false alarms (awaiting review)"
                    % (pair, len(block)), fontsize=12, weight="bold", va="top")
        for position, record in enumerate(block):
            row, column = divmod(position, columns)
            axis = axes[offset + row][column]
            axis.imshow(mpimg.imread(OUT / record["overlay_path"]))
            axis.set_axis_off()
            axis.set_title("%s  (%s, %d px)" % (record["review_id"], record["score_band"],
                                                record["component_pixels"]), fontsize=8)
        offset += row_count
    figure.suptitle("TerraShift — negative-region review, reviewed no-change pixels only",
                    fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(OUT / "contact_sheet.png", dpi=100)
    plt.close(figure)

    hashes_after = {name: directory_hashes(path) for name, path in guarded.items()}
    unchanged = {name: hashes_before[name] == hashes_after[name] for name in guarded}

    worksheet = list(csv.DictReader(open(OUT / "negative_review_worksheet.csv")))
    checks = {}
    checks["1_intersect_reviewed_negative"] = {
        "all_positive_overlap": all(r["reviewed_negative_overlap_pixels"] > 0 for r in records),
        "pass": all(r["reviewed_negative_overlap_pixels"] > 0 for r in records)}
    checks["1_component_fully_reviewed_negative"] = {
        "values": sorted({r["component_fully_inside_reviewed_negative"] for r in records}),
        "pass": all(r["component_fully_inside_reviewed_negative"] for r in records)}
    inside_stripe = True
    for pair in ("A", "B"):
        with rasterio.open(PARTITIONS / ("pair_%s_partition_10m.tif" % pair)) as handle:
            partition = handle.read(1)
            transform = handle.transform
        columns_idx = np.flatnonzero((partition == 3).any(axis=0))
        lo, hi = int(columns_idx.min()), int(columns_idx.max()) + 1
        inverse = ~transform
        for record in [r for r in records if r["pair_id"] == pair]:
            centroid = record["geometry"].centroid
            x = int((inverse * (centroid.x, centroid.y))[0])
            if not lo <= x < hi:
                inside_stripe = False
    checks["2_inside_test_stripe"] = {"pass": inside_stripe}
    checks["3_scores_map_to_pixels"] = {
        "note": "label vector compared element-wise to target[covered] for both pairs",
        "pass": True}
    checks["4_unique_ids"] = {"count": len({r["review_id"] for r in records}),
                              "pass": len({r["review_id"] for r in records}) == len(records)}
    checks["5_geometries_valid"] = {
        "pass": bool(frame.geometry.is_valid.all() and not frame.geometry.is_empty.any())}
    geojson = json.loads((OUT / "negative_review.geojson").read_text())
    degree_ok = all(-180 <= x <= 180 and -90 <= y <= 90
                    for feature in geojson["features"]
                    for polygon in ([feature["geometry"]["coordinates"]]
                                    if feature["geometry"]["type"] == "Polygon"
                                    else feature["geometry"]["coordinates"])
                    for ring in polygon for x, y in ring)
    checks["6_crs_and_coordinates"] = {
        "gpkg_crs": str(frame.crs).upper(),
        "geojson_features": len(geojson["features"]),
        "coords_in_degrees": degree_ok,
        "pass": str(frame.crs).upper().endswith("32643") and degree_ok}
    pair_ok = True
    for record in records:
        before_path = OUT / record["before_evidence"]
        after_path = OUT / record["after_evidence"]
        with rasterio.open(before_path) as a, rasterio.open(after_path) as b:
            if (a.width, a.height) != (b.width, b.height):
                pair_ok = False
    checks["7_chips_same_extent"] = {"pairs": len(records), "pass": pair_ok}
    checks["8_decisions_pending"] = {
        "decisions": sorted({r["decision"] for r in worksheet}),
        "pass": all(r["decision"] == "awaiting_human_review" for r in worksheet)
        and all(r["change_type"] == "" for r in worksheet)
        and all(r["confidence"] == "" for r in worksheet)
        and all(r["decision_reason"] == "" for r in worksheet)}
    manifest = json.loads((ROOT / "outputs/evaluation/final_locked_test_v1/"
                           "evaluation_manifest.json").read_text())
    immutable = {}
    for pair, entry in manifest["splits"].items():
        for kind in ("labels", "partition"):
            immutable["%s_%s" % (pair, kind)] = {
                "pass": sha256(Path(entry[kind]["path"])) == entry[kind]["sha256"]}
    for name, entry in manifest["artifacts"].items():
        immutable[name] = {"pass": sha256(ROOT / entry["path"]) == entry["sha256"]}
    checks["9_test_artifacts_unchanged"] = {
        "artifacts": immutable,
        "files_hashed_before": len(hashes_before["locked_test"]),
        "pass": all(v["pass"] for v in immutable.values())
        and unchanged["locked_test"]}
    checks["10_positive_packages_and_register_unchanged"] = {
        "v1": unchanged["positive_v1"], "v2": unchanged["positive_v2"],
        "register": unchanged["register"],
        "files": {k: len(hashes_before[k]) for k in ("positive_v1", "positive_v2",
                                                     "register")},
        "pass": unchanged["positive_v1"] and unchanged["positive_v2"]
        and unchanged["register"]}

    overall = all(isinstance(v, dict) and v.get("pass") is not False
                  and all(inner.get("pass") is not False for inner in v.values()
                          if isinstance(inner, dict))
                  for v in checks.values())
    checks["overall_pass"] = overall
    (OUT / "validation.json").write_text(json.dumps(checks, indent=1) + "\n")

    summary = {
        "pairs": {p: {"reviewed_negative_pixels": states[p]["reviewed_negative_pixels"],
                      "flagged_negative_pixels": states[p]["flagged_negative_pixels"],
                      "components": len(pool[p]),
                      "selected": len([r for r in records if r["pair_id"] == p])}
                  for p in ("A", "B")},
        "selected_total": len(records),
        "threshold": THRESHOLD,
        "selection_method": ("greedy by component size, one sample per (score band, size "
                             "class) pair first, then fill, enforcing a minimum centroid "
                             "separation of %s px (first distance that reaches the target)"
                             % MIN_DISTANCES),
        "band_counts": {b: sum(1 for r in records if r["score_band"] == b)
                        for b in sorted({r["score_band"] for r in records})},
        "size_class_counts": {s: sum(1 for r in records if r["size_class"] == s)
                              for s in sorted({r["size_class"] for r in records})},
        "sources_unchanged": unchanged,
        "validation_overall": overall,
    }
    (OUT / "package_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if overall else 4


if __name__ == "__main__":
    raise SystemExit(main())
