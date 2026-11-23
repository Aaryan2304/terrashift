"""Candidate masks, component inventories and volume statistics from saved score rasters.

Every mask is derived from a saved continuous score raster. No model is run here, so the
three thresholds cannot disagree about the underlying scores.

    python scripts/evaluation/full_scene_candidate_analysis.py --output <dir> --stage masks
    python scripts/evaluation/full_scene_candidate_analysis.py --output <dir> --stage verify

Thresholds are the pipeline's, not a selection: the summary reports all three and the
manifest records that the operating threshold remains unlocked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]

PAIRS = ("A", "B")
THRESHOLDS = (0.03, 0.05, 0.10)
STRUCTURE = np.ones((3, 3), dtype=np.uint8)      # eight-connected
CHUNK = 40000
NODATA = -9999.0


def suffix(threshold):
    return "%03d" % int(round(threshold * 100))


def paths(out, pair):
    return {"scores": out / "rasters" / ("pair_%s_scores.tif" % pair),
            "valid": out / "rasters" / ("pair_%s_valid_mask.tif" % pair)}


def mask_path(out, pair, threshold):
    return out / "rasters" / ("pair_%s_candidates_%s.tif" % (pair, suffix(threshold)))


def inventory_path(out, pair, threshold):
    return out / "components" / ("pair_%s_component_inventory_%s.csv"
                                 % (pair, suffix(threshold)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_mb():
    info = open("/proc/meminfo").read()
    return (int(info.split("MemAvailable:")[1].split()[0]) // 1024,
            int(info.split("MemTotal:")[1].split()[0]) // 1024)


def guard(stage, floor_mb=1200):
    available, total = memory_mb()
    print("[mem] %-34s available %5d MB / %d MB" % (stage, available, total))
    if available < floor_mb:
        raise SystemExit("stage %s: MemAvailable %d MB below %d MB floor"
                         % (stage, available, floor_mb))


def load_pair(out, pair):
    """Scores with invalid pixels set to 0, plus the validity mask and grid metadata."""
    with rasterio.open(paths(out, pair)["scores"]) as handle:
        scores = handle.read(1)
        grid = {"crs": str(handle.crs), "width": handle.width, "height": handle.height,
                "transform": [float(v) for v in list(handle.transform)[:6]],
                "nodata": handle.nodata, "dtype": handle.dtypes[0]}
    with rasterio.open(paths(out, pair)["valid"]) as handle:
        valid = handle.read(1).astype(bool)
        grid["valid_width"], grid["valid_height"] = handle.width, handle.height
    nodata_mask = scores == NODATA
    if not np.array_equal(nodata_mask, ~valid):
        raise SystemExit("%s: score NoData and validity mask disagree (%d vs %d)"
                         % (pair, int(nodata_mask.sum()), int((~valid).sum())))
    pixel_x = abs(grid["transform"][0])
    pixel_y = abs(grid["transform"][4])
    grid["pixel_size_m"] = [pixel_x, pixel_y]
    grid["pixel_area_km2"] = pixel_x * pixel_y / 1e6
    scores = np.where(valid, scores, 0.0).astype(np.float32)
    return scores, valid, grid


def component_table(labels, scores, valid, n, grid, threshold):
    """One row per component, plus quantiles of the size distribution."""
    rows = []
    if n == 0:
        return rows, {}
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes = sizes[1:].astype(np.int64)
    sums = np.zeros(n, dtype=np.float64)
    maxima = np.zeros(n, dtype=np.float32)
    centroids = np.zeros((n, 2), dtype=np.float64)
    for start in range(0, n, CHUNK):
        index = np.arange(start + 1, min(start + CHUNK, n) + 1)
        sums[start:start + len(index)] = ndi.sum(scores, labels, index)
        maxima[start:start + len(index)] = ndi.maximum(scores, labels, index)
        centroids[start:start + len(index)] = np.asarray(
            ndi.center_of_mass(valid, labels, index))
    objects = ndi.find_objects(labels)
    transform = Affine(*grid["transform"])
    mean_scores = np.divide(sums, sizes, out=np.zeros(n), where=sizes > 0)
    order = np.argsort(-sizes)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    for i in range(n):
        sl = objects[i]
        row0, row1 = int(sl[0].start), int(sl[0].stop)
        col0, col1 = int(sl[1].start), int(sl[1].stop)
        left, top = transform * (col0, row0)
        right, bottom = transform * (col1, row1)
        centre_row, centre_col = centroids[i]
        centre_x, centre_y = transform * (centre_col, centre_row)
        rows.append({
            "pair": grid["pair"], "threshold": threshold,
            "component_id": i + 1, "rank_by_area": int(rank[i]),
            "pixels": int(sizes[i]),
            "area_km2": round(float(sizes[i]) * grid["pixel_area_km2"], 6),
            "row_min": row0, "row_max": row1 - 1, "col_min": col0, "col_max": col1 - 1,
            "bbox_left": round(left, 2), "bbox_bottom": round(bottom, 2),
            "bbox_right": round(right, 2), "bbox_top": round(top, 2),
            "centroid_row": round(float(centre_row), 3),
            "centroid_col": round(float(centre_col), 3),
            "centroid_x": round(centre_x, 2), "centroid_y": round(centre_y, 2),
            "mean_score": round(float(mean_scores[i]), 6),
            "max_score": round(float(maxima[i]), 6),
        })
    quantiles = {}
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00):
        quantiles["size_p%d" % int(q * 100)] = int(np.quantile(sizes, q))
    quantiles["size_mean"] = round(float(sizes.mean()), 2)
    quantiles["size_median"] = int(np.median(sizes))
    return rows, quantiles


def stage_masks(out, args):
    guard("masks")
    thresholds = tuple(float(v) for v in args.thresholds.split(","))
    summary = []
    for pair in PAIRS:
        scores, valid, grid = load_pair(out, pair)
        grid["pair"] = pair
        valid_pixels = int(valid.sum())
        total_pixels = int(valid.size)
        print("[%s] %dx%d, valid %d (%.4f%%), pixel %.1f m"
              % (pair, grid["width"], grid["height"], valid_pixels,
                 100.0 * valid_pixels / total_pixels, grid["pixel_size_m"][0]))
        for threshold in thresholds:
            guard("masks %s %.2f" % (pair, threshold))
            mask = (scores >= threshold) & valid
            labels, n = ndi.label(mask, structure=STRUCTURE)
            rows, quantiles = component_table(labels, scores, valid, n, grid, threshold)
            with open(inventory_path(out, pair, threshold), "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                        ["pair", "threshold", "component_id"])
                writer.writeheader()
                writer.writerows(rows)
            profile = {"driver": "GTiff", "height": grid["height"], "width": grid["width"],
                       "count": 1, "dtype": "uint8", "crs": grid["crs"],
                       "transform": Affine(*grid["transform"]),
                       "nodata": 255, "tiled": True, "compress": "deflate"}
            with rasterio.open(mask_path(out, pair, threshold), "w", **profile) as dst:
                dst.write(np.where(valid, mask.astype(np.uint8), np.uint8(255)), 1)
            sizes = np.bincount(labels.ravel())[1:] if n else np.array([], dtype=np.int64)
            candidate_pixels = int(mask.sum())
            inventory_total = int(sizes.sum())
            if candidate_pixels != inventory_total:
                raise SystemExit("%s %.2f: mask %d != inventory %d"
                                 % (pair, threshold, candidate_pixels, inventory_total))
            entry = {
                "pair": pair, "threshold": threshold,
                "candidate_pixels": candidate_pixels,
                "valid_pixels": valid_pixels, "total_pixels": total_pixels,
                "candidate_fraction_of_valid": round(candidate_pixels / valid_pixels, 6),
                "candidate_area_km2": round(candidate_pixels * grid["pixel_area_km2"], 4),
                "valid_area_km2": round(valid_pixels * grid["pixel_area_km2"], 4),
                "components": int(n),
                "largest_component_pixels": int(sizes.max()) if n else 0,
                "largest_component_km2": round(float(sizes.max()) * grid["pixel_area_km2"], 6)
                if n else 0.0,
                "mean_component_pixels": round(float(sizes.mean()), 2) if n else 0.0,
                "median_component_pixels": int(np.median(sizes)) if n else 0,
                "mean_component_km2": round(float(sizes.mean()) * grid["pixel_area_km2"], 6)
                if n else 0.0,
                "median_component_km2": round(float(np.median(sizes)) * grid["pixel_area_km2"], 6)
                if n else 0.0,
                "components_ge_1km2": int((sizes * grid["pixel_area_km2"] >= 1.0).sum()),
                "components_ge_10px": int((sizes >= 10).sum()),
                "components_1px": int((sizes == 1).sum()),
            }
            entry.update({k: v for k, v in quantiles.items()})
            summary.append(entry)
            print("  %.2f: %10d px (%7.4f%%) %10.2f km2 | %6d comps | largest %8d px "
                  "| median %5d"
                  % (threshold, candidate_pixels,
                     100.0 * candidate_pixels / valid_pixels, entry["candidate_area_km2"],
                     n, entry["largest_component_pixels"],
                     entry["median_component_pixels"]))
            del mask, labels
        del scores, valid
    fields = list(summary[0])
    for row in summary[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    name = "candidate_volume_summary.csv" if args.thresholds == "0.03,0.05,0.1" \
        else "candidate_volume_supplementary.csv"
    with open(out / name, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(summary)
    (out / name.replace(".csv", ".json")).write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote", name, "(%d rows)" % len(summary))


def stage_verify(out, args):
    """Reproduce every mask from its scores and cross-check the inventory."""
    guard("verify")
    thresholds = tuple(float(v) for v in args.thresholds.split(","))
    summary = {}
    for name in ("candidate_volume_summary.json",
                 "candidate_volume_supplementary.json"):
        path = out / name
        if path.exists():
            summary.update({(row["pair"], row["threshold"]): row
                            for row in json.loads(path.read_text())})
    checks = {}
    for pair in PAIRS:
        with rasterio.open(paths(out, pair)["scores"]) as handle:
            scores = handle.read(1)
            grid = {"crs": str(handle.crs), "width": handle.width, "height": handle.height,
                    "transform": [float(v) for v in list(handle.transform)[:6]]}
            tags = handle.tags()
        with rasterio.open(paths(out, pair)["valid"]) as handle:
            valid = handle.read(1).astype(bool)
        pixel_area = abs(grid["transform"][0]) * abs(grid["transform"][4]) / 1e6
        for threshold in thresholds:
            with rasterio.open(mask_path(out, pair, threshold)) as handle:
                stored = handle.read(1)
                mask_grid = {"crs": str(handle.crs), "transform": list(handle.transform)[:6],
                             "height": handle.height, "width": handle.width,
                             "nodata": handle.nodata}
            expected = np.where(valid, ((scores >= threshold) & valid).astype(np.uint8),
                                np.uint8(255))
            rows = list(csv.DictReader(open(inventory_path(out, pair, threshold))))
            inventory_total = sum(int(r["pixels"]) for r in rows)
            count = int((stored == 1).sum())
            entry = {
                "mask_reproduced_from_scores": bool(np.array_equal(stored, expected)),
                "mask_pixels": count,
                "inventory_total_pixels": inventory_total,
                "mask_matches_inventory": count == inventory_total,
                "components_in_inventory": len(rows),
                "components_in_summary": int(summary[(pair, float(threshold))]["components"]),
                "area_consistent_km2": abs(
                    count * pixel_area - float(summary[(pair, float(threshold))]["candidate_area_km2"])
                ) < 1e-3,
                "inventory_area_sum_km2": round(
                    sum(float(r["area_km2"]) for r in rows), 4),
                "grid_matches_scores": (
                    mask_grid["crs"] == grid["crs"]
                    and mask_grid["height"] == grid["height"]
                    and mask_grid["width"] == grid["width"]
                    and all(abs(a - b) < 1e-9 for a, b, in
                            zip(mask_grid["transform"], grid["transform"]))),
                "nodata_is_255": float(mask_grid["nodata"]) == 255.0,
                "no_invalid_pixel_predicted": bool(np.all(stored[~valid] == 255)),
            }
            failed = [k for k, v in entry.items() if isinstance(v, bool) and not v]
            if entry["components_in_inventory"] != entry["components_in_summary"]:
                failed.append("component_count_disagrees_with_summary")
            if entry["inventory_area_sum_km2"] and not entry["area_consistent_km2"]:
                failed.append("area_inconsistent")
            checks["%s_%.2f" % (pair, threshold)] = entry
            status = "ok" if not failed else "FAIL %s" % failed
            print("[%s %.2f] %s | mask %d, inventory %d, %d components, %.4f km2"
                  % (pair, threshold, status, count, inventory_total, len(rows),
                     sum(float(r["area_km2"]) for r in rows)))
            if failed:
                raise SystemExit("verification failed for %s %.2f: %s"
                                 % (pair, threshold, failed))
        checks["%s_score_tags" % pair] = tags
        del scores, valid
    target = "verification_masks.json" if args.thresholds == "0.03,0.05,0.1" \
        else "verification_masks_supplementary.json"
    (out / target).write_text(json.dumps(checks, indent=1) + "\n")
    print("wrote", target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True, choices=["masks", "verify"])
    parser.add_argument("--thresholds", default="0.03,0.05,0.1")
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_dir():
        raise SystemExit("output directory does not exist: %s" % out)
    {"masks": stage_masks, "verify": stage_verify}[args.stage](out, args)


if __name__ == "__main__":
    main()
