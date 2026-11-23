"""Render one evidence panel per review candidate.

Four views of the same ground: date 1 true colour, date 2 true colour, the change score, and date 2
with the candidate outline. All share one extent and stretch, so a real brightness change survives.
These support manual review and are not detections: the polygon is the score tail, no class is
implied, and every file carries ``candidate_evidence_not_a_detection``.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import csv
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.manifest import load_manifest, pair_paths
from terrashift.data.safeloader import SafeProduct
from terrashift.geospatial.imagery import RGB_BANDS, stretch_pair
from terrashift.geospatial.rendering import _window_bounds, read_rgb_window, render_panel, window_for_bounds

STATUS = "candidate_evidence_not_a_detection"


def partition_lookup(partitions, target_crs):
    """Split polygons and guard bands, with a function to classify a point."""
    layers = partitions.to_crs(target_crs)
    pieces = {row["split"]: row.geometry for _, row in layers.iterrows()}
    order = sorted(pieces, key=lambda name: pieces[name].bounds[0])
    guards = []
    for left, right in zip(order, order[1:]):
        gap_start = pieces[left].bounds[2]
        gap_stop = pieces[right].bounds[0]
        if gap_stop > gap_start:
            low = min(pieces[left].bounds[1], pieces[right].bounds[1])
            high = max(pieces[left].bounds[3], pieces[right].bounds[3])
            guards.append((box(gap_start, low, gap_stop, high),
                           "guard_%s_%s" % (left, right)))
    return pieces, guards


def classify(geometry, pieces, guards):
    """Partition of a polygon, plus a flag when it is not wholly inside one."""
    overlaps = {name: geometry.intersection(piece).area
                for name, piece in pieces.items()}
    overlaps = {name: area for name, area in overlaps.items() if area > 0}
    guard_areas = {label: geometry.intersection(band).area
                   for band, label in guards}
    guard_areas = {label: area for label, area in guard_areas.items() if area > 0}

    if guard_areas and not overlaps:
        return max(guard_areas, key=guard_areas.get), "inside_guard_band"
    if not overlaps:
        return "outside_partitions", "outside_partitions"
    dominant = max(overlaps, key=overlaps.get)
    if len(overlaps) > 1 or guard_areas:
        return dominant, "crosses_partition_boundary"
    return dominant, "single_partition"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["A", "B"], default="B")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--partitions", required=True)
    parser.add_argument("--score-dir",
                        default=os.path.join(ROOT, "outputs", "baseline_v2_observation_valid"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin-px", type=int, default=40,
                        help="context around the candidate, in 20 m score pixels")
    parser.add_argument("--limit", type=int, default=None,
                        help="render only the first N candidates, for a smoke check")
    args = parser.parse_args()

    import geopandas as gpd

    candidates = gpd.read_file(args.candidates)
    partitions = gpd.read_file(args.partitions)
    candidates = candidates.to_crs(32643)
    pieces, guards = partition_lookup(partitions, candidates.crs)
    print("candidates:", len(candidates))

    os.makedirs(args.output_dir, exist_ok=True)

    score_path = os.path.join(args.score_dir, "change_score_%s.tif" % args.pair)
    if not os.path.isfile(score_path):
        raise SystemExit("score raster not found: %s" % score_path)

    manifest = load_manifest()
    first_path, second_path = pair_paths(manifest, args.pair)
    first, second = SafeProduct(first_path), SafeProduct(second_path)

    index_rows = []
    rendered = 0
    with rasterio.open(score_path) as score_dataset:
        score = score_dataset.read(1, masked=True).filled(np.nan)
        score_transform = score_dataset.transform
        score_shape = score_dataset.shape
        score_crs = score_dataset.crs
        finite = np.isfinite(score)

        with rasterio.open(first.band_files["B04"]) as dataset:
            native_transform = dataset.transform
            native_shape = dataset.shape

        for counter, (_, row) in enumerate(candidates.iterrows()):
            if args.limit is not None and counter >= args.limit:
                break
            candidate_id = row["candidate_id"]
            geometry = row.geometry
            if geometry is None or geometry.is_empty:
                index_rows.append({"candidate_id": candidate_id, "status": "skipped_empty_geometry"})
                continue

            split, placement = classify(geometry, pieces, guards)
            crop = window_for_bounds(score_transform, geometry.bounds,
                                     score_shape, args.margin_px, minimum_px=16)
            rows = (crop[0], crop[2])
            cols = (crop[1], crop[3])
            score_crop = score[rows[0]:rows[1], cols[0]:cols[1]]
            extent = _window_bounds(score_transform, rows, cols)

            native_crop = window_for_bounds(native_transform, extent, native_shape,
                                            margin_px=0, minimum_px=8)
            native_window = rasterio.windows.Window(
                native_crop[1], native_crop[0],
                native_crop[3] - native_crop[1], native_crop[2] - native_crop[0])

            before = read_rgb_window(first, RGB_BANDS, native_window)
            after = read_rgb_window(second, RGB_BANDS, native_window)
            before, after = stretch_pair(before, after)

            # Outline the candidate itself, rather than re-thresholding the score:
            # the panel then shows exactly the polygon the review queue refers to.
            # The interior stays visible, because a filled polygon hides the
            # feature the reviewer needs to look at through it.
            filled = rasterize([(geometry.__geo_interface__, 1)],
                               out_shape=(rows[1] - rows[0], cols[1] - cols[0]),
                               transform=rasterio.windows.transform(
                                   rasterio.windows.Window(cols[0], rows[0],
                                                           cols[1] - cols[0],
                                                           rows[1] - rows[0]),
                                   score_transform),
                               fill=0, dtype="uint8").astype(bool)
            outline = filled & ~ndi.binary_erosion(
                filled, structure=np.ones((3, 3), dtype=bool), iterations=1,
                border_value=0)
            if not outline.any():
                outline = filled          # a polygon thinner than the erosion
            overlay = outline & finite[rows[0]:rows[1], cols[0]:cols[1]]

            label = "%s (%s)" % (candidate_id, split)
            title = ("Pair %s — candidate %s — %s — CANDIDATE, NOT A DETECTION"
                     % (args.pair, candidate_id, split))
            footer = ("candidate %s | partition %s (%s) | area %d px at 20 m | "
                      "mean score %.4f | crop rows %d:%d cols %d:%d | %s"
                      % (candidate_id, split, placement,
                         int(row.get("area_px_20m") or 0),
                         float(row.get("mean_score") or 0.0),
                         rows[0], rows[1], cols[0], cols[1],
                         "stretch is shared between the two dates within this panel"))

            views = [
                {"image": before, "label": "%s — date 1" % label},
                {"image": after, "label": "%s — date 2" % label},
                {"image": np.where(np.isfinite(score_crop), score_crop, np.nan),
                 "label": "%s — change score" % candidate_id, "cmap": "viridis",
                 "colorbar": True},
                {"image": after, "label": "%s — date 2, candidate in red" % candidate_id,
                 "overlay": overlay},
            ]

            png = os.path.join(args.output_dir, "panel_%s_%s.png" % (args.pair, candidate_id))
            meta = os.path.join(args.output_dir, "panel_%s_%s.json" % (args.pair, candidate_id))
            for path in (png, meta):
                if os.path.exists(path):
                    raise SystemExit("refusing to overwrite %s" % path)

            render_panel(views, png, title, footer, extent)
            with open(meta, "w") as handle:
                json.dump({
                    "candidate_id": candidate_id,
                    "pair": args.pair,
                    "partition": split,
                    "partition_placement": placement,
                    "candidate_area_px_20m": row.get("area_px_20m"),
                    "candidate_mean_score": row.get("mean_score"),
                    "candidate_max_score": row.get("max_score"),
                    "score_raster": os.path.abspath(score_path),
                    "crop_window_rows": list(rows),
                    "crop_window_cols": list(cols),
                    "crop_bounds": [float(v) for v in extent],
                    "native_window": [native_window.col_off, native_window.row_off,
                                      native_window.width, native_window.height],
                    "band_order": list(RGB_BANDS),
                    "crs": str(score_crs),
                    "status": STATUS,
                    "note": ("the red outline is the review candidate, not a "
                             "detected or confirmed change; no class is assigned"),
                }, handle, indent=1)

            index_rows.append({
                "candidate_id": candidate_id, "partition": split,
                "partition_placement": placement,
                "area_px_20m": row.get("area_px_20m"),
                "mean_score": row.get("mean_score"),
                "rows": "%d:%d" % rows, "cols": "%d:%d" % cols,
                "png": os.path.basename(png), "json": os.path.basename(meta),
                "status": STATUS,
            })
            rendered += 1
            if rendered % 25 == 0:
                print("  rendered", rendered)

    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, "candidate_panel_index.csv")
    if os.path.exists(index_path):
        raise SystemExit("refusing to overwrite %s" % index_path)
    fields = ["candidate_id", "partition", "partition_placement", "area_px_20m",
              "mean_score", "rows", "cols", "png", "json", "status"]
    with open(index_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in index_rows:
            writer.writerow({key: record.get(key, "") for key in fields})

    counts = {}
    for record in index_rows:
        counts[record.get("partition", "unknown")] = counts.get(
            record.get("partition", "unknown"), 0) + 1
    print("\nrendered %d panels" % rendered)
    print("by partition:", json.dumps(counts, sort_keys=True))
    print("wrote", index_path)


if __name__ == "__main__":
    main()
