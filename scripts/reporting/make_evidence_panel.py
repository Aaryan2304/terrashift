from pathlib import Path
"""Build an evidence panel for one reviewed polygon.

A panel shows the same ground area four ways:

1. date 1 true colour,
2. date 2 true colour,
3. the change score on its own colour scale,
4. date 2 with the candidate mask.

Every view shares one map extent, so the overlays line up. The contrast stretch
is computed once from the two date crops, so a real brightness change between
the dates stays visible instead of being normalised away.

The polygon comes from the detection register, which is human-authored. This
script renders what a reviewer decided; it does not decide anything. A panel
for an unreviewed candidate is not evidence of change.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.manifest import load_manifest, pair_paths
from terrashift.data.safeloader import SafeProduct
from terrashift.geospatial.imagery import RGB_BANDS, stretch_pair
from terrashift.geospatial.rendering import _window_bounds, read_rgb_window, render_panel, window_for_bounds, polygon_rings, world_to_pixel, read_register_row

DEFAULT_SCORE_DIR = os.path.join(ROOT, "outputs", "baseline_v2_observation_valid")
DEFAULT_OUT_DIR = os.path.join(ROOT, "outputs", "figures", "evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["A", "B"], required=True)
    parser.add_argument("--register", required=True, help="detection register CSV")
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--score-dir", default=DEFAULT_SCORE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--margin-px", type=int, default=24,
                        help="context around the polygon, in score-grid pixels (20 m)")
    parser.add_argument("--percentile", type=float, default=99.9,
                        help="percentile that defines the candidate overlay")
    parser.add_argument("--status", default="rendered_from_register_row",
                        help="status tag written into the panel JSON; use "
                             "smoke_test_not_a_detection for pipeline checks")
    args = parser.parse_args()

    row = read_register_row(args.register, args.record_id)
    if row.get("pair") and row["pair"] != args.pair:
        parser.error("record %s belongs to pair %s" % (args.record_id, row["pair"]))
    if not row.get("polygon_wkt"):
        parser.error("record %s has no polygon_wkt" % args.record_id)

    _, bounds = polygon_rings(row["polygon_wkt"])

    score_path = os.path.join(args.score_dir, "change_score_%s.tif" % args.pair)
    if not os.path.isfile(score_path):
        parser.error("score raster not found: %s" % score_path)
    with rasterio.open(score_path) as dataset:
        score = dataset.read(1, masked=True).filled(np.nan)
        score_transform = dataset.transform
        score_crs = dataset.crs
        score_shape = dataset.shape

    valid = np.isfinite(score)
    threshold = float(np.percentile(score[valid], args.percentile))

    crop = window_for_bounds(score_transform, bounds, score_shape, args.margin_px)
    rows = (crop[0], crop[2])
    cols = (crop[1], crop[3])
    score_crop = score[rows[0]:rows[1], cols[0]:cols[1]]
    candidate = (score_crop > threshold) & np.isfinite(score_crop)
    extent = _window_bounds(score_transform, rows, cols)

    manifest = load_manifest()
    first_path, second_path = pair_paths(manifest, args.pair)
    first = SafeProduct(first_path)
    second = SafeProduct(second_path)

    with rasterio.open(first.band_files["B04"]) as dataset:
        native_transform = dataset.transform
        native_shape = dataset.shape
    native_crop = window_for_bounds(native_transform, extent, native_shape)
    native_window = rasterio.windows.Window(
        native_crop[1], native_crop[0],
        native_crop[3] - native_crop[1], native_crop[2] - native_crop[0])

    before = read_rgb_window(first, RGB_BANDS, native_window)
    after = read_rgb_window(second, RGB_BANDS, native_window)
    before, after = stretch_pair(before, after)

    label = "%s / %s" % (args.record_id, row.get("change_type", "unknown"))
    title = "%s — %s" % (args.pair, label)
    footer = ("register %s | pair %s | score %s | crs %s | rows %d:%d cols %d:%d | "
              "B04/B03/B02 | candidate = top %.1f%% of score, threshold %.4f"
              % (os.path.basename(args.register), args.pair,
                 os.path.basename(args.score_dir), score_crs,
                 rows[0], rows[1], cols[0], cols[1],
                 100 - args.percentile, threshold))

    views = [
        {"image": before, "label": "%s — date 1" % label},
        {"image": after, "label": "%s — date 2" % label},
        {"image": np.where(np.isfinite(score_crop), score_crop, np.nan),
         "label": "change score", "cmap": "viridis", "colorbar": True},
        {"image": after, "label": "date 2, candidates in red", "overlay": candidate},
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, "panel_%s_%s.png" % (args.pair, args.record_id))
    out_json = os.path.join(args.output_dir, "panel_%s_%s.json" % (args.pair, args.record_id))
    for path in (out_png, out_json):
        if os.path.exists(path):
            parser.error("refusing to overwrite %s" % path)

    render_panel(views, out_png, title, footer, extent)
    with open(out_json, "w") as handle:
        json.dump({
            "record_id": args.record_id, "pair": args.pair,
            "register": os.path.abspath(args.register),
            "score_raster": os.path.abspath(score_path),
            "crop_window_rows": list(rows), "crop_window_cols": list(cols),
            "crop_bounds": [float(v) for v in extent],
            "native_window": [native_window.col_off, native_window.row_off,
                              native_window.width, native_window.height],
            "candidate_percentile": args.percentile,
            "candidate_threshold": threshold,
            "band_order": list(RGB_BANDS),
            "crs": str(score_crs),
            "status": args.status,
            "note": ("the red overlay marks the score tail, not detected change; "
                     "the polygon is the reviewer's, not this script's"),
        }, handle, indent=1)
    print("wrote", out_png)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
