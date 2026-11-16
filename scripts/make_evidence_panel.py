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
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.manifest import load_manifest, pair_paths
from data.safeloader import SafeProduct
from utils.imagery import RGB_BANDS, stretch_pair

DEFAULT_SCORE_DIR = os.path.join(ROOT, "outputs", "baseline_v2_observation_valid")
DEFAULT_OUT_DIR = os.path.join(ROOT, "outputs", "figures", "evidence")


def world_to_pixel(transform, x, y):
    """Column and row of a map coordinate, as floats.

    The inverse affine maps map units to pixel units continuously. Plot
    coordinates use these values directly, so no half-pixel offset is applied.
    """
    col, row = (~transform) * (x, y)
    return float(col), float(row)


def window_for_bounds(transform, bounds, shape, margin_px=0, minimum_px=32):
    """Row and column window covering a map bounding box, with a pixel margin.

    Returns ``(row0, col0, row1, col1)`` clipped to the raster. Raises
    ``ValueError`` when the box misses the raster, because a silently empty
    panel is worse than a failure.
    """
    left, bottom, right, top = bounds
    if right <= left or top <= bottom:
        raise ValueError("bounds are empty: %r" % (bounds,))
    col0, row0 = world_to_pixel(transform, left, top)
    col1, row1 = world_to_pixel(transform, right, bottom)
    col0, col1 = sorted((col0, col1))
    row0, row1 = sorted((row0, row1))

    height, width = shape
    col0 = max(0, int(np.floor(col0)) - margin_px)
    row0 = max(0, int(np.floor(row0)) - margin_px)
    col1 = min(width, int(np.ceil(col1)) + margin_px)
    row1 = min(height, int(np.ceil(row1)) + margin_px)

    if col1 <= col0 or row1 <= row0:
        raise ValueError("bounds %r fall outside the raster" % (bounds,))
    if col1 - col0 < minimum_px or row1 - row0 < minimum_px:
        raise ValueError(
            "panel would be %d x %d px, below the %d px minimum; increase "
            "--margin-px" % (col1 - col0, row1 - row0, minimum_px))
    return row0, col0, row1, col1


def polygon_rings(wkt):
    """Exterior ring and bounds of a WKT polygon, in the raster's CRS units."""
    from shapely import wkt as shapely_wkt

    geometry = shapely_wkt.loads(wkt)
    if geometry.geom_type != "Polygon":
        raise ValueError("expected a Polygon, got %s" % geometry.geom_type)
    if geometry.is_empty:
        raise ValueError("polygon is empty")
    return np.asarray(geometry.exterior.coords), geometry.bounds


def read_register_row(path, record_id):
    """One row of the detection register, as a dict."""
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("record_id") == record_id:
                return row
    raise KeyError("record_id %r is not in %s" % (record_id, path))


def read_rgb_window(product, bands, window):
    """RGB stack for one window of a product, at the bands' native resolution."""
    channels = []
    for band in bands:
        with rasterio.open(product.band_files[band]) as dataset:
            channels.append(dataset.read(1, window=window).astype(np.float32))
    if len({c.shape for c in channels}) != 1:
        raise ValueError("bands in %r do not share a shape" % (bands,))
    return np.stack(channels, axis=-1)


def _window_bounds(transform, row, col):
    """Map bounds of a row/column window, as ``(left, bottom, right, top)``."""
    left = transform.c + col[0] * transform.a
    right = transform.c + col[1] * transform.a
    top = transform.f + row[0] * transform.e
    bottom = transform.f + row[1] * transform.e
    return (min(left, right), min(bottom, top), max(left, right), max(bottom, top))


def render_panel(views, out_path, title, footer, extent):
    """Draw a 2 x 2 panel figure and write it.

    ``views`` holds four dictionaries with keys ``image``, ``label``, ``cmap``,
    ``vmin``, ``vmax`` and optional ``overlay`` (a boolean mask drawn in red).
    ``extent`` is ``(left, right, bottom, top)`` in map units, shared by all
    four panels.
    """
    if len(views) != 4:
        raise ValueError("a panel takes exactly four views")
    figure, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    for axis, view in zip(axes.ravel(), views):
        shown = axis.imshow(view["image"], extent=extent, cmap=view.get("cmap"),
                            vmin=view.get("vmin"), vmax=view.get("vmax"),
                            origin="upper", interpolation="nearest")
        overlay = view.get("overlay")
        if overlay is not None and np.asarray(overlay).any():
            shaped = np.ma.masked_where(~np.asarray(overlay, dtype=bool),
                                        np.ones(np.shape(overlay), dtype=float))
            axis.imshow(shaped, extent=extent, cmap="autumn", alpha=0.45,
                        vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
        axis.set_title(view["label"])
        axis.set_xticks([])
        axis.set_yticks([])
        if view.get("colorbar"):
            figure.colorbar(shown, ax=axis, shrink=0.75)
    figure.suptitle(title)
    figure.text(0.01, 0.002, footer, fontsize=6.5, va="bottom", family="monospace")
    figure.savefig(out_path, dpi=110)
    plt.close(figure)


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
