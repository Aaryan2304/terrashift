"""RGB background rendering for validation figures.

Extracted from ``scripts/evaluation/analyse_validation_thresholds.py``. Reads imagery only when
called; importing this module touches no raster.
"""
from __future__ import annotations


import numpy as np

from terrashift.data.manifest import load_manifest, pair_config
from terrashift.data.safeloader import SafeProduct

__all__ = ["rgb_background"]

def rgb_background(args, pair, entry, target_shape):
    """Date-2 true colour over the validation stripe, percentile-stretched."""
    manifest = load_manifest(args.manifest)
    cfg = pair_config(manifest, pair)
    after = SafeProduct(cfg["path_date2"])
    col0 = entry["col_offset"]
    width = target_shape[1]
    window = ((0, target_shape[0]), (col0, col0 + width))
    bands = {}
    for band in ("B04", "B03", "B02"):
        values = after.read_band(band, window=window)[0]
        bands[band] = values
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        plane = rgb[..., channel]
        finite = np.isfinite(plane)
        low, high = np.percentile(plane[finite], [2, 98])
        out[..., channel] = np.clip((plane - low) / max(high - low, 1e-6), 0, 1)
    return out



import csv




import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import rasterio

from terrashift.data.manifest import load_manifest, pair_paths

from terrashift.data.safeloader import SafeProduct

from terrashift.geospatial.imagery import RGB_BANDS, stretch_pair

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


def read_register_row(path, record_id):
    """One row of the detection register, as a dict."""
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("record_id") == record_id:
                return row
    raise KeyError("record_id %r is not in %s" % (record_id, path))
