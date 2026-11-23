"""Review-chip rendering: component geometry, chip windows and true-colour chips.

Extracted from ``scripts/review/build_review_package.py``. The path constants
(``PREDICTIONS``, ``PARTITIONS``, ``LABELS``) intentionally stay in the script: they are
CLI configuration derived from the repository root, not reusable logic.
"""

from __future__ import annotations

import numpy as np
import rasterio
from affine import Affine


MARGIN_PX = 96          # 960 m of context around each candidate

STRUCTURE = np.ones((3, 3), dtype=np.uint8)

def reviewed_components(pair):
    """Return (components, component_of_pixel, rows, cols, transform, col0, target)."""
    labels_path = LABELS / ("pair_%s_v2" % pair.lower()) / ("manual_labels_%s.tif" % pair)
    partition_path = PARTITIONS / ("pair_%s_partition_10m.tif" % pair)
    target, col0, _ = read_test_target(labels_path, partition_path)
    with rasterio.open(labels_path) as handle:
        base = handle.transform
        height = handle.height
    covered = target != 255
    rows, cols = np.nonzero(covered)
    change = covered & (target == 1)
    components, count = ndi.label(change, structure=STRUCTURE)
    return {
        "components": components, "count": count, "rows": rows, "cols": cols,
        "transform": base * Affine.translation(col0, 0), "col0": col0,
        "target": target, "shape": (height, target.shape[1]),
        "labels_path": labels_path,
    }

def component_scores(state, component_id, scores):
    component_of_pixel = state["components"][state["rows"], state["cols"]]
    selected = scores[component_of_pixel == component_id]
    if selected.size == 0:
        return None
    return {
        "mean": round(float(selected.mean()), 6),
        "max": round(float(selected.max()), 6),
        "median": round(float(np.median(selected)), 6),
        "pixels": int(selected.size),
    }

def chip_bounds(state, component_id):
    """Window in tile pixel coordinates around one component, clamped to the stripe."""
    mask = state["components"] == component_id
    rows, cols = np.nonzero(mask)
    row0 = max(0, int(rows.min()) - MARGIN_PX)
    row1 = min(state["shape"][0], int(rows.max()) + 1 + MARGIN_PX)
    col0 = max(0, int(cols.min()) - MARGIN_PX)
    col1 = min(state["shape"][1], int(cols.max()) + 1 + MARGIN_PX)
    tile_col0 = col0 + state["col0"]
    # Keep a square-ish readable chip without leaving the stripe.
    size = max(row1 - row0, col1 - col0)
    size = int(min(size, 512))
    if row1 - row0 > size:
        centre = (row0 + row1) // 2
        row0, row1 = centre - size // 2, centre - size // 2 + size
    if col1 - col0 > size:
        centre = (col0 + col1) // 2
        col0, col1 = centre - size // 2, centre - size // 2 + size
    return row0, row1, col0, col1, tile_col0

def stretch(before, after):
    """One display stretch per candidate, shared by both dates."""
    stack = np.concatenate([before[np.isfinite(before)], after[np.isfinite(after)]])
    if stack.size == 0:
        return (0.0, 1.0)
    low, high = np.percentile(stack, [2, 98])
    if high <= low:
        high = low + 1e-6
    return float(low), float(high)

def render_chips(loader, pair, state, component_id, candidate_id):
    row0, row1, col0, col1, _ = chip_bounds(state, component_id)
    size = max(row1 - row0, col1 - col0)
    before, after = loader(pair, row0, col0 + state["col0"], size)
    low, high = stretch(before, after)
    return {
        "before": before, "after": after, "bounds": (row0, row1, col0, col1),
        "stretch": (low, high), "size": size,
    }

def save_chip(path, image, low, high):
    """One date, true colour, the candidate's shared display stretch, no overlay."""
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

def rings_in_chip_pixels(state, component_id, bounds):
    """Component outline in chip pixel coordinates, using the chip's own window."""
    r0, _r1, c0, _c1 = bounds
    mask = (state["components"] == component_id)[
        bounds[0]:bounds[1], bounds[2]:bounds[3]]
    out = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask,
                              transform=Affine.identity()):
        if int(value) != 1:
            continue
        # ``transform=Affine.identity()`` on the cropped mask already yields chip-local
        # coordinates, so no further offset is applied.
        part = shape(geom)
        if part.geom_type == "Polygon":
            out.append(list(part.exterior.coords))
        else:
            for polygon in part.geoms:
                out.append(list(polygon.exterior.coords))
    return out
