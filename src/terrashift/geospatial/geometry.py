"""Geometry for detection records: components, areas, uncertainty and length.

Area has an exact pixel count and an uncertain boundary: a one-pixel edge shifts it by tens of square
metres at 10 m, so each area carries a boundary-sensitivity range. That range is not a confidence
interval, since nothing here is calibrated against repeated ground truth.

Length is measured on the skeleton, because counting changed pixels overestimates a diagonal by the
diagonal-to-axis ratio and any band by roughly its width.

Connectivity is not interchangeable: ``components()`` uses 8-connectivity for detection geometry,
while the OSCD metrics in ``eval.py`` use 4-connectivity, fixed when the benchmark was recorded.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)
_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))
_EIGHT = ((0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1))

# Components

def components(mask):
    """Label a boolean mask and return ``(labels, areas)`` with area per label.

    Uses 8-connectivity, so a diagonal pair is one component. Anything that
    polygonises this labelling must also use 8-connectivity, otherwise the
    geometry splits into pieces the count says are one object.
    """
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndi.label(mask, structure=_STRUCTURE_8)
    if count == 0:
        return labels, np.zeros(0, dtype=np.int64)
    return labels, np.bincount(labels.ravel(), minlength=count + 1)[1:]

# Area

def boundary_area_range(mask, pixel_area_m2, iterations=1):
    """Nominal, eroded and dilated area of one component in square metres.

    Erosion and dilation both use a 3x3 structuring element, so the range
    spans a one-pixel boundary movement in every direction. It is a sensitivity
    range, not a confidence interval.

    Limitation: dilation treats the array edge as empty, so a component that
    touches the border is under-dilated. Crop with a margin before calling this
    if the component touches the tile edge.
    """
    mask = np.asarray(mask, dtype=bool)
    nominal = int(mask.sum())
    eroded = ndi.binary_erosion(mask, structure=_STRUCTURE_8,
                                iterations=iterations, border_value=0)
    dilated = ndi.binary_dilation(mask, structure=_STRUCTURE_8,
                                  iterations=iterations, border_value=0)
    return {
        "nominal_px": nominal,
        "eroded_px": int(eroded.sum()),
        "dilated_px": int(dilated.sum()),
        "nominal_m2": nominal * pixel_area_m2,
        "min_m2": int(eroded.sum()) * pixel_area_m2,
        "max_m2": int(dilated.sum()) * pixel_area_m2,
    }

def polygon_area_sensitivity(geometry, displacement_m, join_style="round"):
    """Boundary-sensitivity area band for a stored polygon.

    The interval is ``[area(buffer(-d)), area(buffer(+d))]`` where ``d`` is
    ``displacement_m``. Positive displacement dilates, so the upper bound is the
    dilated area. The boundary is unknown to within ``d`` by assumption; this
    function does not claim to know the distribution of that error.

    Parameters
    ----------
    geometry : shapely geometry or None
        A Polygon or MultiPolygon. Holes and multiple parts are handled by the
        geometry engine, not by this function.
    displacement_m : float
        Boundary displacement in metres. Must be non-negative. Pass the pixel
        size for a one-pixel band, or add the measured registration residual
        when comparing against a fixed reference.
    join_style : str
        Buffer join style. Round is the default because an uncertain boundary
        has no sharp corners.

    Returns
    -------
    dict
        ``status`` is ``ok``, ``empty``, ``invalid`` or ``not_areal``. When the
        status is not ``ok`` the metre fields are ``None`` and ``reason`` says
        what was wrong. ``disappeared`` marks a polygon that erodes to nothing,
        which is the correct answer for a structure narrower than ``2 * d``.
    """
    from shapely import make_valid
    from shapely.geometry import MultiPolygon

    def blank(status, reason, parts=0, holes=0):
        return {"status": status, "reason": reason, "displacement_m": displacement_m,
                "nominal_m2": None, "min_m2": None, "max_m2": None,
                "minus_m2": None, "plus_m2": None, "disappeared": False,
                "parts": parts, "holes": holes}

    if displacement_m < 0:
        raise ValueError("displacement_m must be non-negative")
    if geometry is None:
        return blank("empty", "no geometry")
    if geometry.is_empty:
        return blank("empty", "empty geometry")
    if geometry.geom_type not in ("Polygon", "MultiPolygon"):
        return blank("not_areal", "geometry is %s, not areal" % geometry.geom_type)

    if not geometry.is_valid:
        repaired = make_valid(geometry)
        if repaired.is_empty or repaired.geom_type not in ("Polygon", "MultiPolygon"):
            return blank("invalid", "geometry is invalid and could not be repaired")
        geometry = repaired

    parts = 1 if geometry.geom_type == "Polygon" else len(geometry.geoms)
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    holes = sum(len(polygon.interiors) for polygon in polygons)

    nominal = float(geometry.area)
    if displacement_m == 0.0:
        return {"status": "ok", "reason": None, "displacement_m": 0.0,
                "nominal_m2": nominal, "min_m2": nominal, "max_m2": nominal,
                "minus_m2": 0.0, "plus_m2": 0.0, "disappeared": False,
                "parts": parts, "holes": holes}

    eroded = geometry.buffer(-displacement_m, join_style=join_style)
    dilated = geometry.buffer(displacement_m, join_style=join_style)
    minimum = float(eroded.area) if not eroded.is_empty else 0.0
    maximum = float(dilated.area) if not dilated.is_empty else 0.0
    return {"status": "ok", "reason": None, "displacement_m": float(displacement_m),
            "nominal_m2": nominal, "min_m2": minimum, "max_m2": maximum,
            "minus_m2": nominal - minimum, "plus_m2": maximum - nominal,
            "disappeared": bool(minimum == 0.0 and nominal > 0.0),
            "parts": parts, "holes": holes}

def pixel_area(transform):
    """Area of one pixel in square metres from an affine transform.

    Uses the determinant, so a rotated transform gives the true cell area. The
    earlier form ``|a * e|`` assumed the off-diagonal terms were zero and
    silently understated any rotated grid.
    """
    a, b = float(transform.a), float(transform.b)
    d, e = float(transform.d), float(transform.e)
    return abs(a * e - b * d)

# Length

def _adjacent_pairs(skeleton, dr, dc):
    """Count skeleton pixel pairs separated by the offset ``(dr, dc)``."""
    height, width = skeleton.shape
    row_start, row_stop = max(0, -dr), height - max(0, dr)
    col_start, col_stop = max(0, -dc), width - max(0, dc)
    first = skeleton[row_start:row_stop, col_start:col_stop]
    second = skeleton[row_start + dr:row_stop + dr, col_start + dc:col_stop + dc]
    return int(np.count_nonzero(first & second))

def _degree(skeleton):
    """Number of neighbours, counting a diagonal only when it is not bridged.

    A plain 3x3 sum makes a pixel that merely touches a line diagonally look like
    a junction of degree 3. That mislabels the first pixel of every spur and
    blocks spur pruning, so the same bridge rule as the length measurement
    applies here: a diagonal neighbour counts only when neither orthogonal
    bridging pixel is on the skeleton.
    """
    skeleton = np.asarray(skeleton, dtype=bool)
    padded = np.pad(skeleton, 1, constant_values=False)
    centre = padded[1:-1, 1:-1]
    up, down = padded[:-2, 1:-1], padded[2:, 1:-1]
    left, right = padded[1:-1, :-2], padded[1:-1, 2:]
    up_left, up_right = padded[:-2, :-2], padded[:-2, 2:]
    down_left, down_right = padded[2:, :-2], padded[2:, 2:]
    degree = (up.astype(np.int32) + down + left + right)
    degree += (up_left & ~(up | left)).astype(np.int32)
    degree += (up_right & ~(up | right)).astype(np.int32)
    degree += (down_left & ~(down | left)).astype(np.int32)
    degree += (down_right & ~(down | right)).astype(np.int32)
    return (degree * centre.astype(np.int32)).astype(np.uint8)

def _diagonal_pairs(skeleton):
    """Count diagonal adjacencies that are not shortcuts around a junction.

    Two diagonal neighbours are only counted when they are not already joined
    through an orthogonal bridging pixel. At a four-way junction the tips of two
    arms are diagonally adjacent across the centre, and counting those pairs adds
    four phantom diagonals to a plus sign: 56.6 m on a 200 m cross, an 11 %
    overestimate. A clean one-pixel diagonal has no bridging pixel, so it is
    still counted in full.
    """
    if skeleton.size < 4:
        return 0
    first = skeleton[:-1, :-1] & skeleton[1:, 1:]
    bridge = skeleton[:-1, 1:] | skeleton[1:, :-1]
    down_right = int(np.count_nonzero(first & ~bridge))
    first = skeleton[:-1, 1:] & skeleton[1:, :-1]
    bridge = skeleton[:-1, :-1] | skeleton[1:, 1:]
    down_left = int(np.count_nonzero(first & ~bridge))
    return down_right + down_left

def _walk_neighbours(pixel, skeleton, height, width, previous=None):
    """Neighbours of a skeleton pixel, preferring orthogonal over diagonal.

    Walking a skeleton has to use the same rule as the degree count, or the walk
    sees three neighbours at a pixel that merely sits beside a line: a spur one
    pixel from the trunk is diagonally adjacent to two trunk pixels as well as
    orthogonally adjacent to the next spur pixel. Orthogonal neighbours, when
    present, are the true continuation.
    """
    row, col = pixel
    orthogonal, diagonal = [], []
    for dr, dc in _EIGHT:
        r, c = row + dr, col + dc
        if not (0 <= r < height and 0 <= c < width) or not skeleton[r, c]:
            continue
        if (r, c) == previous:
            continue
        (orthogonal if dr == 0 or dc == 0 else diagonal).append((r, c))
    return orthogonal if orthogonal else diagonal

def _prune_spurs(skeleton, max_length_px):
    """Remove skeleton branches shorter than ``max_length_px``.

    A spur is a chain that starts at an endpoint and reaches a junction within
    the limit. Short chains that connect two endpoints are kept: an isolated
    short line is an object, not an artefact of thinning.
    """
    skeleton = np.array(skeleton, dtype=bool, copy=True)
    if max_length_px <= 0 or not skeleton.any():
        return skeleton, 0

    height, width = skeleton.shape
    removed = 0
    degree = _degree(skeleton)
    for _ in range(max_length_px + 2):
        deleted = False
        for row, col in np.argwhere(skeleton & (degree == 1)):
            if not skeleton[row, col]:
                continue
            path = [(int(row), int(col))]
            previous = None
            current = (int(row), int(col))
            reached_junction = False
            while len(path) <= max_length_px:
                neighbours = _walk_neighbours(current, skeleton, height, width,
                                              previous)
                if len(neighbours) != 1:
                    break
                candidate = neighbours[0]
                if degree[candidate] >= 3:
                    reached_junction = True
                    break
                previous, current = current, candidate
                path.append(current)
            if reached_junction and len(path) <= max_length_px:
                for pixel in path:
                    skeleton[pixel] = False
                removed += len(path)
                degree = _degree(skeleton)
                deleted = True
        if not deleted:
            break
    return skeleton, removed

def skeleton_graph(mask, pixel_size_m, spur_length_px=0):
    """Measure a structure's centreline as a graph, in metres.

    Every pair of adjacent skeleton pixels is one edge. A horizontal or vertical
    edge is ``pixel_size_m`` long; a diagonal edge is ``pixel_size_m *
    sqrt(2)``. Each adjacency is counted exactly once, so a junction with four
    branches contributes four edges and not six.

    Counting changed pixels instead is wrong in both directions: a band of width
    ``w`` pixels is overestimated by roughly the factor ``w``, because the pixel
    count measures area and not length, while a one-pixel diagonal is
    underestimated because the count ignores the diagonal.

    Parameters
    ----------
    spur_length_px : int
        Prune branches shorter than this many pixels before measuring. Zero
        keeps every branch. Thinning a noisy boundary produces short spurs that
        are not part of the structure; pruning them is a modelling choice and is
        off by default so the raw measurement stays visible.

    Returns
    -------
    dict
        Length in metres, plus the edge counts, the graph degree statistics and
        how many spur pixels were removed. ``isolated_px`` counts single
        skeleton pixels with no neighbour: they carry no length.
    """
    from skimage.morphology import skeletonize

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"skeleton_px": 0, "orthogonal_edges": 0, "diagonal_edges": 0,
                "length_m": 0.0, "endpoints": 0, "junctions": 0, "isolated_px": 0,
                "components": 0, "spur_length_px": int(spur_length_px),
                "spur_px_removed": 0}

    skeleton, removed = _prune_spurs(skeletonize(mask), spur_length_px)
    if not skeleton.any():
        return {"skeleton_px": 0, "orthogonal_edges": 0, "diagonal_edges": 0,
                "length_m": 0.0, "endpoints": 0, "junctions": 0, "isolated_px": 0,
                "components": 0, "spur_length_px": int(spur_length_px),
                "spur_px_removed": removed}

    orthogonal = _adjacent_pairs(skeleton, 0, 1) + _adjacent_pairs(skeleton, 1, 0)
    diagonal = _diagonal_pairs(skeleton)
    length = (orthogonal + diagonal * np.sqrt(2.0)) * pixel_size_m
    degree = _degree(skeleton)
    # A junction is a cluster of high-degree pixels, not a single pixel: the
    # arms of a plus sign give degree 3 to four pixels around the centre.
    junction_clusters = int(ndi.label((degree >= 3) & skeleton,
                                      structure=_STRUCTURE_8)[1])
    return {"skeleton_px": int(skeleton.sum()),
            "orthogonal_edges": orthogonal,
            "diagonal_edges": diagonal,
            "length_m": float(length),
            "endpoints": int(np.count_nonzero((degree == 1) & skeleton)),
            "junctions": junction_clusters,
            "isolated_px": int(np.count_nonzero((degree == 0) & skeleton)),
            "components": int(ndi.label(skeleton, structure=_STRUCTURE_8)[1]),
            "spur_length_px": int(spur_length_px),
            "spur_px_removed": int(removed)}

def skeleton_length(mask, pixel_size_m):
    """Length of the skeleton of a structure, in metres.

    Kept as the plain entry point. It is ``skeleton_graph`` with no pruning, and
    returns the same four keys it always did.
    """
    result = skeleton_graph(mask, pixel_size_m)
    return {"skeleton_px": result["skeleton_px"],
            "orthogonal_edges": result["orthogonal_edges"],
            "diagonal_edges": result["diagonal_edges"],
            "length_m": result["length_m"]}

def length_sensitivity(mask, pixel_size_m, iterations=1, spur_length_px=0):
    """Sensitivity envelope for skeleton length: nominal, lower and upper.

    Erosion and dilation do not move the measurement monotonically. Thinning a
    dilated band can shorten the centreline, because the skeleton is trimmed at
    its ends by roughly half the local width, and erosion can break a band into
    pieces. So the dilated measurement is not automatically the upper bound and
    the eroded one is not automatically the lower bound.

    The envelope is therefore ``min`` and ``max`` over the three measurements,
    which guarantees that the nominal value is inside its own band. The raw
    eroded and dilated values are kept as ``eroded_m`` and ``dilated_m`` so the
    asymmetry stays visible instead of being silently corrected.
    """
    mask = np.asarray(mask, dtype=bool)
    nominal = skeleton_graph(mask, pixel_size_m, spur_length_px)["length_m"]
    eroded = ndi.binary_erosion(mask, structure=_STRUCTURE_8,
                                iterations=iterations, border_value=0)
    dilated = ndi.binary_dilation(mask, structure=_STRUCTURE_8,
                                  iterations=iterations, border_value=0)
    eroded_length = skeleton_graph(eroded, pixel_size_m, spur_length_px)["length_m"]
    dilated_length = skeleton_graph(dilated, pixel_size_m, spur_length_px)["length_m"]
    return {"nominal_m": nominal,
            "min_m": min(nominal, eroded_length, dilated_length),
            "max_m": max(nominal, eroded_length, dilated_length),
            "eroded_m": eroded_length,
            "dilated_m": dilated_length,
            "disappeared_under_erosion": bool(mask.any() and not eroded.any()),
            "dilated_shorter_than_nominal": bool(dilated_length < nominal)}

# Registration

def registration_displacement(payload, pair, grid_resolution_m=None):
    """Measured registration residual for one pair, in metres and grid pixels.

    Reads the residual block written by the masked registration and reports the
    median and 95th-percentile displacement of the converged windows. A global
    translation does not change an area or a length, so this number is only a
    boundary term when a detection is compared against a fixed reference.

    Raises ``KeyError`` when the pair or its residual block is absent, rather
    than inventing a value.
    """
    entry = payload[pair]
    # The masked registration writes the residual block inside "initial"; older
    # exports put it at the top of the pair entry. Accept both rather than
    # silently returning zero.
    residual = entry.get("residual")
    if not residual and isinstance(entry.get("initial"), dict):
        residual = entry["initial"].get("residual")
    if not residual:
        raise KeyError("registration entry for pair %s has no residual block" % pair)
    source_pixel_m = float(entry.get("pixel_size_m", 10.0))
    median_m = residual.get("median_displacement_m")
    p95_m = residual.get("p95_displacement_m")
    if median_m is None:
        median_dx = float(residual.get("median_dx_px", 0.0)) * source_pixel_m
        median_dy = float(residual.get("median_dy_px", 0.0)) * source_pixel_m
        median_m = float(np.hypot(median_dx, median_dy))
    if p95_m is None:
        p95_px = residual.get("p95_displacement_px")
        p95_m = float(p95_px) * source_pixel_m if p95_px is not None else None
    result = {"pair": pair,
              "source_pixel_size_m": source_pixel_m,
              "n_converged": residual.get("n_converged"),
              "n_failed": residual.get("n_failed"),
              "median_m": float(median_m),
              "p95_m": None if p95_m is None else float(p95_m),
              "rmse_mean": residual.get("rmse_mean"),
              # A tile-wide median describes the windows that converged. Where a
              # window failed there is no estimate at all, so the median must not
              # be applied locally in that area.
              "covers_whole_tile": residual.get("n_failed") == 0,
              "unconverged_windows": [entry.get("window")
                                      for entry in residual.get("windows", [])
                                      if not entry.get("converged", True)]}
    if grid_resolution_m:
        result["median_grid_px"] = float(median_m) / float(grid_resolution_m)
        result["p95_grid_px"] = (None if p95_m is None
                                 else float(p95_m) / float(grid_resolution_m))
    return result

# Mask cleaning

def clean_binary_mask(mask, valid=None, open_size=3, close_size=5, min_area_px=4):
    """Open, close, re-clip to validity, then drop small components.

    This was written four times: in the classical baseline, the review ranking,
    the threshold-sensitivity sweep and the baseline figure. One implementation
    means one place to change the cleaning rule.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        Thresholded candidate mask.
    valid : numpy.ndarray of bool, optional
        Pixels that may be candidates. Re-applied after morphology, because
        closing can grow a component back into water, cloud or a registration
        gap.
    open_size, close_size : int
        Structuring-element sizes. A 3x3 opening deletes any structure narrower
        than three pixels, which is 60 m on a 20 m grid.
    min_area_px : int
        Components smaller than this are dropped. Pass 0 to keep everything,
        which the sensitivity analysis needs so it can report the
        post-morphology count separately.
    """
    mask = np.asarray(mask, dtype=bool)
    cleaned = ndi.binary_opening(mask, np.ones((open_size, open_size), dtype=bool))
    cleaned = ndi.binary_closing(cleaned, np.ones((close_size, close_size), dtype=bool))
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != cleaned.shape:
            raise ValueError("valid mask shape does not match the candidate mask")
        cleaned &= valid
    if min_area_px > 0:
        labels, count = ndi.label(cleaned)
        if count:
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            keep = sizes >= min_area_px
            keep[0] = False
            cleaned = keep[labels]
    return cleaned
