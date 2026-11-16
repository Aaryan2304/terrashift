"""Geometry for detection records: components, areas, uncertainty and length.

Area on a raster has an exact pixel count but an uncertain boundary. A change
edge one pixel wide can move the measured area by tens of square metres at
10 m resolution, so every reported area is given with an erosion/dilation
range rather than as a single number.

Length needs the same care. Counting changed pixels and calling it "road
length" overestimates a diagonal road by the ratio of the diagonal to the
axis. This module skeletonises the mask first and then counts skeleton edges
with the correct diagonal weight.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def components(mask):
    """Label a boolean mask and return ``(labels, areas)`` with area per label."""
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return labels, np.zeros(0, dtype=np.int64)
    return labels, np.bincount(labels.ravel(), minlength=count + 1)[1:]


def boundary_area_range(mask, pixel_area_m2, iterations=1):
    """Nominal, eroded and dilated area of one component in square metres.

    Erosion and dilation both use a 3x3 structuring element, so the range
    spans a one-pixel boundary movement in every direction. It is a sensitivity
    range, not a confidence interval.
    """
    mask = np.asarray(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    nominal = int(mask.sum())
    eroded = ndi.binary_erosion(mask, structure=structure, iterations=iterations,
                                border_value=0)
    dilated = ndi.binary_dilation(mask, structure=structure, iterations=iterations,
                                  border_value=0)
    return {
        "nominal_px": nominal,
        "eroded_px": int(eroded.sum()),
        "dilated_px": int(dilated.sum()),
        "nominal_m2": nominal * pixel_area_m2,
        "min_m2": int(eroded.sum()) * pixel_area_m2,
        "max_m2": int(dilated.sum()) * pixel_area_m2,
    }


def _adjacent_pairs(skeleton, dr, dc):
    """Count skeleton pixel pairs separated by the offset ``(dr, dc)``."""
    height, width = skeleton.shape
    row_start, row_stop = max(0, -dr), height - max(0, dr)
    col_start, col_stop = max(0, -dc), width - max(0, dc)
    first = skeleton[row_start:row_stop, col_start:col_stop]
    second = skeleton[row_start + dr:row_stop + dr, col_start + dc:col_stop + dc]
    return int(np.count_nonzero(first & second))


def skeleton_length(mask, pixel_size_m):
    """Length of the skeleton of a structure, in metres.

    Each pair of adjacent skeleton pixels is one edge. A horizontal or vertical
    edge is ``pixel_size_m`` long; a diagonal edge is ``pixel_size_m *
    sqrt(2)``.

    Counting changed pixels instead is wrong in both directions: a band of
    width ``w`` pixels is overestimated by roughly the factor ``w``, because
    the pixel count measures area and not length, while a one-pixel diagonal
    is underestimated because the count ignores the diagonal.
    """
    from skimage.morphology import skeletonize

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"skeleton_px": 0, "orthogonal_edges": 0, "diagonal_edges": 0,
                "length_m": 0.0}
    skeleton = skeletonize(mask)
    orthogonal = (_adjacent_pairs(skeleton, 0, 1) + _adjacent_pairs(skeleton, 1, 0))
    diagonal = (_adjacent_pairs(skeleton, 1, 1) + _adjacent_pairs(skeleton, 1, -1))
    length = (orthogonal + diagonal * np.sqrt(2.0)) * pixel_size_m
    return {"skeleton_px": int(skeleton.sum()),
            "orthogonal_edges": orthogonal, "diagonal_edges": diagonal,
            "length_m": float(length)}


def pixel_area(transform):
    """Area of one pixel in square metres from an affine transform."""
    return abs(float(transform.a) * float(transform.e))


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
