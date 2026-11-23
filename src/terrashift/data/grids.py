"""Raster grid contracts.

Two rasters can share a CRS and still be spatially misaligned: a different
origin, a different pixel size, or a different shape. Pixel indices from one
are then applied to the other without any warning. Every place that combines
rasters must prove the grids agree first.

``DEFAULT_ATOL`` is 1e-6 in the raster's linear unit (metres for UTM). It is
far below one pixel and far above the floating-point noise that survives GDAL
and PROJ round trips, so it rejects a real offset but accepts the same grid
read twice.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

DEFAULT_ATOL = 1e-6

#: A raster grid: CRS, affine transform, and shape.
Grid = namedtuple("Grid", "crs transform width height")


def grid_from_dataset(ds):
    """Build a :class:`Grid` from an open rasterio dataset."""
    crs = None if ds.crs is None else CRS.from_user_input(ds.crs)
    return Grid(crs, ds.transform, int(ds.width), int(ds.height))


def grid_from_path(path):
    """Build a :class:`Grid` from a raster path."""
    with rasterio.open(path) as ds:
        return grid_from_dataset(ds)


def grid_from_band(product, band="B08", grid_m=None):
    """The native grid of one band, optionally coarsened to ``grid_m`` metres.

    A target pixel size finer than the source is rejected: upsampling creates
    detail that the sensor did not record.
    """
    with rasterio.open(product.band_path(band)) as ds:
        if grid_m is None:
            return grid_from_dataset(ds)
        src_res = float(ds.res[0])
        factor = float(grid_m) / src_res
        if factor < 1.0:
            raise ValueError(
                "target grid (%g m) is finer than the source band (%g m); "
                "upsampling would create false detail" % (grid_m, src_res))
        # The factor must be integral, otherwise the coarsened grid does not
        # align with the source pixel edges and every mask lands half a pixel off.
        if abs(factor - round(factor)) > DEFAULT_ATOL:
            raise ValueError(
                "target grid (%g m) is not an integer multiple of the source "
                "band (%g m)" % (grid_m, src_res))
        factor = int(round(factor))
        width = max(1, int(round(ds.width / factor)))
        height = max(1, int(round(ds.height / factor)))
        transform = ds.transform * Affine.scale(factor, factor)
        return Grid(CRS.from_user_input(ds.crs) if ds.crs else None,
                    transform, width, height)


def _affine_values(transform):
    return (transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f)


def transforms_close(a, b, atol=DEFAULT_ATOL):
    """True when two affine transforms agree to ``atol`` in every term."""
    return all(abs(x - y) <= atol for x, y in zip(_affine_values(a), _affine_values(b)))


def resolution(grid):
    """(x, y) pixel size in the grid's linear unit."""
    return (abs(float(grid.transform.a)), abs(float(grid.transform.e)))


def bounds(grid):
    """(left, bottom, right, top) of the grid."""
    left = float(grid.transform.c)
    top = float(grid.transform.f)
    right = left + grid.width * float(grid.transform.a)
    bottom = top + grid.height * float(grid.transform.e)
    return (min(left, right), min(bottom, top), max(left, right), max(bottom, top))


def grid_mismatches(a, b, atol=DEFAULT_ATOL):
    """Every way in which grid ``a`` differs from grid ``b``.

    Returns a list of human-readable strings. An empty list means the grids
    are interchangeable for pixel indices.
    """
    problems = []
    if a.crs is None or b.crs is None:
        if a.crs is not b.crs:
            problems.append("one grid has no CRS and the other has %s"
                            % (a.crs or b.crs))
    elif CRS.from_user_input(a.crs) != CRS.from_user_input(b.crs):
        problems.append("CRS differs: %s vs %s" % (a.crs, b.crs))

    if (a.width, a.height) != (b.width, b.height):
        problems.append("shape differs: %dx%d vs %dx%d"
                        % (a.width, a.height, b.width, b.height))

    res_a, res_b = resolution(a), resolution(b)
    if abs(res_a[0] - res_b[0]) > atol or abs(res_a[1] - res_b[1]) > atol:
        problems.append("resolution differs: %gx%g vs %gx%g metres"
                        % (res_a[0], res_a[1], res_b[0], res_b[1]))

    if abs(float(a.transform.c) - float(b.transform.c)) > atol or \
            abs(float(a.transform.f) - float(b.transform.f)) > atol:
        problems.append("origin differs: (%.6f, %.6f) vs (%.6f, %.6f)"
                        % (float(a.transform.c), float(a.transform.f),
                           float(b.transform.c), float(b.transform.f)))

    if not transforms_close(a.transform, b.transform, atol=atol):
        problems.append("affine transform differs: %s vs %s"
                        % (tuple(round(v, 9) for v in _affine_values(a.transform)),
                           tuple(round(v, 9) for v in _affine_values(b.transform))))

    if any(abs(x - y) > atol for x, y in zip(bounds(a), bounds(b))):
        problems.append("bounds differ: %s vs %s"
                        % (tuple(round(v, 3) for v in bounds(a)),
                           tuple(round(v, 3) for v in bounds(b))))
    return problems


def require_same_grid(a, b, label_a="raster A", label_b="raster B",
                      atol=DEFAULT_ATOL):
    """Raise ``ValueError`` unless the two grids are interchangeable."""
    problems = grid_mismatches(a, b, atol=atol)
    if problems:
        raise ValueError(
            "%s and %s are not on the same grid: %s"
            % (label_a, label_b, "; ".join(problems)))


def require_expected_crs(grid, expected, label="raster"):
    """Raise unless ``grid.crs`` equals the manifest CRS for the pair."""
    if grid.crs is None:
        raise ValueError("%s has no CRS; expected %s" % (label, expected))
    want = CRS.from_user_input(expected)
    if CRS.from_user_input(grid.crs) != want:
        raise ValueError("%s has CRS %s; expected %s" % (label, grid.crs, want))


def covered_by(valid, grid, transform, width, height, atol=DEFAULT_ATOL):
    """True when ``grid``'s extent is fully inside an array of that transform."""
    observed = Grid(None, transform, width, height)
    left, bottom, right, top = bounds(grid)
    o_left, o_bottom, o_right, o_top = bounds(observed)
    return (left >= o_left - atol and bottom >= o_bottom - atol
            and right <= o_right + atol and top <= o_top + atol)


if __name__ == "__main__":
    base = Grid(CRS.from_epsg(32643), Affine(20.0, 0.0, 500000.0,
                                            0.0, -20.0, 3000000.0), 100, 100)
    shifted = base._replace(transform=Affine(20.0, 0.0, 500020.0,
                                             0.0, -20.0, 3000000.0))
    coarse = base._replace(transform=Affine(40.0, 0.0, 500000.0,
                                            0.0, -40.0, 3000000.0))
    assert grid_mismatches(base, base) == []
    assert any("origin differs" in p for p in grid_mismatches(base, shifted))
    assert any("resolution differs" in p for p in grid_mismatches(base, coarse))
    assert np.isclose(resolution(base)[0], 20.0)
    print("grid contract self-check OK")
