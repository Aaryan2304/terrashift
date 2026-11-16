"""Paired observation validity for a bi-temporal Sentinel-2 pair.

A difference between two dates is only meaningful where **both** dates are
valid observations. A pixel that is finite in both rasters can still be a
cloud, cirrus, saturated pixel, defective detector cell, or a hole created by
the registration warp. Finite reflectance is not the same as a valid
observation.

This module builds that evidence in one place:

``observation_valid``
    One date's validity on the analysis grid: source NoData for every band in
    the spectral contract, plus the L1C ``MSK_CLASSI`` and ``MSK_QUALIT``
    flags. Quality information is never invented: a missing or unexpected
    mask fails loudly (see :class:`data.masks.QualityMasks`).

``registration_coverage``
    The footprint of a shifted image. ``cv2.warpAffine`` fills the uncovered
    border with reflected pixels, which are finite and look like data. The
    coverage mask marks exactly those pixels invalid so the reflected values
    can serve as convolution context without ever being counted as observed.

``paired_validity``
    ``date1_valid & date2_valid & coverage``. This is the only mask that may
    be used for PIF selection, normalisation fitting, change-score
    thresholds, candidate components, and metric denominators.

Water state is deliberately **not** part of validity. A pixel that was water
and is now land is a reclamation candidate, not an invalid observation.

References
----------
- Copernicus SentiWiki, Sentinel-2 products (L1C quality indicators):
  https://sentiwiki.copernicus.eu/web/s2-products
- Sentinel-2 Product Specifications Document (MSK_CLASSI / MSK_QUALIT
  bit definitions).
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling

from data.grids import Grid, transforms_close
from data.masks import QualityMasks

#: The four native 10 m bands of the model and baseline contract.
DEFAULT_BANDS = ("B02", "B03", "B04", "B08")


def _band_reading_grid(ds, out_shape):
    """Transform of ``ds`` read at ``out_shape``, as rasterio computes it."""
    return ds.transform * ds.transform.scale(
        ds.width / out_shape[1], ds.height / out_shape[0])


def observation_valid(product, grid, bands=DEFAULT_BANDS, qualit_bands=None):
    """Build one date's observation-valid mask on ``grid``.

    Parameters
    ----------
    product : data.safeloader.SafeProduct
        One L1C product.
    grid : data.grids.Grid
        The analysis grid. Every band and mask is read onto this grid with
        nearest-neighbour sampling and its resulting transform is checked
        against ``grid``; a mismatch raises instead of silently misaligning.
    bands : sequence of str
        The spectral contract. A pixel with DN == 0 in **any** of these bands
        is invalid.
    qualit_bands : sequence of str, optional
        Bands whose ``MSK_QUALIT`` flags apply. Defaults to ``bands``, which
        is the conservative choice: a pixel flagged in any contract band is
        excluded from the four-band comparison.

    Returns
    -------
    valid : numpy.ndarray, bool, shape ``(grid.height, grid.width)``
    detail : dict
        Pixel counts per contributing flag, plus ``total_invalid``. The
        counts are diagnostics, not a statistical result.
    """
    if not bands:
        raise ValueError("at least one band is required to define validity")
    invalid = np.zeros((grid.height, grid.width), dtype=bool)
    detail = {}

    for band in bands:
        reflectance, transform, crs = product.read_band(
            band, out_shape=(grid.height, grid.width),
            resampling=Resampling.nearest)
        if crs is not None and grid.crs is not None and \
                rasterio.crs.CRS.from_user_input(crs) != rasterio.crs.CRS.from_user_input(grid.crs):
            raise ValueError("band %s CRS %s does not match grid CRS %s"
                             % (band, crs, grid.crs))
        if not transforms_close(transform, grid.transform):
            raise ValueError(
                "band %s read at %dx%d does not land on the analysis grid "
                "(transform %s vs %s)" % (band, grid.height, grid.width,
                                          transform, grid.transform))
        # read_band maps DN == 0 to NaN, so a non-finite pixel is source NoData.
        nodata = ~np.isfinite(reflectance)
        detail["nodata_" + band] = int(nodata.sum())
        invalid |= nodata

    quality = QualityMasks(product)
    qualit_invalid, qualit_detail = quality.invalid_on_grid(
        grid, qualit_bands=(bands if qualit_bands is None else qualit_bands))
    detail.update(qualit_detail)
    invalid |= qualit_invalid

    detail["total_invalid"] = int(invalid.sum())
    detail["total_pixels"] = int(invalid.size)
    detail["valid_pixels"] = int((~invalid).sum())
    return ~invalid, detail


def registration_coverage(shape, shift_x, shift_y):
    """Footprint of an image shifted by ``(shift_x, shift_y)`` pixels.

    Thin alias of :func:`data.register.coverage_mask`, kept here so the
    validity chain has one import. Pass the same displacement that is given to
    :func:`data.register.apply_shift`. The border that the warp fills by
    reflection or replication is therefore marked invalid.

    Returns
    -------
    numpy.ndarray, bool
        True where the shifted image has real source content.
    """
    from data.register import coverage_mask

    return coverage_mask(shape, shift_x, shift_y)


def paired_validity(date1_valid, date2_valid, coverage=None):
    """Combine two per-date masks and optional registration coverage.

    All inputs must share one shape. ``coverage`` defaults to "everything is
    covered", which is correct only when no warp was applied.
    """
    if date1_valid.shape != date2_valid.shape:
        raise ValueError("date masks must have the same shape")
    paired = np.asarray(date1_valid, dtype=bool) & np.asarray(date2_valid, dtype=bool)
    if coverage is not None:
        if coverage.shape != paired.shape:
            raise ValueError("coverage mask must have the same shape as the date masks")
        paired &= np.asarray(coverage, dtype=bool)
    return paired


def build_paired_validity(product1, product2, grid, shift=None,
                          bands=DEFAULT_BANDS, qualit_bands=None):
    """Per-date masks, coverage, and the paired mask for one pair.

    ``shift`` is ``(dx, dy)`` in **grid pixels**, already expressed as the
    displacement that :func:`data.register.apply_shift` receives (that is,
    the negated ECC estimate). ``None`` means no warp was applied.

    Returns
    -------
    dict with ``valid_date1``, ``valid_date2``, ``coverage``, ``paired``,
    ``detail_date1``, ``detail_date2``, ``shift_px``.
    """
    valid1, detail1 = observation_valid(product1, grid, bands=bands,
                                        qualit_bands=qualit_bands)
    valid2, detail2 = observation_valid(product2, grid, bands=bands,
                                        qualit_bands=qualit_bands)
    if shift is None:
        coverage = np.ones((grid.height, grid.width), dtype=bool)
    else:
        coverage = registration_coverage((grid.height, grid.width), *shift)
    return {
        "valid_date1": valid1,
        "valid_date2": valid2,
        "coverage": coverage,
        "paired": paired_validity(valid1, valid2, coverage),
        "detail_date1": detail1,
        "detail_date2": detail2,
        "shift_px": None if shift is None else [float(shift[0]), float(shift[1])],
    }


def window_validity(product, window, band="B08", qualit_bands=None,
                    flag_names=None):
    """Observation validity for one image window.

    The mask is aligned to the image window pixel for pixel: the window is
    read once with :meth:`data.safeloader.SafeProduct.read_band`, which gives
    the window transform and CRS, and the quality masks are placed on that
    same transform with nearest sampling. No resampling of the image is
    involved, so the mask can be passed straight to ECC as ``inputMask``.

    Parameters
    ----------
    product : data.safeloader.SafeProduct
    window : tuple
        ``((row_start, row_end), (col_start, col_end))`` in the band's native
        pixels, as accepted by ``read_band``.
    band : str
        Band that defines the window grid.
    qualit_bands : sequence of str, optional
        Bands whose MSK_QUALIT flags apply. Defaults to ``(band,)``.
    flag_names : sequence of str, optional
        MSK_QUALIT flags that invalidate a pixel. Defaults to the
        conservative set in :mod:`data.masks`.

    Returns
    -------
    valid : numpy.ndarray, bool, shape of the window
    detail : dict
        Pixel counts per flag, plus ``total_invalid``.
    """
    reflectance, transform, crs = product.read_band(band, window=window)
    if crs is None:
        raise ValueError("the product band has no CRS; cannot place quality masks")

    invalid = ~np.isfinite(reflectance)
    detail = {"nodata_" + band: int(invalid.sum())}

    quality = QualityMasks(product)
    kwargs = {"qualit_bands": (band,) if qualit_bands is None else tuple(qualit_bands)}
    if flag_names is not None:
        kwargs["qualit_flags"] = tuple(flag_names)
    quality_invalid, quality_detail = quality.invalid_window(
        transform, reflectance.shape, crs, **kwargs)

    detail.update(quality_detail)
    invalid |= quality_invalid
    detail["total_invalid"] = int(invalid.sum())
    detail["total_pixels"] = int(invalid.size)
    detail["valid_pixels"] = int((~invalid).sum())
    return ~invalid, detail


def write_validity_raster(path, valid, grid, tags=None):
    """Write a validity raster: 1 = valid observation, 0 = NoData/invalid.

    The nodata value is 0, so ``dataset.dataset_mask()`` returns exactly the
    valid footprint. That keeps consumers that already call ``dataset_mask``
    on the change score correct without a second convention.
    """
    if valid.shape != (grid.height, grid.width):
        raise ValueError("validity mask shape does not match the grid")
    profile = dict(driver="GTiff", height=grid.height, width=grid.width,
                   count=1, dtype="uint8", crs=grid.crs,
                   transform=grid.transform, nodata=0,
                   compress="deflate", tiled=True)
    base_tags = {
        "output_type": "observation_validity",
        "valid_value": "1",
        "invalid_or_nodata_value": "0",
        "includes_source_nodata": "true",
        "includes_msk_classi": "true",
        "includes_msk_qualit": "true",
        "includes_registration_coverage": "true",
    }
    if tags:
        base_tags.update({str(k): str(v) for k, v in tags.items()})
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(valid.astype(np.uint8), 1)
        ds.update_tags(**base_tags)


if __name__ == "__main__":
    shape = (8, 8)
    full = np.ones(shape, dtype=bool)
    # A positive displacement moves the content right, so the left columns are
    # the ones with no source pixel.
    shifted = registration_coverage(shape, 2.0, 0.0)
    assert not shifted[0, 0], "a pixel that only sees reflected border is invalid"
    assert not shifted[0, 1] and shifted[0, 2] and shifted[0, 7]
    assert paired_validity(full, full, shifted).sum() < full.sum()
    one_date = full.copy()
    one_date[3, 3] = False
    assert not paired_validity(one_date, full)[3, 3]
    print("validity self-check OK; covered pixels after a 2 px shift:",
          int(shifted.sum()), "of", int(full.sum()))
