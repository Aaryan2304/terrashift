"""Water and quality masks.

Three separate masks are produced:

    stable water core      water in both dates, exclude from infrastructure
                           detection
    dynamic shoreline      water in one date but not the other, keep for
                           reclamation and water-level analysis
    invalid observation    cloud, cirrus, NoData, saturated, unreliable
                           registration, exclude from all metrics

A union water mask is deliberately NOT used. A union mask would mark an area
that was water in one date and reclaimed land in the other as water, which
would remove a target change.

The NDWI threshold is not stable across dates: the same threshold gives a
different water fraction on different dates. The water mask therefore uses a
combination of NDWI and the NIR band B08, and the thresholds are recorded.
"""

import numpy as np


def water_masks(ndwi1, ndwi2, ndwi_threshold=0.0,
                b08_1=None, b08_2=None, nir_threshold=None):
    """Build the water masks for two dates.

    Parameters
    ----------
    ndwi1, ndwi2 : numpy.ndarray
        McFeeters NDWI, (B03 - B08) / (B03 + B08), for date 1 and date 2.
    ndwi_threshold : float
        NDWI threshold. 0.0 is the physical boundary between water and land
        for the McFeeters index.
    b08_1, b08_2 : numpy.ndarray, optional
        NIR reflectance for date 1 and date 2. When given together with
        ``nir_threshold``, water must also satisfy B08 < nir_threshold.
    nir_threshold : float, optional
        NIR reflectance threshold. Clear water absorbs NIR strongly.

    Returns
    -------
    dict with stable, dynamic, land and the two raw water masks.
    """
    valid = np.isfinite(ndwi1) & np.isfinite(ndwi2)
    if b08_1 is not None and b08_2 is not None and nir_threshold is not None:
        valid &= np.isfinite(b08_1) & np.isfinite(b08_2)
        w1 = (ndwi1 > ndwi_threshold) & (b08_1 < nir_threshold) & valid
        w2 = (ndwi2 > ndwi_threshold) & (b08_2 < nir_threshold) & valid
    else:
        w1 = (ndwi1 > ndwi_threshold) & valid
        w2 = (ndwi2 > ndwi_threshold) & valid

    stable = w1 & w2
    dynamic = w1 ^ w2
    land = valid & ~w1 & ~w2

    return {
        "water_date1": w1,
        "water_date2": w2,
        "stable_water": stable,
        "dynamic_shoreline": dynamic,
        "land": land,
        "valid": valid,
        "fraction_stable": float(stable.sum()) / max(valid.sum(), 1),
        "fraction_dynamic": float(dynamic.sum()) / max(valid.sum(), 1),
        "fraction_water_date1": float(w1.sum()) / max(valid.sum(), 1),
        "fraction_water_date2": float(w2.sum()) / max(valid.sum(), 1),
    }


def invalid_mask(classi_flags, qualit_flags=None, extra=None):
    """Combine quality flags into one invalid-observation mask.

    Parameters
    ----------
    classi_flags : dict
        Keys "opaque_cloud", "cirrus", "snow_ice", values boolean arrays.
    qualit_flags : dict, optional
        Keys from MSK_QUALIT, values boolean arrays.
    extra : numpy.ndarray, optional
        Any additional invalid mask, for example a registration failure mask.

    Returns
    -------
    numpy.ndarray, bool
    """
    out = np.zeros_like(next(iter(classi_flags.values())), dtype=bool)
    for v in classi_flags.values():
        out |= v
    if qualit_flags:
        for v in qualit_flags.values():
            out |= v
    if extra is not None:
        out |= extra
    return out
