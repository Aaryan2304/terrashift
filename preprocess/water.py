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
                b08_1=None, b08_2=None, nir_threshold=None,
                observation_valid=None):
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
    observation_valid : numpy.ndarray of bool, optional
        Paired observation validity from :mod:`data.validity`. Finite NDWI is
        not the same as a valid observation: a cloud pixel has a finite NDWI
        and would otherwise be classified as land or water and enter the
        statistics. When supplied, it is ANDed into ``valid``.

    Returns
    -------
    dict with stable, dynamic, land and the two raw water masks.

    Notes
    -----
    A water-to-land transition stays in ``dynamic_shoreline`` and stays
    *valid*. It may be a real reclamation or construction event, so it is not
    excluded. Only quality, NoData, and registration coverage remove a pixel
    from the comparison.
    """
    valid = np.isfinite(ndwi1) & np.isfinite(ndwi2)
    if b08_1 is not None and b08_2 is not None and nir_threshold is not None:
        valid &= np.isfinite(b08_1) & np.isfinite(b08_2)
    if observation_valid is not None:
        observation_valid = np.asarray(observation_valid, dtype=bool)
        if observation_valid.shape != valid.shape:
            raise ValueError("observation_valid must have the same shape as the indices")
        valid &= observation_valid
    if b08_1 is not None and b08_2 is not None and nir_threshold is not None:
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


def thresholds_from_config(cfg):
    """Read the water thresholds from the configuration, with validation.

    The thresholds used to be duplicated as module constants in
    ``scripts/run_classical_baseline.py`` and ``scripts/run_normalization.py``,
    so editing ``config.yaml`` changed nothing. This is now the single source.
    """
    thresholds = cfg["preprocess"]["thresholds"]
    try:
        ndwi_value = float(thresholds["water_ndwi"])
        nir_value = float(thresholds["water_nir_max"])
    except KeyError as exc:
        raise ValueError("config is missing a water threshold: %s" % exc) from exc
    if not -1.0 <= ndwi_value <= 1.0:
        raise ValueError("water_ndwi must lie in [-1, 1], found %r" % ndwi_value)
    if not 0.0 < nir_value < 1.0:
        raise ValueError("water_nir_max must lie in (0, 1), found %r" % nir_value)
    return {"ndwi_threshold": ndwi_value, "nir_threshold": nir_value}


def classify(water, observation_valid=None):
    """Water-state classes as uint8, one class per pixel.

    Values: 0 = invalid observation, 1 = land, 2 = stable water,
    3 = dynamic shoreline. The shoreline keeps its own class because a
    water-to-land transition can be a real reclamation change, so it must not
    be folded into "invalid" or into "land".
    """
    shape = water["land"].shape
    classes = np.ones(shape, dtype=np.uint8)
    classes[water["stable_water"]] = 2
    classes[water["dynamic_shoreline"]] = 3
    classes[~water["valid"]] = 0
    if observation_valid is not None:
        observation_valid = np.asarray(observation_valid, dtype=bool)
        if observation_valid.shape != shape:
            raise ValueError("observation_valid must match the water masks")
        classes[~observation_valid] = 0
    return classes


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
