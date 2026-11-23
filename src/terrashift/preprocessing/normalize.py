"""Relative radiometric normalisation between two dates.

What remains after the product offset (applied in ``safeloader``) comes from sun angle, atmosphere
and processing. Methods: ``none``, ``global`` (one line per band, fitted on valid land pixels),
``pif`` (pseudo-invariant features first, then fit per band), ``histogram`` (per-band quantile
matching). All take arrays on a common grid; invalid pixels must be NaN and stay out of the fit
and the metrics.

Reference: Canty and Nielsen, Remote Sensing of Environment, 2008.
"""

import numpy as np


def reference_scaled_change_score(ref_bands, mov_bands, valid):
    """Euclidean spectral difference, scaled by the reference date's spread.

    This is the single change-score definition used for PIF selection, for
    candidate ranking, and for the normalisation comparison. An earlier
    version of this module standardised each date independently; that makes
    the score depend on both scene-wide distributions, so a scene-wide
    brightening can hide or invent change. One scale per band from the
    reference date, computed on ``valid`` pixels only, keeps the score
    comparable between dates.

    Parameters
    ----------
    ref_bands, mov_bands : sequence of numpy.ndarray
        Matched band stacks, same shape.
    valid : numpy.ndarray of bool
        The pixels that may enter the score *and* its scale estimate. Pass the
        paired observation-valid mask, not a finite-value mask.

    Returns
    -------
    (score, complete) : (float32 array, bool array)
        ``score`` is NaN outside ``complete``. ``complete`` is the intersection
        of ``valid`` and the finite footprint of every band.

    Raises
    ------
    ValueError
        If ``valid`` is empty, or a reference band has zero or invalid spread
        on the valid pixels. There is no silent fallback: a score normalised
        by a zero spread would be meaningless.
    """
    if len(ref_bands) != len(mov_bands):
        raise ValueError("reference and moving band counts differ")
    if not ref_bands:
        raise ValueError("at least one band is required")
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != np.shape(ref_bands[0]):
        raise ValueError("valid mask shape does not match the bands")

    complete = valid.copy()
    for band in list(ref_bands) + list(mov_bands):
        complete &= np.isfinite(band)
    if not complete.any():
        raise ValueError("no valid pixels remain for the change score")

    score_sq = np.zeros(valid.shape, dtype=np.float64)
    for ref, mov in zip(ref_bands, mov_bands):
        scale = float(np.std(ref[complete], dtype=np.float64))
        if not np.isfinite(scale) or scale <= 1e-9:
            raise ValueError("reference band has zero or invalid spread on valid pixels")
        delta = (np.asarray(ref, dtype=np.float64) - np.asarray(mov, dtype=np.float64)) / scale
        score_sq += np.where(complete, delta * delta, 0.0)
    score = np.sqrt(score_sq).astype(np.float32)
    score[~complete] = np.nan
    return score, complete


def select_pifs(change, valid, fraction=0.5):
    """Select the most stable pixels (pseudo-invariant features).

    ``fraction`` is the fraction of valid pixels with the lowest change score.
    """
    thresh = np.nanpercentile(change[valid], fraction * 100.0)
    return valid & (change <= thresh)


def fit_global(ref_bands, mov_bands, valid):
    """Fit one straight line per band: ref = a * mov + b."""
    params = {}
    for i, (r, m) in enumerate(zip(ref_bands, mov_bands)):
        mask = valid & np.isfinite(r) & np.isfinite(m)
        x = m[mask].astype(np.float64)
        y = r[mask].astype(np.float64)
        if x.size < 10:
            params[i] = (1.0, 0.0, 0.0)
            continue
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        params[i] = (float(coef[0]), float(coef[1]), float(r2))
    return params


def apply_linear(mov_bands, params):
    """Apply a per-band linear correction."""
    out = []
    for i, m in enumerate(mov_bands):
        a, b, _ = params.get(i, (1.0, 0.0, 0.0))
        out.append(a * m + b)
    return out


def fit_histogram(ref_bands, mov_bands, valid, n_bins=256):
    """Per-band quantile matching tables.

    For each band, store the reference quantiles and the moving quantiles.
    The moving values are mapped to the reference values by interpolation.
    """
    tables = {}
    for i, (r, m) in enumerate(zip(ref_bands, mov_bands)):
        mask = valid & np.isfinite(r) & np.isfinite(m)
        rv = r[mask]
        mv = m[mask]
        if rv.size < 10:
            tables[i] = None
            continue
        q = np.linspace(0, 1, n_bins)
        ref_q = np.quantile(rv, q)
        mov_q = np.quantile(mv, q)
        tables[i] = (mov_q.astype(np.float32), ref_q.astype(np.float32))
    return tables


def apply_histogram(mov_bands, tables):
    """Apply per-band quantile matching."""
    out = []
    for i, m in enumerate(mov_bands):
        t = tables.get(i)
        if t is None:
            out.append(m.copy())
            continue
        mov_q, ref_q = t
        flat = m.ravel()
        valid = np.isfinite(flat)
        res = flat.copy()
        idx = np.searchsorted(mov_q, flat[valid], side="left")
        idx = np.clip(idx, 1, len(mov_q) - 1)
        x0, x1 = mov_q[idx - 1], mov_q[idx]
        y0, y1 = ref_q[idx - 1], ref_q[idx]
        w = np.where(x1 > x0, (flat[valid] - x0) / (x1 - x0 + 1e-12), 0.0)
        res[valid] = y0 + w * (y1 - y0)
        out.append(res.reshape(m.shape))
    return out


def evaluate(ref_bands, mov_bands, valid):
    """Mean absolute difference and change-score statistics on valid pixels."""
    abs_diff = []
    for r, m in zip(ref_bands, mov_bands):
        mask = valid & np.isfinite(r) & np.isfinite(m)
        if mask.sum() == 0:
            continue
        abs_diff.append(np.abs(r[mask] - m[mask]))
    if not abs_diff:
        return {}
    ad = np.concatenate(abs_diff)
    score, complete = reference_scaled_change_score(ref_bands, mov_bands, valid)
    sv = score[complete]
    return {
        "mae_mean": float(ad.mean()),
        "mae_median": float(np.median(ad)),
        "change_score_mean": float(sv.mean()) if sv.size else None,
        "change_score_std": float(sv.std()) if sv.size else None,
        "n_valid": int(valid.sum()),
    }


def run(ref_bands, mov_bands, valid, method, pif_fraction=0.5):
    """Fit and apply one normalisation method.

    Returns (corrected_bands, params, metrics_before, metrics_after).
    """
    before = evaluate(ref_bands, mov_bands, valid)

    if method == "none":
        corrected = [m.copy() for m in mov_bands]
        params = {"method": "none"}
    elif method == "global":
        params = fit_global(ref_bands, mov_bands, valid)
        params["method"] = "global"
        corrected = apply_linear(mov_bands, params)
    elif method == "pif":
        score, complete = reference_scaled_change_score(ref_bands, mov_bands, valid)
        pif = select_pifs(score, complete, fraction=pif_fraction)
        params = fit_global(ref_bands, mov_bands, pif)
        params["method"] = "pif"
        params["pif_pixels"] = int(pif.sum())
        params["pif_fraction"] = float(pif.sum() / max(valid.sum(), 1))
        corrected = apply_linear(mov_bands, params)
    elif method == "histogram":
        params = fit_histogram(ref_bands, mov_bands, valid)
        params["method"] = "histogram"
        corrected = apply_histogram(mov_bands, params)
    else:
        raise ValueError("unknown normalisation method: " + str(method))

    after = evaluate(ref_bands, corrected, valid)
    return corrected, params, before, after
