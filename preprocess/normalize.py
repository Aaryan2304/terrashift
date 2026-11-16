"""Relative radiometric normalisation between two dates.

The two dates of a pair have a systematic radiometric difference. It comes
from the sun angle, the atmosphere and the processing. The product-level
offset (RADIO_ADD_OFFSET) is already applied by ``data.safeloader``. What
remains is the inter-date difference.

Four methods are implemented:

    none       no correction
    global     one straight line per band, fitted on all valid land pixels
    pif        select pseudo-invariant features first, then fit per band
    histogram  per-band quantile (histogram) matching

All methods work on a common grid. The caller supplies the bands as arrays
with the same shape. Invalid pixels must be NaN; they are excluded from the
fit and from the metrics.

Reference: Canty and Nielsen, "Automatic radiometric normalization of
multitemporal satellite imagery with the iteratively re-weighted MAD
transformation", Remote Sensing of Environment, 2008.
"""

import numpy as np


def _valid(*arrays):
    """Return a mask of pixels that are finite in every array."""
    mask = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def change_score(ref_bands, mov_bands):
    """A per-pixel change magnitude.

    The bands are first normalised by their own standard deviation so that no
    single band dominates. The score is the Euclidean norm of the differences.
    """
    score = np.zeros(ref_bands[0].shape, dtype=np.float32)
    for r, m in zip(ref_bands, mov_bands):
        valid = _valid(r, m)
        rs = np.nanstd(r) + 1e-9
        ms = np.nanstd(m) + 1e-9
        d = (r / rs) - (m / ms)
        score += np.where(valid, d * d, 0.0)
    return np.sqrt(score)


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
    score = change_score(ref_bands, mov_bands)
    sv = score[valid & np.isfinite(score)]
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
        score = change_score(ref_bands, mov_bands)
        pif = select_pifs(score, valid, fraction=pif_fraction)
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
