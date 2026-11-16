"""Registration of two Sentinel-2 dates.

The two products of a pair share the same tile grid, but a small sub-pixel
shift remains. This module measures that shift with Enhanced Correlation
Coefficient (ECC) minimisation on band B08 at 10 m, applies a global
translation, and measures the residual displacement afterwards.

Important: ECC must run on land windows. Over water the texture is unstable
and ECC returns shifts of several kilometres.

Direction convention
--------------------
``cv2.findTransformECC(reference, moving, warp)`` returns a warp whose
translation offsets describe how the content of ``moving`` is displaced
relative to ``reference``. Here the reference is date 1 and the moving image
is date 2, so the offsets say how far date 2 is shifted from date 1.

To cancel that displacement the content must move the opposite way, so
``align_to_reference`` applies the negated offsets. The self-test checks the
alignment RMSE, not only the estimate.

Note on ``cv2.warpAffine``: it inverts the transformation matrix by default
unless ``WARP_INVERSE_MAP`` is set. The matrices here are written for that
default behaviour.

Reference: OpenCV findTransformECC,
https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
"""

import numpy as np
import cv2


def _normalise(a):
    """Zero-mean, unit-std copy of a float array. NaN becomes 0."""
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def estimate_shift(ref, mov, window_size=None, eps=1e-7, max_iter=500):
    """Estimate the translation that aligns ``mov`` to ``ref``.

    Parameters
    ----------
    ref, mov : numpy.ndarray
        Same shape, float or int.
    window_size : tuple, optional
        (rows, cols) of the central crop used for the estimate. The full
        arrays are used when this is None.

    Returns
    -------
    dict with dx, dy (pixels), correlation, converged.
    """
    r = _normalise(ref)
    m = _normalise(mov)
    if r.shape != m.shape:
        raise ValueError("ref and mov must have the same shape")

    if window_size is not None:
        wr, wc = window_size
        h, w = r.shape
        r0 = max(0, (h - wr) // 2)
        c0 = max(0, (w - wc) // 2)
        r = r[r0:r0 + wr, c0:c0 + wc]
        m = m[r0:r0 + wr, c0:c0 + wc]

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    try:
        corr, warp = cv2.findTransformECC(r, m, warp, cv2.MOTION_TRANSLATION,
                                          criteria, None, 5)
        return {"dx": float(warp[0, 2]), "dy": float(warp[1, 2]),
                "correlation": float(corr), "converged": True}
    except cv2.error as exc:
        return {"dx": None, "dy": None, "correlation": None,
                "converged": False, "error": str(exc)[:200]}


def apply_shift(array, dx, dy, border_mode=cv2.BORDER_REFLECT):
    """Shift the content of an array.

    Positive dx moves the content to the right, positive dy moves it down.

    cv2.warpAffine inverts the matrix by default (unless WARP_INVERSE_MAP is
    set), so dst(x, y) = src(x - dx, y - dy) for the matrix used here.
    """
    if dx is None or dy is None:
        return array.copy()
    h, w = array.shape[:2]
    warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(array.astype(np.float32), warp, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=border_mode)


def align_to_reference(mov, dx, dy, border_mode=cv2.BORDER_REFLECT):
    """Apply the ECC estimate to align ``mov`` to the reference.

    ``cv2.findTransformECC(reference, moving, warp)`` returns the warp matrix
    that, when applied to ``moving``, aligns it with ``reference``. The
    offsets it returns are the content displacement of ``moving`` relative to
    ``reference``. To cancel that displacement the content must move by the
    opposite amount, so this calls ``apply_shift(mov, -dx, -dy)``.
    """
    return apply_shift(mov, -dx, -dy, border_mode=border_mode)


def summarise(results, pixel_size_m):
    """Summarise a list of estimate_shift results."""
    ok = [r for r in results if r["converged"]]
    failed = [r for r in results if not r["converged"]]
    out = {"n_windows": len(results), "n_converged": len(ok),
           "n_failed": len(failed), "failed": failed}
    if ok:
        dxs = np.array([r["dx"] for r in ok])
        dys = np.array([r["dy"] for r in ok])
        corrs = np.array([r["correlation"] for r in ok])
        out.update({
            "median_dx_px": float(np.median(dxs)),
            "median_dy_px": float(np.median(dys)),
            "std_dx_px": float(dxs.std()),
            "std_dy_px": float(dys.std()),
            "median_dx_m": float(np.median(dxs) * pixel_size_m),
            "median_dy_m": float(np.median(dys) * pixel_size_m),
            "std_dx_m": float(dxs.std() * pixel_size_m),
            "std_dy_m": float(dys.std() * pixel_size_m),
            "median_correlation": float(np.median(corrs)),
            "min_correlation": float(corrs.min()),
            "max_correlation": float(corrs.max()),
            "n_above_0_5": int((corrs > 0.5).sum()),
        })
    return out


def residual_rmse(ref, mov_aligned, window_size=None):
    """RMSE between the reference and the aligned moving image.

    A small RMSE is necessary but not sufficient: a smooth image can have a
    low RMSE and still be misaligned. Use it together with the residual shift.
    """
    r = _normalise(ref)
    m = _normalise(mov_aligned)
    if window_size is not None:
        wr, wc = window_size
        h, w = r.shape
        r0 = max(0, (h - wr) // 2)
        c0 = max(0, (w - wc) // 2)
        r = r[r0:r0 + wr, c0:c0 + wc]
        m = m[r0:r0 + wr, c0:c0 + wc]
    diff = r - m
    return float(np.sqrt(np.mean(diff ** 2)))


def self_test():
    """Verify the direction convention and the alignment with a known shift."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:256, 0:256].astype(np.float32)
    base = (np.sin(x / 7.0) + np.cos(y / 5.0)
            + 0.3 * rng.normal(0, 1, (256, 256))).astype(np.float32)
    true_dx, true_dy = 3.0, -2.0
    moved = apply_shift(base, true_dx, true_dy)
    est = estimate_shift(base, moved)
    aligned = align_to_reference(moved, est["dx"], est["dy"])
    rmse_before = residual_rmse(base, moved)
    rmse_after = residual_rmse(base, aligned)
    print("self-test: true dx=%.1f dy=%.1f" % (true_dx, true_dy))
    print("self-test: estimated dx=%.2f dy=%.2f corr=%.4f converged=%s" %
          (est["dx"], est["dy"], est["correlation"], est["converged"]))
    print("self-test: rmse before=%.4f after=%.4f" % (rmse_before, rmse_after))
    ok = (abs(est["dx"] - true_dx) < 0.3 and abs(est["dy"] - true_dy) < 0.3
          and est["converged"] and rmse_after < rmse_before)
    print("self-test:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    self_test()
