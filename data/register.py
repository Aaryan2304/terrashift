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


def _normalise(a, valid=None):
    """Zero-mean, unit-std copy of a float array, using valid pixels only.

    NaN becomes 0 before the statistics so the arithmetic stays finite, but
    the mean and standard deviation are computed over ``valid`` only. Pixels
    outside ``valid`` are set to 0 after normalisation, which is the mean of
    the valid population, so they add no correlation signal. They are also
    excluded by the ECC mask. Zero is never allowed to stand in for a real
    observation in the statistics.
    """
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if valid is None:
        mean, std = a.mean(), a.std()
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != a.shape:
            raise ValueError("valid mask must have the same shape as the image")
        if not valid.any():
            raise ValueError("valid mask has no True pixels")
        mean, std = float(a[valid].mean()), float(a[valid].std())
    out = (a - mean) / (std + 1e-6)
    if valid is not None:
        out[~valid] = 0.0
    return out


def estimate_shift(ref, mov, window_size=None, eps=1e-7, max_iter=500,
                   ref_valid=None, mov_valid=None, min_valid_pixels=64):
    """Estimate the translation that aligns ``mov`` to ``ref``.

    Parameters
    ----------
    ref, mov : numpy.ndarray
        Same shape, float or int.
    window_size : tuple, optional
        (rows, cols) of the central crop used for the estimate. The full
        arrays are used when this is None.
    ref_valid, mov_valid : numpy.ndarray of bool, optional
        Observation-validity masks on the same shape. They are cropped exactly
        like the images. Invalid pixels are excluded from the normalisation
        statistics, from the ECC objective (OpenCV documents ``inputMask`` as
        "an optional single channel mask to indicate valid values of
        inputImage"), and from the returned correlation. A shifted-in border
        that only contains reflected pixels is therefore never treated as
        evidence.

    Returns
    -------
    dict with dx, dy (pixels), correlation, converged, and n_valid_pixels.
    """
    if np.shape(ref) != np.shape(mov):
        raise ValueError("ref and mov must have the same shape")
    if ref_valid is not None and np.shape(ref_valid) != np.shape(ref):
        raise ValueError("ref_valid must have the same shape as ref")
    if mov_valid is not None and np.shape(mov_valid) != np.shape(mov):
        raise ValueError("mov_valid must have the same shape as mov")

    joint = None
    if ref_valid is not None or mov_valid is not None:
        joint = np.ones(np.shape(ref), dtype=bool)
        if ref_valid is not None:
            joint &= np.asarray(ref_valid, dtype=bool)
        if mov_valid is not None:
            joint &= np.asarray(mov_valid, dtype=bool)

    if window_size is not None:
        wr, wc = window_size
        h, w = np.shape(ref)
        r0 = max(0, (h - wr) // 2)
        c0 = max(0, (w - wc) // 2)
        ref = ref[r0:r0 + wr, c0:c0 + wc]
        mov = mov[r0:r0 + wr, c0:c0 + wc]
        if joint is not None:
            joint = joint[r0:r0 + wr, c0:c0 + wc]

    if joint is not None and int(joint.sum()) < min_valid_pixels:
        return {"dx": None, "dy": None, "correlation": None, "converged": False,
                "error": "insufficient valid pixels for ECC (%d)" % int(joint.sum()),
                "n_valid_pixels": int(joint.sum())}

    r = _normalise(ref, joint)
    m = _normalise(mov, joint)

    mask = None if joint is None else joint.astype(np.uint8) * 255

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    try:
        corr, warp = cv2.findTransformECC(r, m, warp, cv2.MOTION_TRANSLATION,
                                          criteria, mask, 5)
        return {"dx": float(warp[0, 2]), "dy": float(warp[1, 2]),
                "correlation": float(corr), "converged": True,
                "n_valid_pixels": None if joint is None else int(joint.sum())}
    except cv2.error as exc:
        return {"dx": None, "dy": None, "correlation": None,
                "converged": False, "error": str(exc)[:200],
                "n_valid_pixels": None if joint is None else int(joint.sum())}


def apply_shift(array, dx, dy, border_mode=cv2.BORDER_REFLECT):
    """Shift the content of an array.

    Positive dx moves the content to the right, positive dy moves it down.

    cv2.warpAffine inverts the matrix by default (unless WARP_INVERSE_MAP is
    set), so dst(x, y) = src(x - dx, y - dy) for the matrix used here.

    The border mode fills the strip that no source pixel reaches. Reflected
    values are finite and look like data. They may be kept for convolution
    context, but the caller must exclude them with
    :func:`coverage_mask`; they are never observed pixels.
    """
    if dx is None or dy is None:
        return array.copy()
    h, w = array.shape[:2]
    warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(array.astype(np.float32), warp, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=border_mode)


def coverage_mask(shape, shift_x, shift_y):
    """Footprint of an image shifted by ``(shift_x, shift_y)`` pixels.

    Pass the same displacement that :func:`apply_shift` receives. A pixel is
    covered only when the shifted image supplies a real source pixel for it,
    so the reflected border is marked invalid regardless of the border mode.
    """
    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1:
        raise ValueError("coverage shape must be positive")
    ones = np.ones((height, width), dtype=np.uint8)
    warp = np.array([[1.0, 0.0, float(shift_x)],
                     [0.0, 1.0, float(shift_y)]], dtype=np.float32)
    covered = cv2.warpAffine(ones, warp, (width, height),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return covered.astype(bool)


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


def residual_rmse(ref, mov_aligned, window_size=None, valid=None):
    """RMSE between the reference and the aligned moving image.

    A small RMSE is necessary but not sufficient: a smooth image can have a
    low RMSE and still be misaligned. Use it together with the residual shift.

    ``valid`` restricts the comparison to real overlap. Without it, the RMSE
    includes the reflected border, which is synthetic content and always
    agrees with itself.
    """
    r = _normalise(ref, valid)
    m = _normalise(mov_aligned, valid)
    if window_size is not None:
        wr, wc = window_size
        h, w = r.shape
        r0 = max(0, (h - wr) // 2)
        c0 = max(0, (w - wc) // 2)
        r = r[r0:r0 + wr, c0:c0 + wc]
        m = m[r0:r0 + wr, c0:c0 + wc]
        if valid is not None:
            valid = np.asarray(valid, dtype=bool)[r0:r0 + wr, c0:c0 + wc]
    diff = r - m
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if not valid.any():
            raise ValueError("valid mask has no True pixels in this window")
        diff = diff[valid]
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

    # A cloud over one corner must not influence the estimate: the masked
    # estimate has to recover the same translation from the remaining pixels.
    cloudy = moved.copy()
    cloudy[0:90, 120:256] = 5.0
    validity = np.ones((256, 256), dtype=bool)
    validity[0:90, 120:256] = False
    est_masked = estimate_shift(base, cloudy, mov_valid=validity)
    print("self-test: masked dx=%.2f dy=%.2f corr=%.4f" %
          (est_masked["dx"], est_masked["dy"], est_masked["correlation"]))
    ok = ok and est_masked["converged"] and abs(est_masked["dx"] - true_dx) < 0.5 \
        and abs(est_masked["dy"] - true_dy) < 0.5

    # Coverage: a shifted image cannot supply pixels near the trailing border.
    cov = coverage_mask((256, 256), -true_dx, -true_dy)
    print("self-test: covered pixels %d of %d" % (int(cov.sum()), cov.size))
    ok = ok and 0 < cov.sum() < cov.size and not cov[0, 255]

    print("self-test:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    self_test()
