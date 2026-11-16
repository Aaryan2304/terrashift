"""Spectral indices, with a single rule for undefined results.

A normalised difference ``(a - b) / (a + b)`` is undefined when ``a + b`` is
zero. That happens with real Sentinel-2 data: DN = 0 is the NO_DATA value, and
two NO_DATA pixels give ``0 + 0``.

Undefined results are ``NaN``, never ``0.0``. ``0.0`` is a *legitimate* index
value that means "the two bands are equal", so returning it for a division by
zero makes an invalid pixel indistinguishable from a real measurement, and
every downstream mean, histogram and threshold silently absorbs it.

``NaN`` has three useful properties here:

- ``np.mean`` over it returns ``NaN``, so the corruption is loud;
- ``np.nanmean`` excludes it by name;
- ``NaN`` compares ``False`` against any threshold, so an invalid pixel is
  never falsely flagged as change.

The bands are cast to ``float32`` before any arithmetic. Sentinel-2 stores
``uint16`` digital numbers, and ``uint16`` subtraction wraps around silently:
``1200 - 1600`` becomes ``65136``.
"""

import numpy as np

EPSILON = 1e-9


def normalized_difference(a, b, eps=EPSILON):
    """Return ``(a - b) / (a + b)`` as float32, with NaN where undefined.

    Parameters
    ----------
    a, b : array-like
        The two bands. They must be broadcastable to a common shape.
    eps : float
        The denominator is treated as zero when ``|a + b| <= eps``.

    Returns
    -------
    numpy.ndarray, float32
        The index. ``NaN`` where the denominator is zero or an input is NaN.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denominator = a + b
    out = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan, dtype=np.float32)
    np.divide(a - b, denominator, out=out, where=np.abs(denominator) > eps)
    return out


def ndvi(b08, b04):
    """NDVI = (B08 - B04) / (B08 + B04)."""
    return normalized_difference(b08, b04)


def ndwi(b03, b08):
    """McFeeters NDWI = (B03 - B08) / (B03 + B08).

    NDWI has more than one definition. This is the McFeeters definition, which
    uses the green band B03. The physical boundary between water and land for
    this index is 0.0.
    """
    return normalized_difference(b03, b08)


def ndbi(b11, b08):
    """NDBI = (B11 - B08) / (B11 + B08). Both bands are 20 m."""
    return normalized_difference(b11, b08)


if __name__ == "__main__":
    # 0/0 must be NaN, not 0.0.
    a = np.array([[0.0, 1000.0], [500.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1000.0], [500.0, 1.0]], dtype=np.float32)
    idx = ndwi(a, b)
    print("inputs:\\n", a, "\\n", b)
    print("ndwi:\\n", idx)
    assert np.isnan(idx[0, 0]), "0/0 must be NaN"
    assert idx[0, 1] == 0.0, "equal bands are a real 0.0"
    assert idx[1, 0] == 0.0, "equal bands are a real 0.0"
    assert idx[1, 1] == -1.0, "green 0 and NIR 1 is a real -1.0 (water)"
    print("0/0 -> NaN, equal bands -> 0.0, distinguishable              OK")

    # uint16 must not wrap.
    u1 = np.array([1200], dtype=np.uint16)
    u2 = np.array([1600], dtype=np.uint16)
    with np.errstate(over="ignore"):
        wrapped = int(u1[0] - u2[0])   # numpy uint16 arithmetic, which wraps
    print("uint16 %d - %d = %d (wraps), float32 = %.1f"
          % (u1[0], u2[0], wrapped, float(np.float32(u1[0]) - np.float32(u2[0]))))
    assert wrapped == 65136 and float(np.float32(u1[0]) - np.float32(u2[0])) == -400.0
    print("float32 arithmetic does not wrap                              OK")

    # NaN propagates and never passes a threshold.
    with_nan = ndwi(np.array([np.nan], dtype=np.float32), np.array([1.0], dtype=np.float32))
    assert np.isnan(with_nan[0])
    assert not (with_nan[0] > 0.0)
    print("NaN input -> NaN output, and NaN > threshold is False         OK")
