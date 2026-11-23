"""RGB rendering helpers shared by the review figures.

The stretch is computed once from a subsample of both dates and then applied to
both, so the two panels are directly comparable. A per-panel stretch would
hide a real brightness change between the dates.
"""

import numpy as np

RGB_BANDS = ("B04", "B03", "B02")


def rgb(product, shape):
    """True-colour RGB stack in red=B04, green=B03, blue=B02 order."""
    return np.stack([product.read_band(b, out_shape=shape)[0]
                     for b in RGB_BANDS], axis=-1)


def stretch_pair(first, second):
    """Apply one shared 2-98 percentile stretch to both RGB images."""
    limits = []
    for band in range(3):
        sample = np.concatenate([first[::8, ::8, band].ravel(),
                                 second[::8, ::8, band].ravel()])
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            raise ValueError("no finite pixels to stretch")
        limits.append(np.percentile(sample, [2, 98]))

    def apply(image):
        out = np.zeros_like(image, dtype=np.float32)
        for band, (low, high) in enumerate(limits):
            out[..., band] = np.clip((image[..., band] - low) / max(high - low, 1e-9),
                                     0.0, 1.0)
        return np.nan_to_num(out)

    return apply(first), apply(second)
