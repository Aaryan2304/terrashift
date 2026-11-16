"""Find water thresholds that behave consistently across the two dates.

The NDWI threshold is not stable across dates: the same threshold gives a
different water fraction. This script measures the water fraction for a range
of thresholds and for a combined NDWI + NIR rule, so the audit can record a
justified choice.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.safeloader import SafeProduct

SAFE_DIR = "/home/aaryan/projects/skyserve"
PAIRS = {
    "A": ("S2A_MSIL1C_20221223T054241_N0510_R005_T43QBB_20240809T025636.SAFE",
          "S2A_MSIL1C_20230512T053641_N0510_R005_T43QBB_20240908T023323.SAFE"),
    "B": ("S2A_MSIL1C_20231006T052701_N0510_R105_T43RGM_20241101T065943.SAFE",
          "S2A_MSIL1C_20261022T053241_N0513_R105_T43RGM_20261022T091024.SAFE"),
}
OUT_SHAPE = (1098, 1098)   # 100 m grid, fast overview


def read(prod, band):
    r, _, _ = prod.read_band(band, out_shape=OUT_SHAPE)
    return r


def ndwi(b03, b08):
    denom = b03 + b08
    return np.divide(b03 - b08, denom, out=np.zeros_like(b03, dtype=np.float32),
                     where=np.abs(denom) > 1e-9)


def main():
    for name, (s1, s2) in PAIRS.items():
        p1 = SafeProduct(os.path.join(SAFE_DIR, s1))
        p2 = SafeProduct(os.path.join(SAFE_DIR, s2))
        b03_1, b08_1 = read(p1, "B03"), read(p1, "B08")
        b03_2, b08_2 = read(p2, "B03"), read(p2, "B08")
        w1 = ndwi(b03_1, b08_1)
        w2 = ndwi(b03_2, b08_2)
        valid = np.isfinite(w1) & np.isfinite(w2)
        print("=" * 80)
        print("PAIR", name, " valid pixels", int(valid.sum()))
        print("  NDWI date1 mean %.4f  date2 mean %.4f" %
              (float(np.nanmean(w1)), float(np.nanmean(w2))))
        print("  NDWI threshold sweep (water fraction %):")
        print("    threshold   date1     date2     abs diff")
        for t in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            f1 = float(((w1 > t) & valid).sum()) / valid.sum() * 100
            f2 = float(((w2 > t) & valid).sum()) / valid.sum() * 100
            print("    %+.1f       %6.2f    %6.2f    %6.2f" % (t, f1, f2, abs(f1 - f2)))
        print("  NIR (B08) threshold sweep (water fraction %):")
        for t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            f1 = float(((b08_1 < t) & valid).sum()) / valid.sum() * 100
            f2 = float(((b08_2 < t) & valid).sum()) / valid.sum() * 100
            print("    %.2f       %6.2f    %6.2f    %6.2f" % (t, f1, f2, abs(f1 - f2)))
        print("  Combined NDWI > 0.0 and B08 < t:")
        for t in [0.08, 0.10, 0.12, 0.15]:
            f1 = float(((w1 > 0.0) & (b08_1 < t) & valid).sum()) / valid.sum() * 100
            f2 = float(((w2 > 0.0) & (b08_2 < t) & valid).sum()) / valid.sum() * 100
            print("    %.2f       %6.2f    %6.2f    %6.2f" % (t, f1, f2, abs(f1 - f2)))


if __name__ == "__main__":
    main()
