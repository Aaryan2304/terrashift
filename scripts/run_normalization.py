"""Compare radiometric normalisation methods on held-out stable land targets.

For each pair:
  1. read the four native 10 m bands of both dates onto a common 100 m grid
  2. compute a per-pixel change score
  3. exclude water with the stable-water mask
  4. select stable pixels (lowest change score)
  5. split the stable pixels spatially into a fit set and a validation set
  6. fit each method on the fit set
  7. evaluate each method on the validation set

Methods: none, global, pif, histogram.

Output: outputs/audit/normalization_comparison.csv and
        outputs/audit/normalization_parameters.json
"""

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.safeloader import SafeProduct
from preprocess.normalize import (change_score, select_pifs, fit_global,
                                  apply_linear, fit_histogram, apply_histogram,
                                  evaluate)
from preprocess.water import water_masks

SAFE_DIR = "/home/aaryan/projects/skyserve"
OUT_CSV = "outputs/audit/normalization_comparison.csv"
OUT_JSON = "outputs/audit/normalization_parameters.json"

PAIRS = {
    "A": ("S2A_MSIL1C_20221223T054241_N0510_R005_T43QBB_20240809T025636.SAFE",
          "S2A_MSIL1C_20230512T053641_N0510_R005_T43QBB_20240908T023323.SAFE"),
    "B": ("S2A_MSIL1C_20231006T052701_N0510_R105_T43RGM_20241101T065943.SAFE",
          "S2A_MSIL1C_20261022T053241_N0513_R105_T43RGM_20261022T091024.SAFE"),
}
BANDS = ["B02", "B03", "B04", "B08"]
OUT_SHAPE = (1098, 1098)          # 100 m grid
NDWI_THRESHOLD = 0.0              # McFeeters physical boundary
NIR_THRESHOLD = 0.15              # B08 reflectance, clear water absorbs NIR
STABLE_FRACTION = 0.5             # lowest-change half of the land pixels
SEED = 42


def read_bands(product, bands, out_shape=OUT_SHAPE):
    out = {}
    for b in bands:
        a, _, _ = product.read_band(b, out_shape=out_shape)
        out[b] = a
    return out


def spatial_split(mask, block=8):
    """Split a boolean mask into two spatially interleaved sets."""
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    checker = ((yy // block) + (xx // block)) % 2 == 0
    return mask & checker, mask & ~checker


def ndwi(b03, b08):
    denom = b03 + b08
    return np.divide(b03 - b08, denom, out=np.zeros_like(b03, dtype=np.float32),
                     where=np.abs(denom) > 1e-9)


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    rows = []
    params_all = {}

    for name, (s1, s2) in PAIRS.items():
        p1 = SafeProduct(os.path.join(SAFE_DIR, s1))
        p2 = SafeProduct(os.path.join(SAFE_DIR, s2))
        b1 = read_bands(p1, BANDS)
        b2 = read_bands(p2, BANDS)

        ref = [b1[b] for b in BANDS]
        mov = [b2[b] for b in BANDS]

        w1 = ndwi(b1["B03"], b1["B08"])
        w2 = ndwi(b2["B03"], b2["B08"])
        wm = water_masks(w1, w2, ndwi_threshold=NDWI_THRESHOLD,
                         b08_1=b1["B08"], b08_2=b2["B08"],
                         nir_threshold=NIR_THRESHOLD)
        valid_land = wm["land"]
        print("=" * 80)
        print("PAIR", name, "valid land pixels:", int(valid_land.sum()),
              "stable water:", int(wm["stable_water"].sum()),
              "dynamic:", int(wm["dynamic_shoreline"].sum()))

        score = change_score(ref, mov)
        stable = select_pifs(score, valid_land, fraction=STABLE_FRACTION)
        fit_mask, val_mask = spatial_split(stable)
        print("  stable pixels:", int(stable.sum()),
              "fit:", int(fit_mask.sum()), "validation:", int(val_mask.sum()))

        params_all[name] = {"n_stable": int(stable.sum()),
                            "n_fit": int(fit_mask.sum()),
                            "n_validation": int(val_mask.sum()),
                            "water_fraction_stable": wm["fraction_stable"],
                            "water_fraction_dynamic": wm["fraction_dynamic"]}

        for method in ["none", "global", "pif", "histogram"]:
            if method == "none":
                corrected = [m.copy() for m in mov]
                params = {}
            elif method == "global":
                params = fit_global(ref, mov, fit_mask)
                corrected = apply_linear(mov, params)
            elif method == "pif":
                pif = select_pifs(score, fit_mask, fraction=STABLE_FRACTION)
                params = fit_global(ref, mov, pif)
                params["pif_pixels"] = int(pif.sum())
                corrected = apply_linear(mov, params)
            elif method == "histogram":
                params = fit_histogram(ref, mov, fit_mask)
                corrected = apply_histogram(mov, params)

            before = evaluate(ref, mov, val_mask)
            after = evaluate(ref, corrected, val_mask)

            row = {
                "pair": name,
                "method": method,
                "mae_before": before.get("mae_mean"),
                "mae_after": after.get("mae_mean"),
                "mae_reduction_pct": (100.0 * (before["mae_mean"] - after["mae_mean"])
                                      / before["mae_mean"]) if before.get("mae_mean") else None,
                "change_score_before": before.get("change_score_mean"),
                "change_score_after": after.get("change_score_mean"),
                "change_score_std_before": before.get("change_score_std"),
                "change_score_std_after": after.get("change_score_std"),
                "n_validation": int(val_mask.sum()),
            }
            rows.append(row)
            print("  %-10s MAE %.5f -> %.5f  (%.1f%%)  change score %.4f -> %.4f" % (
                method, before["mae_mean"], after["mae_mean"],
                row["mae_reduction_pct"], before["change_score_mean"],
                after["change_score_mean"]))

            # per-band fit parameters
            if method in ("global", "pif"):
                params_all[name][method] = {
                    "bands": {BANDS[i]: {"a": params[i][0], "b": params[i][1],
                                         "r2": params[i][2]}
                              for i in range(len(BANDS)) if i in params}
                }

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_JSON, "w") as fh:
        json.dump(params_all, fh, indent=1)
    print("\nwrote", OUT_CSV)
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
