"""Measure and correct the registration of each pair.

Reads band B08 at 10 m over several land windows, estimates the translation
that aligns date 2 to date 1 with ECC, applies the median translation, and
measures the residual displacement on the same windows.

The residual is measured on the same windows that produced the estimate, so
it is a consistency check, not an independent validation set. A truly
independent check would need separate control features.

Output: outputs/audit/registration.json
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.safeloader import SafeProduct
from data.register import estimate_shift, align_to_reference, summarise, residual_rmse

SAFE_DIR = "/home/aaryan/projects/skyserve"
OUT = "outputs/audit/registration.json"

PAIRS = {
    "A": {
        "tile": "T43QBB",
        "date1": "S2A_MSIL1C_20221223T054241_N0510_R005_T43QBB_20240809T025636.SAFE",
        "date2": "S2A_MSIL1C_20230512T053641_N0510_R005_T43QBB_20240908T023323.SAFE",
        "windows": [
            (2000, 4000, 5000, 7000),
            (4000, 6000, 6000, 8000),
            (6000, 8000, 5000, 7000),
            (7000, 9000, 4000, 6000),
            (3000, 5000, 7000, 9000),
        ],
    },
    "B": {
        "tile": "T43RGM",
        "date1": "S2A_MSIL1C_20231006T052701_N0510_R105_T43RGM_20241101T065943.SAFE",
        "date2": "S2A_MSIL1C_20261022T053241_N0513_R105_T43RGM_20261022T091024.SAFE",
        "windows": [
            (2000, 4000, 2000, 4000),
            (4000, 6000, 4000, 6000),
            (6000, 8000, 6000, 8000),
            (3000, 5000, 7000, 9000),
            (5000, 7000, 2000, 4000),
        ],
    },
}

BAND = "B08"
PIXEL_SIZE_M = 10.0


def read_window(product, window):
    r0, r1, c0, c1 = window
    refl, _, _ = product.read_band(BAND, window=((r0, r1), (c0, c1)))
    return refl


def run_pair(name, cfg):
    p1 = SafeProduct(os.path.join(SAFE_DIR, cfg["date1"]))
    p2 = SafeProduct(os.path.join(SAFE_DIR, cfg["date2"]))

    windows = []
    for w in cfg["windows"]:
        r0, r1, c0, c1 = w
        ref = read_window(p1, w)
        mov = read_window(p2, w)
        est = estimate_shift(ref, mov)
        est["window"] = list(w)
        windows.append(est)
        print("  pair %s window %s: dx=%s dy=%s corr=%s converged=%s" % (
            name, w,
            None if est["dx"] is None else round(est["dx"], 3),
            None if est["dy"] is None else round(est["dy"], 3),
            None if est["correlation"] is None else round(est["correlation"], 3),
            est["converged"]))

    summary = summarise(windows, PIXEL_SIZE_M)
    summary["windows"] = windows

    # Apply the median shift and measure the residual on the same windows.
    if "median_dx_px" in summary:
        dx, dy = summary["median_dx_px"], summary["median_dy_px"]
        residuals = []
        for w in cfg["windows"]:
            ref = read_window(p1, w)
            mov = read_window(p2, w)
            mov_aligned = align_to_reference(mov, dx, dy)
            res = estimate_shift(ref, mov_aligned)
            res["window"] = list(w)
            res["rmse"] = residual_rmse(ref, mov_aligned)
            residuals.append(res)
            print("  pair %s residual %s: dx=%s dy=%s corr=%s rmse=%.4f" % (
                name, w,
                None if res["dx"] is None else round(res["dx"], 3),
                None if res["dy"] is None else round(res["dy"], 3),
                None if res["correlation"] is None else round(res["correlation"], 3),
                res["rmse"]))
        ok_res = [r for r in residuals if r["converged"]]
        if ok_res:
            rdx = np.array([r["dx"] for r in ok_res])
            rdy = np.array([r["dy"] for r in ok_res])
            rrmse = np.array([r["rmse"] for r in ok_res])
            summary["residual"] = {
                "n_converged": len(ok_res),
                "n_failed": len(residuals) - len(ok_res),
                "median_dx_px": float(np.median(rdx)),
                "median_dy_px": float(np.median(rdy)),
                "median_dx_m": float(np.median(rdx) * PIXEL_SIZE_M),
                "median_dy_m": float(np.median(rdy) * PIXEL_SIZE_M),
                "rmse_mean": float(rrmse.mean()),
                "rmse_max": float(rrmse.max()),
                "windows": residuals,
            }
        summary["applied_shift_px"] = [dx, dy]
        summary["applied_shift_m"] = [dx * PIXEL_SIZE_M, dy * PIXEL_SIZE_M]

    return summary


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    result = {}
    for name, cfg in PAIRS.items():
        print("=" * 70)
        print("PAIR", name, cfg["tile"])
        result[name] = {"tile": cfg["tile"],
                        "date1": cfg["date1"], "date2": cfg["date2"],
                        "band": BAND, "pixel_size_m": PIXEL_SIZE_M}
        result[name]["initial"] = run_pair(name, cfg)

    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
