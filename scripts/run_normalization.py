"""Compare radiometric normalisation methods on held-out stable land targets.

For each pair:
  1. build a 100 m analysis grid from date 1's native 10 m bands
  2. build the paired observation-valid mask (source NoData, MSK_CLASSI,
     MSK_QUALIT, registration coverage)
  3. read the four native 10 m bands of both dates onto that grid and apply
     the measured registration shift to date 2
  4. classify water on the paired-valid footprint only
  5. select stable pixels (lowest reference-scaled change score)
  6. split the stable pixels spatially into a fit set and a validation set
  7. fit each method on the fit set
  8. evaluate each method on the validation set

Methods: none, global, pif, histogram.

Every statistic uses the paired-valid footprint. Invalid pixels never enter a
fit, a validation metric, or a PIF selection.

Known limitation: the fit/validation split is an interleaved block pattern.
Interleaved blocks are not geographically independent, so the validation
figures measure agreement on nearby pixels, not generalisation to another
area. The docstring of ``spatial_split`` says so; it is not fixed here.

Output: <output-dir>/normalization_comparison.csv and
        <output-dir>/normalization_parameters.json.
Nothing is overwritten; pass a new ``--output-dir`` for a fresh run.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.grids import grid_from_band
from data.manifest import load_manifest, pair_paths
from data.register import align_to_reference
from data.safeloader import SafeProduct
from data.validity import build_paired_validity
from preprocess.indices import ndwi
from preprocess.normalize import (apply_histogram, apply_linear, evaluate,
                                  fit_global, fit_histogram,
                                  reference_scaled_change_score, select_pifs)
from preprocess.water import thresholds_from_config, water_masks

DEFAULT_OUT_DIR = "outputs/audit"
REG_PATH = "outputs/audit/registration_v2_observation_valid/registration.json"

MANIFEST = load_manifest()
PAIRS = {name: pair_paths(MANIFEST, name) for name in sorted(MANIFEST["pairs"])}
BANDS = ["B02", "B03", "B04", "B08"]
GRID_M = 100                     # comparison grid, metres
SOURCE_RES = 10                  # native resolution of the bands
STABLE_FRACTION = 0.5            # lowest-change half of the land pixels
SEED = 42


def read_bands(product, bands, shape):
    out = {}
    for b in bands:
        a, _, _ = product.read_band(b, out_shape=shape)
        out[b] = a
    return out


def spatial_split(mask, block=8):
    """Split a boolean mask into two interleaved block sets.

    Interleaved blocks are **not** geographically independent: neighbouring
    fit and validation blocks share the same land use and often the same
    building. Treat the validation numbers as an agreement check on nearby
    pixels, not as a generalisation estimate.
    """
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    checker = ((yy // block) + (xx // block)) % 2 == 0
    return mask & checker, mask & ~checker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR,
                        help="directory for the CSV and JSON; existing files "
                             "are never replaced")
    parser.add_argument("--registration", default=REG_PATH)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml",
                        help="configuration with the water thresholds")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "normalization_comparison.csv")
    out_json = os.path.join(args.output_dir, "normalization_parameters.json")
    existing = [p for p in (out_csv, out_json) if os.path.exists(p)]
    if existing:
        raise SystemExit(
            "refusing to overwrite existing normalisation outputs: %s\n"
            "choose a new --output-dir." % ", ".join(existing))

    with open(args.registration) as fh:
        reg = json.load(fh)

    with args.config.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    water_thresholds = thresholds_from_config(config)

    rows = []
    params_all = {}

    for name, (s1, s2) in PAIRS.items():
        p1 = SafeProduct(s1)
        p2 = SafeProduct(s2)
        grid = grid_from_band(p1, "B08", GRID_M)
        b1 = read_bands(p1, BANDS, (grid.height, grid.width))
        b2 = read_bands(p2, BANDS, (grid.height, grid.width))

        ref = [b1[b] for b in BANDS]
        mov = [b2[b] for b in BANDS]

        shift = reg[name]["initial"].get("applied_shift_px")
        coverage_shift = None
        if shift is not None:
            grid_dx = shift[0] * SOURCE_RES / GRID_M
            grid_dy = shift[1] * SOURCE_RES / GRID_M
            mov = [align_to_reference(b, grid_dx, grid_dy) for b in mov]
            coverage_shift = (-grid_dx, -grid_dy)

        validity = build_paired_validity(p1, p2, grid, shift=coverage_shift,
                                        bands=tuple(BANDS))
        paired = validity["paired"]

        w1 = ndwi(b1["B03"], b1["B08"])
        w2 = ndwi(b2["B03"], b2["B08"])
        wm = water_masks(w1, w2, ndwi_threshold=water_thresholds["ndwi_threshold"],
                         b08_1=b1["B08"], b08_2=b2["B08"],
                         nir_threshold=water_thresholds["nir_threshold"],
                         observation_valid=paired)
        valid_land = wm["land"]
        print("=" * 80)
        print("PAIR", name, "grid", grid.width, "x", grid.height,
              "paired-valid", int(paired.sum()),
              "valid land", int(valid_land.sum()),
              "stable water", int(wm["stable_water"].sum()),
              "dynamic", int(wm["dynamic_shoreline"].sum()))

        score, complete = reference_scaled_change_score(ref, mov, valid_land)
        stable = select_pifs(score, complete, fraction=STABLE_FRACTION)
        fit_mask, val_mask = spatial_split(stable)
        print("  stable pixels:", int(stable.sum()),
              "fit:", int(fit_mask.sum()), "validation:", int(val_mask.sum()))

        params_all[name] = {"n_paired_valid": int(paired.sum()),
                            "n_valid_land": int(valid_land.sum()),
                            "n_stable": int(stable.sum()),
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

            if method in ("global", "pif"):
                params_all[name][method] = {
                    "bands": {BANDS[i]: {"a": params[i][0], "b": params[i][1],
                                         "r2": params[i][2]}
                              for i in range(len(BANDS)) if i in params}
                }

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_json, "w") as fh:
        json.dump(params_all, fh, indent=1)
    print("\nwrote", out_csv)
    print("wrote", out_json)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
