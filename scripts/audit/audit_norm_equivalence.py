"""Prove or reject the affine-equivalence claim on real raw DN pixels.

Claim: for ``R = (DN - 1000) / 10000``, normalising with ``mean_R = (mean_DN - 1000) / 10000`` and
``std_R = std_DN / 10000`` gives exactly the same z-score as normalising DN with ``mean_DN`` and
``std_DN``.

It reads raw DN straight from the JP2, so the local reflectance path does not mediate the comparison,
and excludes DN 0 to match the OSCD ``image > 0`` rule. What the current local implementation produces
is reported alongside.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.manifest import load_manifest, pair_config
from terrashift.data.safeloader import SafeProduct

QUANTIFICATION = 10000.0
OFFSET = 1000.0
WINDOW = 256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=os.path.join(ROOT, "configs",
                                                           "data_manifest.yaml"))
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    import torch
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    bands = list(state["bands"])
    mean_dn = np.array(state["mean"], dtype=np.float64)
    std_dn = np.array(state["std"], dtype=np.float64)
    mean_r = (mean_dn - OFFSET) / QUANTIFICATION
    std_r = std_dn / QUANTIFICATION

    manifest = load_manifest(args.manifest)
    report = {"formulas": {
        "reflectance": "R = (DN - 1000) / 10000",
        "equivalent_mean": "mean_R = (mean_DN - 1000) / 10000",
        "equivalent_std": "std_R = std_DN / 10000",
        "z_source": "z = (DN - mean_DN) / std_DN",
        "z_reflectance": "z = (R - mean_R) / std_R",
    }, "checkpoint_DN": {"mean": mean_dn.tolist(), "std": std_dn.tolist()},
        "equivalent_reflectance": {"mean": mean_r.tolist(), "std": std_r.tolist()},
        "pairs": {}}

    for pair in ("A", "B"):
        cfg = pair_config(manifest, pair)
        product = SafeProduct(cfg["path_date1"])
        per_band = {}
        for index, band in enumerate(bands):
            path = product.band_path(band)
            with rasterio.open(path) as handle:
                row = handle.height // 2
                col = handle.width // 2
                raw = handle.read(1, window=((row, row + WINDOW), (col, col + WINDOW)))
            # OSCD statistics exclude DN <= 0, so the comparison uses the same rule.
            values = raw[raw > 0].astype(np.float64)
            if not values.size:
                per_band[band] = {"note": "no positive DN in the sampled window"}
                continue
            z_source = (values - mean_dn[index]) / std_dn[index]
            reflectance = (values - OFFSET) / QUANTIFICATION
            z_reflectance = (reflectance - mean_r[index]) / std_r[index]
            difference = np.abs(z_source - z_reflectance)
            # What the pipeline does instead: statistics from these same local pixels.
            local_mean = reflectance.mean()
            local_std = reflectance.std()
            z_local = (reflectance - local_mean) / local_std
            per_band[band] = {
                "pixels": int(values.size),
                "dn_min": float(values.min()), "dn_max": float(values.max()),
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
                "identical_to_float32": bool(difference.max() < 1e-4),
                "local_stats": {"mean_R": float(local_mean), "std_R": float(local_std)},
                "z_source_vs_z_local": {
                    "max_abs_difference": float(np.abs(z_source - z_local).max()),
                    "mean_abs_difference": float(np.abs(z_source - z_local).mean()),
                    "z_source_mean": float(z_source.mean()),
                    "z_local_mean": float(z_local.mean()),
                },
            }
            print("%s %s: max |z_source - z_reflectance| = %.3e | "
                  "max |z_source - z_local| = %.2f (z_source mean %+.2f)"
                  % (pair, band, difference.max(),
                     np.abs(z_source - z_local).max(), z_source.mean()))
        report["pairs"][pair] = per_band

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
