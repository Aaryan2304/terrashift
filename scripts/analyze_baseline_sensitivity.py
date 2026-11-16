"""Measure how a pair's exploratory mask changes with the score threshold.

This is a sensitivity analysis, not threshold calibration. Percentile
thresholds are reported as such; they are not selected as final operating
points or ground truth.

``--score-dir`` must point at a baseline run that applied the paired
observation-valid mask, so ``valid_pixels`` is the quality-valid footprint and
not merely "pixels with a finite score".
"""

import argparse
import csv
import os
import sys

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.geometry import clean_binary_mask

PERCENTILES = (95.0, 99.0, 99.5, 99.9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["A", "B"], default="A")
    parser.add_argument("--score-dir", default=os.path.join(ROOT, "outputs", "baseline"),
                        help="directory holding change_score_<PAIR>.tif")
    parser.add_argument("--output", default=None,
                        help="CSV path; defaults to the existing audit name")
    args = parser.parse_args()
    pair = args.pair
    score_path = os.path.join(args.score_dir, "change_score_%s.tif" % pair)
    if not os.path.isfile(score_path):
        parser.error("score raster not found: %s" % score_path)
    out_path = args.output or os.path.join(
        ROOT, "outputs", "audit",
        "baseline_pair_%s_sensitivity_reference_scaled.csv" % pair)
    if os.path.exists(out_path):
        parser.error("refusing to overwrite existing sensitivity CSV: %s "
                     "(pass --output with a new path)" % out_path)
    with rasterio.open(score_path) as ds:
        score = ds.read(1, masked=True).filled(np.nan)
    valid = np.isfinite(score)
    values = score[valid]
    thresholds = np.percentile(values, PERCENTILES)
    rows = []

    for percentile, threshold in zip(PERCENTILES, thresholds):
        raw = (score > threshold) & valid
        cleaned = clean_binary_mask(raw, valid=valid, min_area_px=0)
        labels, n_labels = ndi.label(cleaned)
        sizes = np.bincount(labels.ravel())[1:]
        kept = sizes[sizes >= 4]
        rows.append({
            "score_percentile": percentile,
            "threshold_value": float(threshold),
            "valid_pixels": int(valid.sum()),
            "raw_positive_pixels": int(raw.sum()),
            "raw_positive_pct_valid": 100.0 * float(raw.sum()) / valid.sum(),
            "components_after_morphology": int(n_labels),
            "components_area_ge_4px": int(kept.size),
            "largest_component_px": int(kept.max()) if kept.size else 0,
            "largest_component_m2_at_20m": int(kept.max() * 20 * 20) if kept.size else 0,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
