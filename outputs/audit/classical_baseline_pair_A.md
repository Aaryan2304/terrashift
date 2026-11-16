# Classical baseline audit — Pair A

Initial audit: 2026-11-13. Latest percentile run: 2026-11-14.

## Scope

This is an exploratory baseline audit. It is not a validated change map. Pair A is the T43QBB Sentinel-2A pair from 2022-12-23 and 2023-05-12.

The script uses four 10 m bands, resampled to a 20 m grid. It applies the measured date-2 shift, masks water and invalid observations, fits histogram matching on the lowest-change half of valid land, scores spectral differences, applies morphology and lists connected components.

## Current score definition

For each band, the absolute date difference is divided by the standard deviation of that band on valid reference-date land. The four scaled differences are combined with an Euclidean norm. This avoids standardising the two dates independently.

The earlier absolute score threshold of 2.0 was exploratory and is no longer used. It was not a significance threshold and was not calibrated against labels. Current runs use the configured 99th-percentile prior, also uncalibrated. Do not treat the connected components as confirmed construction sites.

## Historical absolute-threshold run

Source: `outputs/baseline_run_A.txt` and `outputs/baseline/`.

- Analysis grid: 5,490 x 5,490 pixels at 20 m.
- Valid land pixels: 11,019,960 (36.6% of the grid).
- Mean absolute reflectance difference: 0.02227 before normalisation and 0.01668 after it (25.1% lower).
- Change-score percentiles at 50, 90, 95, 99, 99.5 and 99.9: 1.586, 2.801, 3.218, 4.297, 4.976 and 7.754.
- Threshold 2.0 marked 3,541,670 pixels (32.139% of valid land).
- After morphology, validity re-masking and the four-pixel area filter: 13,072 components.
- Largest component: 290,736 pixels, or 116,294,400 m2 (116.3 km2).

These outputs are too broad for a reliable site list. They do not distinguish construction from seasonal, water-edge, radiometric or residual registration effects.

## Latest percentile run — 2026-11-14

This run reads percentile 99.0 from `config.yaml`. It used score threshold
4.2968297. The threshold marked 110,200 valid pixels before morphology (1.000%
of valid land). Morphology, validity re-masking and the four-pixel component
filter left 11,554 candidate pixels in 484 components. The largest component
has 262 pixels (104,800 m2). The GeoTIFF tags record the percentile and
`percentile_prior_not_calibrated` status. These are candidates only.


## Threshold sensitivity

`outputs/audit/baseline_pair_A_sensitivity_reference_scaled.csv` reports fixed percentiles of the score. The percentile values are a sensitivity analysis, not selected operating points.

- 95th percentile: 5% of valid pixels before morphology; largest retained component 0.954 km2.
- 99th percentile: 1%; largest retained component 0.105 km2.
- 99.5th percentile: 0.5%; largest retained component 0.104 km2.
- 99.9th percentile: 0.1%; largest retained component 0.047 km2.

The result changes substantially with the threshold. The assignment needs manually verified labels or independent evidence before we can select a threshold or call these changes.

## Visual review

The RGB panels at `outputs/figures/pair_A_visual_review.png` use a shared 2–98% stretch. The vision review found broad brightness and colour differences between dates. It did not show a clear compact set of highlighted areas at full-tile scale. The view cannot establish the cause or validate individual components.

## Known implementation corrections

1. Corrected the output-grid scale. The 20 m grid is 5,490 x 5,490, not an upsampled grid.
2. Added a guard against an oversized output grid.
3. Added an all-bands finite-pixel mask.
4. Reapply validity after morphological closing so morphology cannot restore water or invalid pixels.
5. Replaced one full-raster scan per component with raster-wide statistics. This avoids work proportional to component count times raster size.
6. Save the score raster and tag both GeoTIFFs with the score method and exploratory threshold status.

## Files

- Code: `scripts/run_classical_baseline.py`
- Sensitivity code: `scripts/analyze_baseline_sensitivity.py`
- Mask: `outputs/baseline/candidate_mask_A.tif`
- Score: `outputs/baseline/change_score_A.tif`
- Candidate components: `outputs/baseline/candidate_components_A.csv`
- Sensitivity table: `outputs/audit/baseline_pair_A_sensitivity_reference_scaled.csv`

Pair B has completed the same exploratory run. Its results are in `outputs/audit/classical_baseline_pair_B.md`. Neither pair has a confirmed candidate site.
