# Classical baseline audit — Pair B

Initial audit: 2026-11-13. Latest percentile run: 2026-11-14.

## Scope

This is an exploratory baseline audit. It is not a validated change map. Pair B is the T43RGM Sentinel-2A pair from 2023-10-06 and 2026-10-22.

The script uses four 10 m bands, resampled to a 20 m grid. It applies the measured date-2 shift, masks water and invalid observations, fits histogram matching on the lowest-change half of valid land, scores spectral differences, applies morphology and lists connected components.

## Current score definition

For each band, the absolute date difference is divided by the standard deviation of that band on valid reference-date land. The four scaled differences are combined with an Euclidean norm. This avoids standardising the two dates independently.

The earlier absolute score threshold of 2.0 was exploratory and is no longer used. It was not a significance threshold and was not calibrated against labels. Current runs use the configured 99th-percentile prior, also uncalibrated. Do not treat the connected components as confirmed construction sites.

## Historical absolute-threshold run

Source: `outputs/baseline_run_B.txt` and `outputs/baseline/`.

- Analysis grid: 5,490 x 5,490 pixels at 20 m.
- Valid land pixels: 29,713,987 (98.6% of the grid).
- Mean absolute reflectance difference: 0.02475 before normalisation and 0.02169 after it (12.4% lower).
- Change-score percentiles at 50, 90, 95, 99, 99.5 and 99.9: 1.534, 3.502, 4.128, 5.388, 5.943 and 7.690.
- Threshold 2.0 marked 10,765,652 pixels (36.231% of valid land).
- After morphology, validity re-masking and the four-pixel area filter: 36,377 components.
- Largest component: 1,917,446 pixels, or 766,978,400 m2 (767.0 km2).

These outputs are too broad for a reliable site list. They do not distinguish construction from seasonal, water-edge, radiometric or residual registration effects.

## Latest percentile run — 2026-11-14

This run reads percentile 99.0 from `config.yaml`. It used score threshold
5.3880568. The threshold marked 297,140 valid pixels before morphology (1.000%
of valid land). Morphology, validity re-masking and the four-pixel component
filter left 67,145 candidate pixels in 2,133 components. The largest component
has 717 pixels (286,800 m2). The GeoTIFF tags record the percentile and
`percentile_prior_not_calibrated` status. These are candidates only.


## Threshold sensitivity

`outputs/audit/baseline_pair_B_sensitivity_reference_scaled.csv` reports fixed score percentiles. These are sensitivity checks, not selected operating points.

- 95th percentile: 5% of valid pixels before morphology; largest retained component 2.549 km2.
- 99th percentile: 1%; largest retained component 0.287 km2.
- 99.5th percentile: 0.5%; largest retained component 0.218 km2.
- 99.9th percentile: 0.1%; largest retained component 0.087 km2.

The result changes substantially with the threshold. The assignment needs manually verified labels or independent evidence before we can select a threshold or call these changes.

## Visual review

The RGB panels at `outputs/figures/pair_B_visual_review.png` use a shared 2–98% stretch. The vision review found widespread tonal and texture differences, not a clear set of compact highlighted areas. The view cannot establish the cause or validate individual components.

## Files

- Code: `scripts/run_classical_baseline.py`
- Sensitivity code: `scripts/analyze_baseline_sensitivity.py`
- Mask: `outputs/baseline/candidate_mask_B.tif`
- Score: `outputs/baseline/change_score_B.tif`
- Candidate components: `outputs/baseline/candidate_components_B.csv`
- Sensitivity table: `outputs/audit/baseline_pair_B_sensitivity_reference_scaled.csv`

No candidate site has been confirmed.
