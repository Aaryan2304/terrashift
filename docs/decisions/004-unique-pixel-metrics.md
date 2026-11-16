# 004 — Validation counts each reviewed pixel once

**Status:** accepted
**Date:** 2026-11-15

## Context

Local patches are 256 px with stride 128, so one geographic pixel appears in up
to four patches. Computing validation metrics per patch counts that pixel up to
four times. The error is not uniform: a pixel near a patch corner is counted
once or twice, a pixel at a patch centre four times. Metrics are then weighted by
an artefact of the tiling.

The same mistake was present in the training statistics, where the class fraction
and the normalisation mean were computed over patch pixels.

## Decision

- **Validation metrics** stitch the patch probabilities onto the pair grid,
  average the overlaps, and score each reviewed pixel once
  (`UniqueValidationAccumulator`).
- **Training statistics** keep the overlapping patches, because overlap acts as
  sampling weight, but compute the mean, the standard deviation and the class
  counts with exact coverage weights: a pixel's multiplicity is
  `row_count x col_count`, and its weight is the reciprocal.
- Both patch counts and unique-pixel counts are reported, so the two bases are
  visible rather than implied.

## Alternatives rejected

- **Stride equal to patch size.** It would remove the overlap and the problem,
  at the cost of dropping every pixel that falls in a gap, and it would change
  the training set. The defect was in the measurement, not in the sampling.
- **Drop the overlap only in validation.** Half the fix. The training class
  fraction feeds `pos_weight`, so a patch-weighted fraction would have produced a
  biased loss weight.

## Consequence

On a synthetic 4 x 4 patch layout over a 6 x 6 grid, 4 patches produce 64 patch
slots but 36 unique pixels, and the evaluator reports 36. Fourteen tests cover
the weighting, the stitching and the overlap averaging.

The user-visible effect on real data is not yet measured, because the local
manual labels do not exist. The fix is in place before the first local number is
produced.

## Evidence

`utils/unique_metrics.py`, `tests/test_unique_metrics.py`,
`train_local_manual.py`, `docs/report.md` section 5.3.
