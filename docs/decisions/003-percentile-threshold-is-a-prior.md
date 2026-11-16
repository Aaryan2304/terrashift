# 003 — The percentile threshold is a prior, not an operating point

**Status:** accepted
**Date:** 2026-11-14

## Context

The classical baseline thresholds the change score at a percentile of that
scene's own scores. Earlier documentation called this the one quantity that
transfers between scenes. That claim was wrong in a specific way.

## Decision

The percentile is invariant to the units of the score, and it is relative to the
scene it was computed on. It is therefore a scene-relative prior, not a
calibrated operating point, and it is documented as
`percentile_prior_not_calibrated` in every output that depends on it.

A percentile selects a fixed fraction of pixels by construction. It cannot tell
you whether those pixels are change.

## Alternatives rejected

- **Keep the transferable-threshold claim.** It is what the earlier text said,
  and it is false: two scenes with different change density produce different
  fractions of positives at the same percentile.
- **Calibrate on the validation split.** Not possible before labels exist, and
  the two local pairs have no reviewed labels at all. A calibrated threshold
  belongs after annotation, not before.

## Consequence

- The candidate mask is tagged `exploratory_candidate_mask_not_a_detection`.
- The model, by contrast, selects on average precision, which is
  threshold-free, and reports F1 at a fixed 0.5 without claiming 0.5 is optimal.
- Two thresholds are in use and both are named explicitly: 99.0 for the baseline
  candidate mask, 99.9 for the analyst review layer.

## Evidence

`config.yaml` (`evaluation.selection_metric`, `preprocess.thresholds`),
`scripts/run_classical_baseline.py`, `docs/reference_analysis.md`,
`docs/report.md` section 5.1.
