# 002 — Registration is estimated on valid pixels only

**Status:** accepted
**Date:** 2026-11-14

## Context

Phase correlation was estimated over every pixel of the overlapping area. Two
things distorted it: cloud and cirrus, which are not ground, and saturated or
defective pixels, which are not measurements. The Pair A shift moved when the
estimate was recomputed with a mask, which showed the first estimate was partly
tracking cloud.

## Decision

The registration estimate uses per-date validity masks, aligned to the analysis
grid before the transform is applied. The sign and transform convention are
unchanged: the stored shift is the translation that aligns date 2 onto date 1,
and it is verified on synthetic images before any real run.

## Alternatives rejected

- **Keep the unmasked estimate.** It is reproducible but biased, and the bias
  varies with cloud cover between the two dates.
- **Feature matching (SIFT/ORB).** More robust in principle, but it adds a large
  dependency, needs tuning per scene, and would have to be validated on the same
  evidence. The masked phase correlation already reports a residual below one
  pixel on both pairs.

## Consequence

Pair A shift moved from (-0.118, -0.639) px to (-0.122, -0.639) px, which is
0.004 px. Pair B was unchanged. Both are small, and the point is not the size of
the change but that the estimate no longer depends on cloud.

Every raster step now goes through `data/grids.py`, which refuses to combine
rasters whose CRS, transform or shape disagree.

## Evidence

`scripts/run_registration.py`, `outputs/audit/registration_v2_observation_valid/registration.json`,
`docs/audit_reconciliation.md`, `docs/report.md` section 2.
