# 001 — Observation validity and water are separate concepts

**Status:** accepted
**Date:** 2026-11-14

## Context

The baseline needed a mask saying which pixels could be compared at all. The
existing code used a land mask derived from a water index, so a pixel over water
was removed from the change search. Two different things were being conflated:

- a pixel that carries no usable observation, because of NoData, cloud, cirrus,
  saturation, a defective detector or a registration gap;
- a pixel that carries a usable observation of water.

## Decision

Observation validity is built only from source NoData, `MSK_CLASSI`,
`MSK_QUALIT` and the registration coverage. Water is never invalid. A
water-to-land transition is a target class, because reclamation and port
construction are exactly the kind of man-made change the system exists to find.

Change scoring still runs on land, and the report states that water is outside
the candidate search. That is a property of the score, not of validity.

## Alternatives rejected

- **One combined mask.** Simplest to implement, but it makes a real reclamation
  change indistinguishable from a bad observation, and the rasterizer would have
  refused polygons over water.
- **Treat water as valid everywhere and score it too.** The spectral difference
  over open water is dominated by wave, sun-glint and tide, so it would have
  flooded the candidate layer. The honest middle is: valid, but not scored, and
  documented as a blind spot.

## Consequence

- The manual-label rasterizer accepts polygons over water. It rejects polygons
  that touch invalid pixels, and says so with the words "not valid observations".
- The candidate layer cannot propose reclamation changes. A reviewer can still
  draw one. Both facts are stated in the report.
- Validity is one raster per pair, consumed by the baseline, the partitions and
  the rasterizer, so all three agree by construction.

## Evidence

`data/validity.py`, `data/grids.py`, `scripts/run_classical_baseline.py`,
`tests/test_observation_validity.py`, `docs/report.md` section 5.5.
