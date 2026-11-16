# 005 — Original OSCD is the benchmark; the official test split is opened once

**Status:** accepted
**Date:** 2026-11-14

## Context

Two datasets were available: the 24-pair original OSCD, and a derived
OSCD100-style bundle. A benchmark and a smoke test need different properties, and
the derived bundle had been described as both.

The derived bundle has identical acquisition dates in both folders
(2015-08-01 and 2017-08-01 are both labelled as such), no CRS, an identity
transform, and a random crop split of one acquisition pair. It cannot support a
geographic split, and it cannot support a claim about change detection.

## Decision

- **Original OSCD** is the benchmark: 24 pairs, 13 bands, 14 train / 10 test
  regions from the official split.
- **The derived bundle** is demoted to a pipeline smoke test and labelled as
  such everywhere.
- **The 10 official test regions are locked.** They are used exactly once, for
  the checkpoint selected by validation AP, and only after all tuning is
  finished. Training scripts build only train and validation datasets, and the
  index writer refuses to include test rows without an explicit flag.
- **`mumbai` is excluded** from local training and validation, because it shares
  tile T43QBB with Pair A and would leak the local scene into the benchmark.

## Alternatives rejected

- **Tune on test and report the best.** It would produce a better number and a
  meaningless one.
- **Use the derived bundle for the benchmark.** Its dates, split and geometry do
  not support it, and its "test" set is a random crop of the same acquisition.

## Consequence

The reported test figures (AP 0.5068, AUC 0.8942, object F1 0.0559) are single
measurements on a held-out set, and the training code prints
`official test labels were not read` on every run. Quantisation was evaluated on
validation only, for the same reason.

## Evidence

`data/oscd_original.py`, `configs/oscd_region_splits.yaml`,
`docs/oscd_data.md`, `docs/report.md` sections 4.5 and 6.2.
