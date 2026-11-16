# Provisional Pair A label artefacts — not for training

Everything in this directory comes from the 66 Pair A annotations that pass
validation, **excluding four records that failed**:

| Record | Reason |
|---|---|
| `A_9` | overlaps `A_8` by 5 pixels |
| `A_46` | lies entirely inside the validation/test guard band |
| `A_51` | lies entirely inside the validation/test guard band |
| `A_58` | 78 % inside the train/validation guard band |

The exclusion is deliberate and explicit: `scripts/rasterize_manual_labels.py`
stops at the first failure, so a complete mask cannot be produced yet. This
directory exists to prove the rasterisation path and to measure class balance.

**Do not train on these files and do not quote their metrics.** Once the four
records are corrected, rasterise `labels/manual_labels_A.geojson` (the full,
faithful 70-record export) into a fresh directory.

Contents:

- `manual_labels_A.tif` — 10,980 × 10,980, EPSG:32643, 10 m, nodata 255
- `manual_labels_A_audit.csv` — per-record split, class, pixel count
- `local_patch_index_A.csv` and `index_A_f*.csv` — patch-index smoke runs at
  several labelled-fraction thresholds; none is a training artefact

Full analysis: `docs/pair_a_annotation_validation.md`.
