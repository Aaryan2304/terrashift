# Manual review queue — Pair A and Pair B

**Status: review aid only. Nothing here is a label, a detection or ground truth.**
Every row is `unreviewed` with the human fields left empty. The rasterizer still
has the final say on every polygon you draw.

Generated from `outputs/labels/v2_observation_valid/review_candidates_{A,B}_p99_9.geojson`
and `outputs/partitions_v2_observation_valid/pair_{A,B}_partition_20m.tif`.

## Files

| File | Contents |
|---|---|
| `review_queue.csv` | 249 rows, one per candidate, 30 columns |
| `review_queue_summary.json` | counts per pair and partition, flagged candidates |
| `REVIEW_GUIDE.md` | this file |

Regenerate with:

    python scripts/build_review_queue.py --output-dir review/manual_review_v1

The script refuses to overwrite an existing queue. Use a new directory to
rebuild.

## Loading it in QGIS

1. Add the candidate layer for the pair, for example
   `outputs/labels/v2_observation_valid/review_candidates_A_p99_9.geojson`.
2. Add the before and after imagery for that pair. Either the `.SAFE` bands
   directly, or your `A_before_RGB.vrt` and `A_after_RGB.vrt` built from B04/B03/B02.
3. Open the candidates' attribute table and use **Join attributes by field value**
   (or the field calculator's `attribute()` lookup) on `candidate_id` to pull
   `partition`, `review_order`, `crosses_partition_boundary` and
   `touches_guard_band` onto the layer.
4. Filter to one partition at a time, for example `partition = 'train'`, and work
   through `review_order` from 1 upward.

## What the queue contains

| Column group | Columns |
|---|---|
| Identity | `candidate_id`, `pair`, `component_id` |
| Partition | `partition`, `n_partitions`, `partition_codes`, `crosses_partition_boundary`, `touches_guard_band`, `guard_fraction` |
| Extent | `bbox_width_px_20m`, `bbox_height_px_20m`, `bbox_width_m`, `bbox_height_m`, `bbox_area_m2` |
| Size | `component_area_px_20m`, `component_area_m2` |
| Score | `mean_score`, `max_score`, `score_threshold`, `priority_percentile` |
| Location | `centroid_lon`, `centroid_lat` (EPSG:4326), `centroid_easting`, `centroid_northing` (EPSG:32643) |
| Review | `review_order`, `review_status`, `human_decision`, `evidence`, `confidence`, `note` |

`partition` is decided from every pixel under the polygon, not from its centroid:

- `train`, `validation`, `test`: all pixels fall in one split;
- `guard`: the polygon lies wholly inside a 500 m guard band;
- `mixed`: the polygon covers pixels from more than one split.

## Flagged candidates — read these first

| Candidate | Partitions under the polygon | Issue |
|---|---|---|
| `A_58` | `0|1` guard + train, guard fraction 0.78 | Touches the guard band. Cannot be labelled as drawn. |
| `A_51` | `0` guard only | Lies entirely in the guard band. Cannot be labelled at all. |
| `A_46` | `0` guard only | Lies entirely in the guard band. Cannot be labelled at all. |

The rasterizer rejects any polygon that crosses a split boundary or a buffer
(the guard band), so these three need a decision before they can enter the
label set: either redraw a smaller polygon wholly inside one split, or skip
them. **No candidate crosses a split boundary** — several splits under one
polygon does not occur in this queue.

## Required fields for every label

From `configs/manual_label_schema.yaml`, version `manual-labels-v1`. All eight
are mandatory and must be non-empty:

| Property | Allowed values |
|---|---|
| `record_id` | unique non-empty string |
| `pair` | `A` or `B`, and must match the file it lives in |
| `label_class` | `change`, `no_change`, `uncertain` |
| `reviewer` | your name or identifier |
| `reviewed_at` | ISO-8601 date or datetime |
| `evidence_source` | `before_after_visual`, `field_verified`, `other_manual` |
| `evidence` | short note citing what you saw in the before/after pair |
| `confidence` | `high`, `medium`, `low` |

Coordinates must be EPSG:4326 longitude/latitude. File names follow
`labels/manual_labels_A.geojson` and `labels/manual_labels_B.geojson`, one file
per pair, copied from `labels/manual_labels.template.geojson`.

The three blank queue columns map onto the schema as follows: `human_decision`
becomes `label_class`, `evidence` becomes `evidence`, `confidence` becomes
`confidence`. The queue does not carry `record_id`, `reviewer`, `reviewed_at` or
`evidence_source`; add those when you write the GeoJSON.

## Do negative examples need explicit polygons?

**Yes.** Pixels outside a reviewed polygon are *unlabeled*, not `no_change`. A
candidate you look at and decide is not change contributes nothing to training
unless you draw the polygon and label it `no_change`. The candidate layer only
proposes high-score areas, so without explicit negatives the training set
contains no negatives except whatever falls inside a `change` polygon's
neighbourhood — which is none.

Practical consequence: budget roughly the same effort for negatives as for
positives. Every candidate you reject is a free training negative, but only if
you draw it.

## Rasterizer rules for the three classes

`scripts/rasterize_manual_labels.py` writes one raster per pair:

| Label | Raster value | Use in training |
|---|---|---|
| `no_change` | 0 | negative target |
| `change` | 1 | positive target |
| `uncertain` | 255 | excluded from loss and metrics, same as unreviewed |
| outside any polygon | 255 | unlabeled, excluded |

The rasterizer refuses, per polygon, on any of:

- invalid geometry;
- a missing required property, or a missing evidence note;
- an `evidence_source` of OSM, candidate score or model prediction;
- an overlap with another reviewed polygon;
- geometry outside the tile;
- geometry crossing a split boundary or a guard band;
- geometry covering pixels that the observation-validity raster marks invalid —
  source NoData, `MSK_CLASSI` or `MSK_QUALIT` flags, or a registration gap.

It does **not** reject a polygon for touching water. Water is a ground cover, not
an invalid observation, and a water-to-land transition can be real reclamation.

Use `uncertain` when the evidence is genuinely inconclusive. It is not a
throwaway class: it costs you the polygon but keeps a record of what was
examined, and it never corrupts the training targets.

## A minimal annotation strategy

The queue is unbalanced. Candidates per partition:

| Pair | train | validation | test | guard |
|---|---|---|---|---|
| A | 24 | 11 | 33 | 2 |
| B | 157 | 8 | 14 | 0 |

Pair A has 24 training candidates and 33 in the test partition; Pair B has 157
in training and only 8 in validation. Annotating everything is not the cheapest
path to a trainable, defensible model.

Suggested order, smallest useful first:

1. **Calibration pass — 10 candidates.** Review 5 from A/train and 5 from B/train,
   in `review_order` sequence, and write full evidence notes. This is not for
   training; it is to check that the evidence vocabulary fits what the imagery
   actually shows, and to measure how long one polygon takes.
2. **Training set — 20 to 30 per pair.** Work down `review_order` within
   `train`, stopping when you have 20-30 reviewed polygons per pair, roughly half
   `change` and half `no_change`. Include the guard-adjacent `A_58` only after
   redrawing it wholly inside the train stripe.
3. **Validation set — 8 to 12 per pair.** Take from `validation`, again in
   `review_order`. Pair B has only 8 candidates there, so annotate all of them
   and add explicit `no_change` polygons nearby to make up the count.
4. **Test set — hold out.** Do not review the `test` partition until the model
   and its threshold are frozen. When you do, take every candidate in
   `test` for Pair A (33) and Pair B (14), plus explicit negatives, in one
   uninterrupted pass, and never revisit them after a model has seen the result.

Why this shape works: the partitions are contiguous geographic stripes with
500 m guard bands, so train, validation and test share no ground and no
near-boundary leakage. Reviewing in a fixed order makes a partial pass
reproducible, and holding the test stripe until the end keeps it a real held-out
set rather than a second validation split.

Keep `review_status` current in the CSV as you go (`reviewed`, `skipped_guard`,
`redrawn`) so a resumed session knows where it stopped. That column is for your
bookkeeping only; it never becomes a model input.

## What this is not

- Not ground truth. The candidates come from an uncalibrated 99.9th-percentile
  score threshold on a classical baseline, and they are biased toward large,
  high-contrast features.
- Not exhaustive. Anything below the threshold is absent, and the baseline cannot
  propose changes over water at all.
- Not a decision. The `human_decision`, `evidence` and `confidence` columns are
  empty by construction, and no script may fill them.
