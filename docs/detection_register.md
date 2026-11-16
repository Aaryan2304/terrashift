# Detection register

The register is the deliverable list of places that were inspected and the
decision recorded for each. It is **not** a list of automatic detections. An
entry exists only after a human has reviewed the before and after imagery, or
after it is explicitly marked as an unreviewed candidate.

## Files

- Schema: `configs/detection_register_schema.yaml`
- Blank CSV template: `labels/detection_register.template.csv`
- Intended outputs: `outputs/register/detections.csv` and
  `outputs/register/detections.geojson`
- Status today: **neither output exists**. No polygons have been reviewed yet,
  so the register is empty by design. An empty register is not evidence that
  no change occurred.

## Lifecycle of one entry

    candidate  an unreviewed location proposed by the classifier, the
               classical baseline, or a systematic sweep of the shoreline.
               No class is assigned and it must not be counted as a detection.
    reviewed   a human looked at both dates and set change_type and
               confidence, with the evidence written down.
    confirmed  reviewed, and the reviewer is willing to defend the call.
               Requires reviewer, reviewed_at and evidence.
    rejected   reviewed and found not to be structural man-made change.
               Kept, because a rejected entry is evidence of the review.

A row may never move from `candidate` to `confirmed` without a reviewer, a
review date and a written evidence note. The schema enforces the fields; the
reviewer enforces the judgement.

## Fields

| Field | Meaning |
|---|---|
| `register_id` | Stable identifier, `PAIR-####`. |
| `pair` | `A` or `B`. Not a site identifier; see the caveat below. |
| `tile`, `date1`, `date2` | Sentinel-2 tile and the two acquisition dates. |
| `geometry_wkt` | Polygon in the pair's UTM CRS, EPSG:32643. |
| `crs` | CRS of `geometry_wkt`. |
| `longitude`, `latitude` | Centroid in WGS84, for the GeoJSON export. |
| `area_m2`, `area_ha` | Polygon area on the 10 m grid. |
| `change_type` | `construction`, `demolition`, `reclamation`, `other_structural`, `not_change`, or `unknown`. |
| `confidence` | `high`, `medium`, `low`. |
| `status` | `candidate`, `reviewed`, `confirmed`, `rejected`. |
| `detection_source` | `model`, `classical_baseline`, `shoreline_sweep`, or `manual_only`. |
| `score`, `threshold` | Model probability or classical score and the operating point, when applicable. Blank for `manual_only`. |
| `evidence` | Short human note stating what changed and where it is visible. |
| `before_image`, `after_image` | Paths to the panels the reviewer used. |
| `reviewer`, `reviewed_at` | Who decided, and when, in ISO-8601. |
| `notes` | Anything that qualifies the entry. |

## Rules

1. A register row is never generated from an unreviewed candidate mask alone.
2. `change_type` is never inferred from OSM, from the model score, or from a
   candidate label. OSM may appear as context in `notes` only.
3. A water-to-land transition is `reclamation` only when the before and after
   images show fill, dredging or new structure. Otherwise the row is
   `not_change` or `unknown`.
4. Rejected entries stay in the register. Deleting them destroys the record of
   how many candidates failed review.
5. Every confirmed row must carry a reviewer, a review date and an evidence
   note. The schema marks those three as required when `status` is `confirmed`.

## What the register must not claim

- It must not claim that the list is complete. Recall is unmeasured.
- It must not claim an exact event date. Two acquisitions bracket the change;
  they do not date it.
- It must not merge Pair A and Pair B into one time series. They are different
  tiles with different acquisition geometries.

## GeoJSON export

The GeoJSON is a projection of the same rows, one feature per register entry,
in WGS84 longitude/latitude with no legacy `crs` member (RFC 7946). Properties
mirror the CSV columns except `geometry_wkt` and `crs`, which the geometry
carries. The exporter must read the register, not the candidate layers.
