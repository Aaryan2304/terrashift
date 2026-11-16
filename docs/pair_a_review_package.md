# Pair A — annotation corrections, validation coverage and patch strategy

Follows `docs/pair_a_annotation_validation.md`. Four deliverables, none of which
changes an annotation, a partition or a model.

---

## 1. Guard-band exclusions

**Manifest:** `labels/exclusions_A.json`. **Applied by:**
`scripts/apply_label_exclusions.py`. **Result:** `labels/manual_labels_A_filtered.geojson`
(70 → 67 features). **Report:** `outputs/labels/pair_a_qa/exclusions_applied.json`.

| Record | Class | Guard px / burned px | Reason |
|---|---|---|---|
| `A_46` | uncertain | 48 / 48 | wholly inside the validation/test guard band, x 292,800–293,800 m |
| `A_51` | no_change | 574 / 574 | wholly inside the same band |
| `A_58` | change | 448 / 576 (77.8 %) | straddles the train/validation guard band, x 276,340–277,340 m |

The master GeoPackage is unchanged: 70 rows, mtime 11:44, opened read-only
throughout. The three records stay in it, keep their geometry and class, and are
excluded only from exports.

The manifest is data, not code, so the same three records can be re-admitted by
editing one file. The filter refuses to write if an excluded ID is missing from
the source — a typo cannot silently remove nothing. Eleven tests cover it,
including that geometry passes through untouched
(`tests/test_label_exclusions.py`).

Note: `A_9` is **not** in the manifest. Its failure is an overlap with `A_8`,
not a guard-band problem, and it is handled in section 2.

---

## 2. `A_8` and `A_9`

**Overlay:** `outputs/labels/pair_a_qa/overlap_A8_A9/overlap_A_8_A_9.geojson` —
three features, `A_8`, `A_9` and their intersection, EPSG:4326, openable in QGIS.
**Figure:** `overlap_A_8_A_9.png`. **Summary:** `overlap_summary.json`.

| | `A_8` | `A_9` |
|---|---|---|
| Class | no_change | no_change |
| Confidence | high | high |
| Area | 15,540.7 m² (156 px) | 14,910.2 m² (157 px) |
| Evidence | "The reviewed feature remains structurally consistent between the Before and After images. No relevant man-made structural change is observed." | identical text |

**Intersection:** 630.1 m² = **6 px** at 10 m geometric, **8 px** when both
polygons are rasterised — 4.0 % of `A_8` and 4.2 % of `A_9`, IoU 0.021. They
touch along one edge; this is a digitising sliver, not a shared feature.

**Are they the same physical feature?** No, and the question is moot for the
label: both are `no_change`, so the intersection carries no class conflict. The
two outlines cover adjacent parts of the same built-up area rather than one
structure drawn twice.

**Two things the review turned up that need your attention.**

1. **The evidence text is boilerplate.** Both records carry the same sentence,
   which describes nothing about what was seen. The schema asks for a note that
   cites the before/after evidence; this does not.
2. **The imagery does not obviously support `no_change`.** A visual check of the
   before/after crops shows dark surfaces in date 1 and bright, sharp-edged
   surfaces in date 2 inside both outlines. Quantitatively, sampling the
   classical change score inside each polygon:

   | Class | n | mean score |
   |---|---|---|
   | no_change | 29 | 9.27 |
   | uncertain | 13 | 10.53 |
   | change | 28 | 13.27 |

   `A_8` sits at the 17th percentile of `no_change` (mean 7.34), which is
   consistent with its label. `A_9` sits at the 69th percentile with mean 10.00,
   and its **maximum of 28.5 is the highest maximum of any `no_change` record**
   and higher than the mean of every `change` record. `A_9` is also the
   candidate the calibration shortlist picked as the extreme of the score
   distribution.

I am **not** claiming these are mislabelled; seasonal ground cover, bare soil and
shadow differences produce exactly this signature. I am saying the evidence does
not let anyone check, and the scores are inconsistent with a high-confidence
`no_change`.

**Proposed correction, not applied:**

- Trim the 6–8 pixel intersection from `A_9` (or from `A_8`). Since both carry
  the same class, trimming is semantically neutral and would let the rasteriser
  accept the file. Record the edit in the annotation log with the reason.
- Re-review `A_9` against the imagery and replace the boilerplate evidence text
  with what was actually seen. If it is change, that also fixes the validation
  split's single-positive problem in section 3.

---

## 3. Validation-split coverage

**Layers, all EPSG:4326, ready for QGIS:**

| Layer | Contents |
|---|---|
| `validation_candidates.geojson` | 11 candidate rectangles in the validation partition, each with the class it was annotated as |
| `validation_annotations.geojson` | the 11 annotations in that partition |
| `validation_proposal_areas.geojson` | 1,366 score-tail areas **not** covered by any annotation or candidate |

Current state of the validation stripe: **11 annotations — 1 change, 8 no_change,
2 uncertain**, over 9.8 km² of stripe. The single positive is the reason
validation cannot select a model today.

**Where else could change be?** Applying the 99.0 percentile of the change score
inside the validation partition, then removing everything already annotated or
already offered as a candidate, leaves:

| Filter | Proposals | Area |
|---|---|---|
| all (≥ 4 px) | 1,366 | 4.67 km² |
| ≥ 16 px (6,400 m²) | **145** | 1.6 km² |
| ≥ 64 px (25,600 m²) | 5 | 0.2 km² |

The unfiltered set is too many to review and is dominated by 4–6 pixel specks.
The 145 proposals of 6,400 m² or more are the actionable set; the largest is
50,800 m² at 72.97781 E, 19.67169 N.

**Can additional independent positives be annotated? Yes** — at least 145
candidate areas inside the validation stripe are untouched by the current
annotation, in a range of sizes, and none of them crosses a split or touches a
guard band (the layer is clipped to the validation partition). They are
score-tail proposals from an uncalibrated percentile, so each needs visual
confirmation exactly like the original candidates.

Target for a usable validation split: **8–12 change polygons** spanning several
sizes. At the current one, a single missed or spurious object moves recall by
100 %.

---

## 4. Patch-sampling comparison

`scripts/compare_patch_sampling.py` → `outputs/labels/pair_a_qa/patch_sampling.json`.
Test-partition windows are excluded, matching the indexer's default.

Annotation density is 17,696 px in a 120,560,400-px tile — **0.0147 %**.

### Grid sampling by labelled-fraction threshold

| Patch | Threshold | Patches | train | validation | with change | change px | no_change px |
|---|---|---|---|---|---|---|---|
| 256 | **0.05 (configured)** | **0** | 0 | 0 | 0 | 0 | 0 |
| 256 | 0.02 | 4 | 4 | 0 | 4 | 8,218 | 2,322 |
| 256 | 0.01 | 14 | 14 | 0 | 14 | 17,800 | 2,322 |
| 256 | 0.005 | 24 | 24 | 0 | 24 | 22,808 | 2,322 |
| 256 | 0.002 | 42 | 34 | 8 | 30 | 24,002 | 4,771 |
| 256 | 0.001 | 57 | 40 | 17 | 32 | 24,254 | 6,074 |
| 128 | 0.05 | 13 | 13 | 0 | 13 | 13,914 | 2,322 |
| 128 | 0.02 | 27 | 27 | 0 | 27 | 21,777 | 2,322 |
| 128 | 0.01 | 45 | 36 | 9 | 34 | 23,359 | 4,786 |
| 128 | 0.005 | 77 | 52 | 25 | 44 | 24,659 | 7,281 |
| 128 | **0.002** | **83** | 56 | 27 | 48 | 24,769 | 7,471 |
| 128 | 0.001 | 87 | 58 | 29 | 50 | 24,805 | 7,528 |

### Annotation-centred and balanced

| Strategy | Patches | train | validation | with change | change px | no_change px | multi-covered | median NN |
|---|---|---|---|---|---|---|---|---|
| centred 256 | 9 | 9 | 0 | 9 | 11,237 | 1,659 | 25.2 % | 88 px |
| centred 128 | 11 | 10 | 1 | 11 | 8,850 | 603 | 13.4 % | 282 px |
| balanced 256 | 72 | 50 | 22 | 36 | 24,298 | 6,230 | 64.0 % | 128 px |
| balanced 128 | 108 | 76 | 32 | 54 | 24,832 | 7,582 | 65.4 % | 64 px |

### What the numbers mean

- **The 0.05 threshold is simply incompatible with polygon annotation.** It
  requires 3,277 reviewed pixels in a 256² window; the largest annotation is
  14,100 m² = 141 px. No threshold on this data will satisfy it.
- **128 px windows more than double the yield** at every threshold, because the
  same annotation occupies four times the window fraction. 128/0.002 gives 83
  patches (56 train, 27 validation) against 42 (34/8) at 256/0.002.
- **Annotation-centred sampling is the most efficient per patch but yields too
  few.** Only 9–11 patches, because 16 of the 28 change records are in the test
  split and several components fall inside the same window. Its coverage is
  minimal (13–25 % multi-covered), so neighbouring context is thin.
- **Balanced sampling doubles the pool but the negatives are almost empty.**
  A 256² no_change patch carries roughly 173 reviewed pixels — 0.26 % of the
  window. Everything else is 255 and excluded from the loss. The model would
  learn "negative" from a handful of pixels per patch. This is a data problem,
  not a sampling problem: **explicit no_change polygons are needed**, not a
  smarter sampler.
- **Overlap is high by construction.** With stride = size/2, 50–65 % of covered
  pixels appear in more than one patch. Training tolerates that, but the metrics
  must stay unique-pixel based, which the existing evaluator already does.
- **Partition containment holds everywhere.** Every window in every strategy is
  wholly inside one split; `window_stats` rejects any window touching a guard
  band or spanning two codes.

### Recommendation for the config (not applied)

Do not simply lower `local_min_labeled_fraction`. The defensible combination on
the current data is: **128 px patches, threshold 0.002, balanced sampling**, and
accept that this is a thin set — 76 training patches and 32 validation patches
from one pair. It becomes viable when Pair B lands and the negative polygons are
drawn. The threshold should be re-chosen against the combined dataset, not this
one.

---

## 5. Deliverables and state

| Deliverable | Path |
|---|---|
| Guard exclusion manifest | `labels/exclusions_A.json` |
| Manifest application + report | `labels/manual_labels_A_filtered.geojson`, `outputs/labels/pair_a_qa/exclusions_applied.json` |
| A_8/A_9 overlay | `outputs/labels/pair_a_qa/overlap_A8_A9/` (GeoJSON, PNG, JSON) |
| Validation review layers | `outputs/labels/pair_a_qa/validation_review/` (3 GeoJSON + summary) |
| Patch-sampling comparison | `outputs/labels/pair_a_qa/patch_sampling.json` |

New code: `scripts/apply_label_exclusions.py` (11 tests),
`scripts/analyse_label_overlap.py`, `scripts/build_validation_review.py`,
`scripts/compare_patch_sampling.py`. **202 tests pass.**

Unchanged: the master GeoPackage (70 rows, read-only), the spatial partitions,
every OSCD checkpoint, prediction and experiment record, and the training
configuration — `local_min_labeled_fraction` is still 0.05.

**Open, and waiting on you:**

1. Approve or reject the A_9 trim, and re-review `A_9`'s class and evidence.
2. Annotate 8–12 change polygons in the validation stripe from the 145 proposals.
3. Draw explicit no_change polygons; the balanced strategy shows they are nearly
   absent.
4. Pair B annotations, then re-choose the patch threshold on the combined set.
