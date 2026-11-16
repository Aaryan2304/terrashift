# Calibration shortlist — 10 candidates

**Status: selection aid only. No decision has been made and no label exists.**
The review sheet is written with every decision column empty.

Built from `review/manual_review_v1/review_queue.csv` by
`scripts/build_calibration_shortlist.py`, which refuses to overwrite.

## Files

| File | Contents |
|---|---|
| `calibration_shortlist.csv` | the 10 candidates with geometry, score and context |
| `review_sheet.csv` | the same 10 with empty decision columns |
| `calibration_summary.json` | selection parameters and the chosen set |

## The shortlist

| Nav | Candidate | Pair | Area | Bounding box | Max score | Context | Lon | Lat |
|---|---|---|---|---|---|---|---|---|
| 1 | A_1 | A | 15 px, 6,000 m² | 100 x 60 m | 9.65 | inland | 72.73430 | 19.86692 |
| 2 | A_2 | A | 117 px, 46,800 m² | 360 x 160 m | 12.63 | inland | 72.73843 | 19.86544 |
| 3 | A_9 | A | 20 px, 8,000 m² | 100 x 80 m | 28.52 | inland | 72.76229 | 19.72214 |
| 4 | A_13 | A | 9 px, 3,600 m² | 60 x 60 m | 22.10 | coastal margin | 72.75977 | 19.61563 |
| 5 | A_59 | A | 9 px, 3,600 m² | 60 x 60 m | 14.12 | coastal margin | 72.86644 | 19.09521 |
| 6 | B_115 | B | 9 px, 3,600 m² | 60 x 60 m | 8.64 | inland | 77.04769 | 28.27178 |
| 7 | B_136 | B | 133 px, 53,200 m² | 240 x 360 m | 18.12 | inland | 77.16904 | 28.21965 |
| 8 | B_148 | B | 24 px, 9,600 m² | 60 x 160 m | 15.66 | inland | 77.22408 | 28.19639 |
| 9 | B_107 | B | 12 px, 4,800 m² | 80 x 60 m | 20.20 | inland | 77.36545 | 28.33071 |
| 10 | B_100 | B | 9 px, 3,600 m² | 60 x 60 m | 8.37 | inland | 77.72426 | 28.37850 |

Selection constraints, all satisfied: every candidate is in a `train`
partition and clear of the guard band; area spans 9 to 133 px, a factor of 14.8;
max score spans 8.37 to 28.52.

## How these ten were chosen

Score was **not** used to rank or filter. Selection is by three properties:

1. **Size coverage.** Five targets per pair at the minimum, 25th, 50th, 75th and
   maximum component area. The extreme targets take the candidate closest to the
   extreme, so the smallest and the largest are always present. The middle
   targets look at the five closest candidates by area and pick for spread.
2. **Spatial spread.** At each target, take the candidate farthest from those
   already chosen, so the ten are distributed across the tile instead of
   clustering in one hot area.
3. **Context.** Land, stable-water and shoreline fractions within 1 km of the
   centroid, from `outputs/audit/water_masks_{A,B}.tif`. Two candidates sit on a
   coastal margin; eight are inland.

Navigation order is a greedy nearest-neighbour tour from the westernmost
candidate, restarted for each pair. Steps 1 to 5 walk Pair A, 6 to 10 walk
Pair B. The last leg of Pair A (A_13 to A_59, 58.7 km) is long because that
candidate is isolated; the tour visits it last deliberately.

## What to look at, and why these were chosen this way

- **A_9, max score 28.52** is far above every other candidate here (the next is
  22.10, and the layer threshold is 7.77). It is the extreme of the score
  distribution, and the most likely to be either a very obvious change or a
  very confident false positive. Either answer is useful calibration.
- **A_2, 117 px** is the largest Pair A component and is elongated, 360 x 160 m.
  Large thin structures are usually roads or quay edges, which is exactly the
  case where "structural man-made change" and "seasonal or tidal difference"
  need to be told apart.
- **A_13 and A_59** sit on a coastal margin, where the water mask shows a
  shoreline class within 1 km. Calibration must include these, because the
  classical baseline cannot propose changes over water and the annotator must
  still decide what a shoreline transition means.
- **B_136, 133 px** is the largest Pair B component. Pair B is 99.4 % land, so no
  Pair B candidate has a water context; land-cover variety in Pair B comes from
  urban versus open ground, which the water mask cannot see. That is a limit of
  the context measure, not a property of the selection.
- The four 9 px candidates are the smallest the 99.9th-percentile threshold
  produces. They test whether a 3,600 m² footprint is even annotatable at 20 m
  resolution, which is an open question for the whole labelling effort.

## Recording results in `review_sheet.csv`

Fill these three columns for each row. Leave them empty if you have not looked
yet; do not guess.

**`interpretability`** — can you tell what you are looking at?

| Value | Meaning |
|---|---|
| `interpretable` | the before/after pair shows the feature clearly enough to judge |
| `partially_interpretable` | something is visible but cloud, haze, shadow, resolution or misregistration limits the judgement |
| `unassessable` | you cannot reach a judgement from these images |

**`apparent_class`** — what do you see, before any policy decision?

| Value | Meaning |
|---|---|
| `apparent_structural_change` | a new built structure or groundworks appear in the later image |
| `apparent_non_target_change` | the difference is real but not structural man-made change: vegetation, seasonal water, tide, agriculture, construction of a non-structural kind |
| `no_visible_change` | the two dates look the same in this footprint |
| `uncertain` | the evidence does not support any of the above |

**`suitable_for_polygon`** — should this footprint become an annotated polygon?

| Value | Meaning |
|---|---|
| `yes` | the feature is clear and can be drawn as-is |
| `yes_with_adjustment` | workable, but the polygon needs redrawing to match the feature or to stay in the partition |
| `no` | not worth annotating: unassessable, or the visible change is not a target |

**`reason`** — one sentence on why you chose that combination.
**`evidence`** — what you actually saw, citing the before and after views, for
example "new light-grey rectangular roof appears in the 2023-05 view, absent in
2022-12; shadow direction differs between dates".
**`notes`** — anything that does not fit the other fields.

Record your name and the date in `notes` if you want the sheet to be traceable,
or add them when you copy a decision into the annotation GeoJSON. The sheet is
deliberately not the label file: it is a worksheet.

## Using it in QGIS

1. Add the candidate layer for the pair, and the before/after imagery.
2. For each nav step, use **Zoom to Coordinate** (EPSG:32643) with the
   `centroid_easting` / `centroid_northing` values from
   `calibration_shortlist.csv`; they are exact to 1 cm, whereas the lon/lat pair
   is rounded to 7 decimals for convenience.
3. Set the scale to about 1:5,000 to see a 60 x 60 m candidate with context.
4. Work down the nav order. Do not skip ahead to the high scores.

## What this does not do

- It does not label anything. `review_sheet.csv` ships with the three decision
  columns and the three free-text columns empty.
- It does not decide whether a candidate is real change. Only you can, from the
  imagery.
- It does not replace the annotation stage. A `yes` here means "worth drawing",
  not "this is change", and nothing from this sheet may be used as a training
  target.
