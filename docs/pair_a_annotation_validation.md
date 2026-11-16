# Pair A manual annotation — validation report

**Source:** `data/manual_annotations.gpkg`, layer `manual_annotations`, EPSG:32643.
The GeoPackage is preserved unmodified; it was opened read-only.

**Date of review:** 2026-11-16 annotations, reviewed against the v2 observation-validity
pipeline and the 10 m partitions.

**Verdict:** 66 of 70 records validate cleanly. **Four block rasterisation** and
need a decision before Pair A can be used for training. Nothing was modified,
discarded or relabelled; the records are listed in section 2 with their reasons.

Machine-readable evidence: `outputs/labels/pair_a_qa/qa_report.json` and
`qa_report.csv`.

---

## 1. Quality assurance

| Check | Result |
|---|---|
| Total features | **70** |
| Class distribution | **change 28**, **no_change 29**, **uncertain 13** |
| Geometry validity | 70 of 70 valid `Polygon`, none empty, none null |
| Required attributes | 70 of 70 complete: `record_id`, `pair`, `label_class`, `reviewer`, `reviewed_at`, `evidence_source`, `evidence`, `confidence` |
| Duplicate record IDs | **none** |
| Polygons outside the tile | **none** |
| Polygons intersecting guard regions | **3** — `A_46`, `A_51`, `A_58` |
| Polygons crossing split boundaries | **none** (`A_58` straddles train/guard but not train/validation) |
| Invalid-observation intersections | **none** — every burned pixel is valid |
| Overlapping annotations | **1** — `A_9` overlaps `A_8` by 5 pixels |
| Rasterisation | blocked by the four records above; see section 4 for the provisional run |

Other facts: all features are pair `A`, reviewer `Aaryan`, evidence_source
`before_after_visual`, with confidence high 35 / medium 23 / low 12. Annotation
areas range from 4,173 m² to 141,038 m², median 22,198 m².

## 2. Records requiring manual correction

### 2.1 The three guard-band records

The three previously flagged candidates were re-examined. Their **annotation
polygons are not identical to the candidate rectangles** — two were redrawn — but
all three still lie inside or across a guard band.

| Record | Annotation area | Candidate area | IoU(annotation, candidate) | Burned px in guard | Verdict |
|---|---|---|---|---|---|
| `A_46` | 4,944 m² | 3,600 m² | 0.515 | 48 of 48 | entirely inside guard band |
| `A_51` | 57,521 m² | 57,200 m² | 0.678 | 574 of 574 | entirely inside guard band |
| `A_58` | 58,280 m² | 57,600 m² | 0.988 | 448 of 576 | 78 % inside guard band, 128 px in train |

The 10 m partition on Pair A is split on the x axis, with 100-pixel (1,000 m)
guard bands at both boundaries:

| Band | x range (m) |
|---|---|
| train | 199,980 – 276,340 |
| guard (train/validation) | 276,340 – 277,340 |
| validation | 277,340 – 292,800 |
| guard (validation/test) | 292,800 – 293,800 |
| test | 293,800 – 309,780 |

Against that:

- `A_46` spans x 293,239–293,300 — wholly inside the validation/test guard band.
- `A_51` spans x 293,377–293,724 — wholly inside the same guard band.
- `A_58` spans x 276,300–276,481 — starts 40 m inside train and continues 141 m
  into the train/validation guard band.

Redrawing moved `A_46` and `A_51` relative to their candidates but did not move
them out of the band. They cannot be rasterised and cannot belong to any split as
drawn. **Options, for you to decide:** drop the three records, or move them into
the nearest split (train for `A_58`; validation for `A_46` and `A_51`), which
changes which partition that ground truth belongs to and must be recorded.

### 2.2 The overlapping record

| Record | Error |
|---|---|
| `A_9` | overlaps `A_8` by 5 pixels (500 m²) |

Five pixels is presumably a digitising slip at a shared boundary. Either shrink
`A_9` to remove the overlap or merge the records, but the rasteriser rejects any
overlap.

### 2.3 What was not a problem

Every polygon passed the attribute checks, the tile-containment check and the
observation-validity check. No polygon touches an invalid observation, and none
crosses between two splits.

## 3. Validated GeoJSON export

`labels/manual_labels_A.geojson` — all 70 records, RFC 7946, EPSG:4326, with the
eight schema properties preserved.

Export path used: `geopandas.read_file` on the GeoPackage layer, `to_crs(4326)`,
then removal of the legacy `crs` member that GDAL writes. That member is valid
GDAL but is not RFC 7946, and the pipeline's own exporters drop it; the data is
unchanged by its removal.

The file is the faithful conversion of the GeoPackage. It contains the four
failing records, because removing them silently is exactly what must not happen.

## 4. Rasterisation and class distribution

The full export **cannot** be rasterised: `scripts/rasterize_manual_labels.py`
stops at the first failure, `A_9`, with
`ValueError: annotation A_9 overlaps another reviewed polygon`.

To exercise the rest of the pipeline, a **provisional** raster was built from the
66 passing records only, in a clearly separate location:

| Artefact | Path |
|---|---|
| Provisional GeoJSON (66 records) | `labels/manual_labels_A_provisional_excluding_unresolved.geojson` |
| Provisional mask | `outputs/labels/pair_a_provisional/manual_labels_A.tif` |
| Per-record audit | `outputs/labels/pair_a_provisional/manual_labels_A_audit.csv` |

**This mask must not be used for training.** It omits four reviewed records and
exists only to prove the rasterisation path and to measure class balance.

Mask: 10,980 × 10,980, EPSG:32643, 10 m, nodata 255.

| Value | Meaning | Pixels | Area |
|---|---|---|---|
| 0 | no_change | 6,277 | 0.628 km² |
| 1 | change | 11,419 | 1.142 km² |
| 255 | uncertain and unreviewed | 120,542,704 | — |

Of the 255 pixels, 3,668 come from the 13 `uncertain` records (0.367 km²) and the
rest was never reviewed. Annotated area totals 17,696 px = 1.770 km².

Class balance by split, from the audit CSV (records: change / no_change / uncertain):

| Split | change | no_change | uncertain |
|---|---|---|---|
| train | 10 | 5 | 7 |
| validation | **1** | 8 | 2 |
| test | 16 | 13 | 4 |

## 5. Dataset preparation and integration

| Check | Result |
|---|---|
| Uncertain excluded from training loss | **yes** — `data/local_patches.py` builds `valid_mask = (raw_label != 255)`, and `train_local_manual.py` records `ignored_label_value: 255` |
| Uncertain excluded from evaluation | **yes** — the same validity mask feeds the unique-pixel evaluator |
| Partitions spatially separated | **yes** — contiguous x-axis stripes with 1,000 m guard bands at both boundaries; no polygon spans two splits |
| Unit tests | **191 pass** (184 before this work, plus 7 new) |
| Pair A index build | runs with `--pairs A`; yield depends on the labelled-fraction threshold, see 6.1 |
| Inference pipeline compatibility | unchanged — `scripts/predict_safe_pair.py` consumes imagery and a checkpoint, not labels |

## 6. Engineering fixes in this pass

1. **`data/local_patches.py` — pair filter.** `build_local_patch_index` gained
   `pairs=`, plus a `select_pairs` helper and a `--pairs` CLI flag. Before this,
   indexing Pair A alone raised on the missing Pair B raster, because the builder
   requires every pair in the manifest. The default is unchanged and still strict:
   an unrequested missing raster remains a hard failure, so a production run
   cannot silently index half the dataset.
2. **`tests/test_local_patch_pair_filter.py` — new, 7 tests.** Covers the default,
   an explicit subset, caller ordering, the empty list, unknown pairs raising and
   naming the offender, and input immutability.
3. **`scripts/qa_manual_annotations.py` — new.** Per-record QA that applies the
   rasteriser's own rules through its imported `validate_feature`, but reports
   every record instead of stopping at the first failure. This is what produced
   the table in section 1.

### 6.1 A configuration value that does not fit sparse annotation

`local_min_labeled_fraction` is 0.05 in `config.yaml`. Pair A annotation covers
17,696 pixels of a 120,560,400-pixel tile — **0.0147 %**. A 256 × 256 window is
65,536 pixels, so a 5 % threshold demands 3,277 reviewed pixels per window, which
no window reaches.

Measured yield with `--pairs A`:

| `min_labeled_fraction` | windows kept | train | validation | test |
|---|---|---|---|---|
| 0.05 (configured) | **0** | 0 | 0 | 0 |
| 0.01 | 14 | 14 | 0 | 0 |
| 0.005 | 24 | 24 | 0 | 0 |
| 0.002 | 42 | 34 | 8 | 0 |

The configured value produces an empty index. I have **not** changed it: it is a
training parameter, and a lower value changes what the model sees. It needs to be
chosen deliberately once Pair B exists.

## 7. Remaining blockers before combined training

1. **Four records need correction** — `A_9` (overlap), `A_46`, `A_51`, `A_58`
   (guard band). Section 2. Nothing can be rasterised until they are resolved.
2. **The validation split has one change polygon.** Eight of the nine validation
   records are `no_change`. Any validation metric would be decided by a single
   object. More validation-split change annotations are needed.
3. **`local_min_labeled_fraction` must be re-chosen** (section 6.1); at 0.05 the
   index is empty.
4. **Pair B annotations are incomplete**, so there is no cross-pair check and no
   combined dataset.
5. **Class balance is thin everywhere:** 28 change records in total, of which 16
   are in the test split and therefore unusable for training.
