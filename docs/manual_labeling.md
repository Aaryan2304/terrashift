# Manual labels for the local Sentinel-2 pairs

The annotation tools do not make ground-truth decisions. You must make every
label call. The candidate layers only show places to inspect.

## Files

- Schema: `configs/manual_label_schema.yaml`.
- Blank RFC 7946 GeoJSON: `labels/manual_labels.template.geojson`. Coordinates
  use WGS84 longitude/latitude. Do not add a legacy top-level `crs` member.
- Pair A candidate bboxes: `outputs/labels/review_candidates_A_p99_9.geojson`.
- Pair B candidate bboxes: `outputs/labels/review_candidates_B_p99_9.geojson`.
- Rasterizer: `scripts/rasterize_manual_labels.py`.
- Patch index builder: `python -m data.local_patches`.

The candidate GeoJSON contains 71 Pair A and 181 Pair B component bounding boxes.
Each feature has `review_status=unreviewed_candidate_not_ground_truth`. It has
no label class. The rectangles are only a navigation aid. They are not
segmentation masks and must not be copied as labels without manual tracing.

## Review procedure

A prepared review queue for both pairs is in `review/manual_review_v1/`. It
lists every candidate with its partition, extent, score and centroid, flags the
candidates that touch a guard band, and leaves the decision columns empty. Start
from its `REVIEW_GUIDE.md`; regenerate the queue with
`python scripts/build_review_queue.py` into a new directory.

1. Open the before and after B02/B03/B04 JP2 files from the two `.SAFE`
   products for the selected pair in a GIS viewer. Use the product paths in
   `configs/data_manifest.yaml`; do not use names copied from an old run.
2. Add that pair's candidate GeoJSON. Inspect each candidate against both
   dates. The raster previews can have different brightness because the
   products have different acquisition conditions.
3. Draw a polygon around the area that you reviewed. Assign one class:
   - `change`: a new structural man-made feature is visible in the later image.
   - `no_change`: you reviewed the area and found no new structural man-made
     feature.
   - `uncertain`: evidence is not clear. This class is ignored during training.
4. Add the required properties from the schema. Use `before_after_visual`,
   `field_verified`, or `other_manual` as `evidence_source`. Write a short
   note that states what you saw. Do not use OSM, candidate scores, or model
   predictions as label evidence.
5. Save annotations as WGS84 longitude/latitude GeoJSON, with one file per
   pair, such as `labels/manual_labels_A.geojson` and
   `labels/manual_labels_B.geojson`. Copy the blank template; do not edit it in
   place. Every feature in a file must name the same pair as that file.
6. Rasterize one pair at a time. For example:

   `python scripts/rasterize_manual_labels.py --pair A --labels labels/manual_labels_A.geojson`

   The tool writes `outputs/labels/manual_labels_A.tif` and an audit CSV. It
   rejects invalid geometry, missing evidence, automated evidence sources,
   overlaps, polygons outside the tile, polygons crossing a split or guard
   band, and pixels that the observation-validity raster marks as invalid.

   The validity check reads `observation_valid_A.tif` from the baseline output
   directory (`--validity-dir`, default: the score directory). The rasterizer
   stops if that file is missing; it will not fall back to the change-score
   raster, because a finite score is not a valid observation. It also requires
   the validity and score rasters to share one CRS, transform, shape and
   resolution, and it checks the partition and score CRS against the pair's CRS
   in `configs/data_manifest.yaml`. The 20 m validity raster is then reprojected
   to the 10 m partition grid with nearest sampling, so a 10 m pixel inherits
   the validity of the 20 m pixel that contains it.

   The rasterizer also refuses to overwrite an existing mask or audit CSV. The
   candidate exporter and patch-index builder refuse to overwrite their output
   files. Use fresh output paths when you revise or regenerate data. Training,
   partition, and materialization commands also refuse non-empty output
   directories; use a new `--output-dir` for each run.
7. Build the local patch index after both pair rasters exist:

   `python -m data.local_patches`

   The index omits buffer-crossing patches and patches with less than the
   configured reviewed fraction. It does not create labels. Unreviewed and
   uncertain pixels keep value 255 and are excluded from the training loss.
   Test partitions and test-label pixels are excluded by default. The local
   trainer rejects an index built with `--include-test`. Use that indexer flag
   only after model selection, if you need test-window indexing.
8. Materialize train and validation data with
   `python scripts/materialize_local_patches.py`. It writes compressed NPZ
   patches with both dates, the binary mask, and a valid-pixel mask. It skips
   test patches by default. Use its `--include-test` option only after the final
   model is selected.
9. Start local training with `train.py --dataset local_manual`. To fine-tune a
   compatible original-OSCD model, pass `--init-checkpoint PATH`. Without that
   option, the model starts from random weights. Training keeps the 50 % patch
   overlap, but the band statistics, the class balance and every validation
   metric are computed over unique geographic pixels: overlapping predictions
   are stitched onto the pair grid and averaged before scoring, so one reviewed
   pixel counts once. The run config records both the patch count and the unique
   labelled-pixel count. The current run outputs are smoke checks; no full
   training result exists.

## Raster values

- 0: human-reviewed `no_change`.
- 1: human-reviewed `change`.
- 255: `uncertain` or not reviewed; excluded from loss and metrics.

## Water and reclamation

Observation validity and water are two different things.

- Observation validity answers "is there a trustworthy measurement here?".
  It covers source NoData, the L1C quality flags, and the registration gap.
- Water answers "what is on the ground?". It is not part of validity.

The rasterizer reads `observation_valid_<PAIR>.tif` only. It does not read a
water mask, and it does not reject a polygon that touches water. This is
deliberate: a water-to-land transition can be a real reclamation or coastal
construction event, and that is a target class for this assignment. Removing
water pixels would delete that evidence.

Decide the class from the images:

- stable water with no fill, dredging, or construction: `no_change`, or
  `uncertain` if you are not sure;
- water that became built or filled land, with before/after evidence: `change`;
- the shoreline itself, where the water edge moved: draw the changed strip and
  record `change` only if you can see new structure or fill.

The dynamic shoreline is a water state, not an invalid observation. The
baseline keeps it out of the "land" statistics so that a water-level change
does not look like a spectral anomaly, but the annotation tool still accepts
it as a target area.

Each polygon must lie wholly inside one train, validation, or test partition.
The partitions are spatial and pair-specific. Do not draw across the 500 m
buffer or a split boundary. The official OSCD test cities remain separate; this
manual local label workflow does not edit their labels.

## Current status

The candidate layers and annotation checks are ready. No local manual labels
have been created. The generated local patch index will remain unavailable
until you supply reviewed GeoJSON annotations. No OSM query or model output has
been used to create a label.
