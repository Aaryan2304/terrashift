# Original OSCD data and split notes

Last verified: 14 November 2026.

This note records the local audit of the original Onera Satellite Change
Detection Dataset. The audit script is `scripts/audit_oscd_original.py`.
The generated per-region record is `outputs/audit/oscd_original_data_audit.json`.

## Sources

- OSCD authors' project page: https://rcdaudt.github.io/oscd/
- Dataset README files under `data/Onera Satellite Change Detection dataset - ...`.
- TorchGeo OSCD API and implementation notes:
  https://docs.torchgeo.org/en/stable/api/datasets/oscd.html
- Rasterio windowed I/O documentation:
  https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html

The authors describe 24 registered Sentinel-2 pairs with 13 bands, 14 official
training regions and 10 official test regions. The labels focus on urban change,
such as new buildings and roads. The official region lists are in `train.txt`
and `test.txt`.

## Local audit results

- 24 scene folders match `all.txt`.
- `train.txt` and `test.txt` are disjoint. Their union equals `all.txt`.
- Each region has 13 bands in both native-date folders and both rectified-date
  folders: 624 native band files and 624 rectified band files were inspected.
- All native bands report EPSG:4326. Their pixel dimensions vary by band.
  The authors state that the original bands have 10 m, 20 m and 60 m native
  resolutions.
- The `_rect` folders place all 13 bands of both dates on one common array
  shape per region. The dataset README says these are resampled to 10 m.
  The files have no CRS and use an identity pixel transform. Use them only as
  registered image arrays. Do not treat their bounds as map coordinates.
- All 24 TIFF label masks match the corresponding image dimensions.
  The PNG visualizations agree with the TIFF classes at an intensity boundary
  of 128. The PNGs can include antialiased boundary values.
- A separate scan checked B02, B03, B04 and B08 in both rectified dates for
  all 24 regions (48 date folders). It found no zero-valued pixels in those
  four input bands. This check does not replace a full quality-mask audit.

## Label TIFF encoding discrepancy

The downloaded `Train Labels/README.txt` and `Test Labels/README.txt` say that
TIFF labels use 0 for no change and 1 for change. The downloaded TIFF arrays
instead use values 1 and 2 in all 24 regions. Their PNG visualizations support
the mapping: TIFF class 1 is dark (no change), and TIFF class 2 is light
(change). The loader maps 1 to binary 0 and 2 to binary 1. It also accepts the
documented 0/1 encoding, but it rejects all other value sets.

The audit checked this mapping for every region. It did not infer a label from
OSM or from a model prediction.

## Training and validation regions

The training config preserves all 10 official test regions and does not use
their labels during training or index construction. It divides the 14 official
training regions into:

- 11 train regions.
- 2 validation regions (`bordeaux`, `pisa`). These were selected by ascending
  SHA-256 of `oscd-val-v1:<city>` after excluding Mumbai.
- 1 excluded region (`mumbai`). Its geographic bounds fall inside the local
  Pair A T43QBB tile. Excluding it avoids training on imagery from the same
  local tile.

The split is stored in `configs/oscd_region_splits.yaml`. The loader checks the
configured test list against the official `test.txt` list and checks the
validation cities against the recorded hash rule.

The current original-OSCD index uses 128 x 128 pixel windows with stride 128.
It contains 399 train windows, 62 validation windows and 230 official test
windows. It is an index only; it does not duplicate imagery. The index builder
reads image dimensions and split files. It does not read test-label pixels.

## Local geographic partitions

`configs/local_partitions.yaml` sets 70/15/15 contiguous spatial stripes and a
500 m guard band on each side of each split boundary. The script evaluates the
four axis/direction options by valid-land coverage only. It does not use labels
or candidate scores to choose a split. Both dates of a pair share the same
partition map.

The current 20 m masks use the score raster grid. The 10 m masks are nearest-
resampled to the date-1 B02 grid. Pair A retained 95.7% of valid land outside
the guard bands; Pair B retained 98.2%. The output maps use codes 1=train,
2=validation, 3=test and 0=guard band. The actual valid-land pixel counts are
in `outputs/partitions/partition_summary.json`.

## Manual label status

The annotation format and rasterizer must preserve three states:

- `change`: verified new structural man-made change.
- `no_change`: reviewed area with no structural man-made change.
- `uncertain`: do not use as a training target.

Pixels outside reviewed polygons are unlabeled, not negative. OSM has not been
used to create labels. The project is waiting for the user's visual decisions
before it can produce local labels or local supervised patch files.
