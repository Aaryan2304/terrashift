# Man-Made Change Detection

A reproducible system that detects, localises and analyses structural man-made
changes in Sentinel-2 MSI Level-1C imagery from two dates.

**Status:** work in progress. See `PLAN.md` for the full plan.

## Data

Four Sentinel-2A MSI Level-1C products, supplied with the assignment. They form
two bi-temporal pairs:

| Pair | Tile | Date 1 | Date 2 | Gap |
|------|------|--------|--------|-----|
| A | T43QBB | 2022-12-23 | 2023-05-12 | 4.7 months |
| B | T43RGM | 2023-10-06 | 2026-10-22 | 3.0 years |

The supplied `.SAFE` products stay outside this repository. OSCD100 and the original OSCD data stay under `data/` and are not to be committed.

## Requirements

- Python 3.11.
- See `requirements.txt` and `environment.yml` for dependencies.

## Setup

```bash
conda env create -f environment.yml
conda activate skyserve-cd
```

For the supplied Sentinel-2 products, set the parent folder that contains all
four `.SAFE` directories. The manifest uses this variable and does not store
machine-specific paths:

```bash
export SENTINEL_DATA_ROOT="/path/to/folder/containing/the/SAFE/directories"
python -m data.manifest
```

The data products are not included in this repository. The original OSCD and
OSCD100 downloads are also local-only.

## Usage

Day 1 audit (data loader, masks, registration, normalisation):

```bash
python -m data.register                    # self-test of the ECC convention
python scripts/run_registration.py --output outputs/audit/registration_v2_observation_valid/registration.json
python scripts/run_normalization.py        # normalisation comparison
python scripts/evaluate_normalization_positive_controls.py \
    --output outputs/audit/normalization_positive_controls_v1   # does a correction preserve change?
python scripts/audit.py                    # consolidated audit
```

The audit is written to `outputs/audit/data_audit.md`.

Original OSCD training uses the authors' 14/10 region split. The local config
holds out `bordeaux` and `pisa` for validation, excludes `mumbai` because it
overlaps local Pair A, and preserves all 10 official test regions. Training and
indexing do not read test-image pixels or test-label pixels. The index builder
checks test image dimensions only. `data/oscd_original.py` builds a window index.
It does not copy the imagery.

```bash
python train.py                          # original OSCD from config.yaml
python train.py --dataset oscd100        # pipeline smoke test only
python train.py --batch-size 4 --epochs 20 --output-dir outputs/checkpoints/run2
```

Each run writes `train_log.csv`, `best.pt` and `run_config.json` under its
output directory. `run_config.json` records the exact resolved configuration,
including the region split and command-line overrides.

Important: OSCD100 is 100 random crops of ONE acquisition pair
(2015-08-01 and 2017-08-01). Its train/val/test split is not geographic, so
its validation and test scores are not independent of training. Treat an
OSCD100 result as a pipeline smoke test, not as a benchmark number. Original
OSCD smoke-run metrics are also pipeline checks, not model results.

## Manual local labels

The candidate layers are navigation aids, not ground truth. Use the schema and
review steps in `docs/manual_labeling.md`. The label tool accepts human-edited
WGS84 GeoJSON only. It creates no labels until you provide and review polygons.
After you rasterize labels, build and materialize training/validation patches with
`python -m data.local_patches` and
`python scripts/materialize_local_patches.py`. The training entry point is
`train.py --dataset local_manual`. To start from a trained original-OSCD
checkpoint, add `--init-checkpoint PATH`. Without it, local training starts
from random weights. Test patches are not materialized or indexed by default;
use `--include-test` only after model selection is complete. The trainer rejects
an index built with `--include-test`.

## Classical baseline

Run one pair at a time to limit memory use. The script refuses to overwrite
existing outputs, so give each corrected run its own directory:

```bash
python scripts/run_classical_baseline.py A --output-dir outputs/baseline_v2
python scripts/run_classical_baseline.py B --output-dir outputs/baseline_v2
```

Each run writes four files per pair: `observation_valid_<PAIR>.tif` (1 = valid
comparison pixel, 0 = NoData/invalid), `change_score_<PAIR>.tif`,
`candidate_mask_<PAIR>.tif` (0 = valid no-change, 1 = candidate, 255 = invalid)
and `candidate_components_<PAIR>.csv`.

The observation-validity raster combines source NoData, the L1C `MSK_CLASSI`
cloud/cirrus/snow mask, the `MSK_QUALIT` quality flags for every contract band,
and the footprint left uncovered by the registration warp. It is the mask that
every later step must use; the change score alone is not a validity record.

Downstream scripts read that run explicitly. None of them default to the old
`outputs/baseline` directory any more than they must, and each refuses to
overwrite an existing file:

```bash
python scripts/rank_review_candidates.py --pair A \
    --score-dir outputs/baseline_v2_observation_valid \
    --output outputs/audit/v2/review_candidates_A_p99_9.csv
python scripts/export_review_candidates.py --pair A \
    --input outputs/audit/v2/review_candidates_A_p99_9.csv \
    --output outputs/labels/v2/review_candidates_A_p99_9.geojson \
    --source-crs EPSG:32643
python scripts/make_baseline_figure.py --pair A \
    --score-dir outputs/baseline_v2_observation_valid \
    --out outputs/figures/v2/pair_A_visual_review.png
python scripts/analyze_baseline_sensitivity.py --pair A \
    --score-dir outputs/baseline_v2_observation_valid \
    --output outputs/audit/v2/baseline_pair_A_sensitivity.csv
python scripts/make_partition_figure.py --pair A \
    --partition-dir outputs/partitions_v2_observation_valid \
    --out outputs/figures/v2/partition_A.png
```

The baseline threshold is exploratory. The ranked components are review
candidates, not validated detections. The scripts write reports under
`outputs/audit/`; see `PLAN.md` for recorded measurements.

## Georeferenced inference

`scripts/predict_safe_pair.py` runs a trained checkpoint over a `.SAFE` pair and
writes map products. It is the SAFE counterpart of `scripts/predict_folder.py`,
which only accepts paired `.npy` arrays.

```bash
python scripts/predict_safe_pair.py --pair A \
    --checkpoint outputs/checkpoints/local_manual_v1/seed_42/best.pt \
    --output-dir outputs/predictions/local_v1 \
    --stride 128 --threshold 0.5
```

Add `--window r0,r1,c0,c1` to predict a sub-region, and `--device cpu` to run
without a GPU. The script writes `prediction_prob_<PAIR>.tif`,
`prediction_mask_<PAIR>.tif`, `prediction_valid_<PAIR>.tif` and
`prediction_config.json`. The three rasters share the date 1 10 m grid: same
CRS, transform, width and height.

Preprocessing is the training contract, not a new one: four native 10 m bands,
date 2 aligned with the stored shift, per-band z-score from the checkpoint, no
histogram matching and no water masking. The tile is processed in row strips
with a halo so the sliding-window accumulators never hold a full-tile float32
array.

## Detection register and evidence panels

`scripts/build_detection_register.py` validates a human-authored register CSV
against `configs/detection_register_schema.yaml` and writes an RFC 7946 GeoJSON
plus a normalised CSV. It never invents a row: `candidate` rows must carry
`change_type=unknown` and are excluded from the detection count.

`scripts/make_evidence_panel.py` renders one validated record as a four-view
panel: date 1, date 2, the change score and the candidate mask, all on one map
extent and one contrast stretch.

```bash
python scripts/build_detection_register.py \
    --input labels/detection_register.csv \
    --output-csv outputs/labels/register/detection_register.csv \
    --output-geojson outputs/labels/register/detection_register.geojson
python scripts/make_evidence_panel.py --pair A \
    --register outputs/labels/register/detection_register.csv \
    --record-id REC-0001 --output-dir outputs/figures/evidence
```

Panels from `candidate` rows show the score tail, not detected change. Use
`--status smoke_test_not_a_detection` for a pipeline check.

`scripts/components_to_geojson.py --mask <tif> --pair A` converts a prediction
mask to polygons with an erosion/dilation area range and a skeleton-based
length. Area is reported as a range because a raster boundary is quantised to
whole pixels; length comes from the skeleton because a pixel count measures
area, not length.

## Reproducibility manifest

```bash
export SENTINEL_DATA_ROOT=/path/containing/the/SAFE/folders
python scripts/make_run_manifest.py --output outputs/audit/reproducibility/run_manifest.json \
    --run-dir outputs/baseline_v2_observation_valid
```

It records the git commit, package versions, each product's metadata hash and
band-file inventory, and an inventory of the output directory. It refuses to
overwrite. See `docs/reproducibility.md`.

## Project structure

```
terrashift/
├── PLAN.md            # the living plan and decision log
├── AI_USAGE.md        # AI usage record
├── README.md          # setup and exact commands
├── config.yaml        # the configuration the scripts read
├── eval.py            # pixel, object and size-stratified evaluation
├── quantize.py        # ONNX export, parity, INT8 QDQ, benchmark
├── train.py           # training entry point (three dataset modes)
├── train_oscd_original.py, train_local_manual.py
├── configs/           # data manifest, region splits, label and register schemas
├── data/              # SAFE loading, grids, validity, registration, patches
├── models/            # Siamese U-Net
├── preprocess/        # indices, normalisation, water
├── utils/             # metrics, geometry, imagery, inference, unique-pixel stats
├── scripts/           # audits, partitions, labels, figures, inference, manifests
├── labels/            # annotation template and detection-register template
├── outputs/           # generated local-only outputs (gitignored)
├── docs/              # report, architecture, reproducibility, audit notes
└── tests/             # unit and regression tests
```

## Data sources and software license

See `docs/oscd_data.md` for dataset provenance and attribution. The final
assignment report and AI-use record are not complete yet. This repository does
not define a software license.
