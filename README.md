# Man-Made Change Detection

A reproducible system that detects, localises and analyses structural man-made
changes in Sentinel-2 MSI Level-1C imagery from two dates.

**Status:** complete. The technical report `docs/report_2026-11-22.pdf` is the authoritative
record of method and results, and `docs/report.md` is the Markdown source it is rendered from.
Internal planning documents are not part of this repository.

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
python -m terrashift.data.manifest
```

The data products are not included in this repository. The original OSCD and
OSCD100 downloads are also local-only.

## Usage

Day 1 audit (data loader, masks, registration, normalisation):

```bash
python -m terrashift.registration          # self-test of the ECC convention
python scripts/utilities/run_registration.py --output outputs/audit/registration_v2_observation_valid/registration.json
python scripts/utilities/run_normalization.py        # normalisation comparison
python scripts/evaluation/evaluate_normalization_positive_controls.py \
    --output outputs/audit/normalization_positive_controls_v1   # does a correction preserve change?
python scripts/audit/audit.py                    # consolidated audit
```

The audit is written to `outputs/audit/data_audit.md`.

Original OSCD training uses the authors' 14/10 region split. The local config
holds out `bordeaux` and `pisa` for validation, excludes `mumbai` because it
overlaps local Pair A, and preserves all 10 official test regions. Training and
indexing do not read test-image pixels or test-label pixels. The index builder
checks test image dimensions only. `src/terrashift/data/oscd_original.py` builds a window index.
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
review steps in the annotation section of the technical report. The label tool accepts human-edited
WGS84 GeoJSON only. It creates no labels until you provide and review polygons.
After you rasterize labels, build and materialize training/validation patches with
`python -m terrashift.data.local_patches` and
`python scripts/utilities/materialize_local_patches.py`. The training entry point is
`train.py --dataset local_manual`. The reported run initialised from a trained
original-OSCD checkpoint with `--init-checkpoint PATH`; without it, local training
starts from random weights and follows a different curve. To produce that checkpoint,
run `train.py --dataset oscd_original` first, or use `artifacts/models/oscd_best.pt`. Test patches are not materialized or indexed by default;
use `--include-test` only after model selection is complete. The trainer rejects
an index built with `--include-test`.

## Classical baseline

Run one pair at a time to limit memory use. The script refuses to overwrite
existing outputs, so give each corrected run its own directory:

```bash
python scripts/utilities/run_classical_baseline.py A --output-dir outputs/baseline_v2
python scripts/utilities/run_classical_baseline.py B --output-dir outputs/baseline_v2
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
python scripts/review/rank_review_candidates.py --pair A \
    --score-dir outputs/baseline_v2_observation_valid \
    --output outputs/audit/v2/review_candidates_A_p99_9.csv
python scripts/review/export_review_candidates.py --pair A \
    --input outputs/audit/v2/review_candidates_A_p99_9.csv \
    --output outputs/labels/review_candidates_A_p99_9.geojson \
    --source-crs EPSG:32643
python scripts/reporting/make_baseline_figure.py --pair A \
    --score-dir outputs/baseline_v2_observation_valid \
    --out outputs/figures/pair_A_visual_review.png
python scripts/audit/analyze_baseline_sensitivity.py --pair A \
    --score-dir outputs/baseline_v2_observation_valid \
    --output outputs/audit/v2/baseline_pair_A_sensitivity.csv
python scripts/reporting/make_partition_figure.py --pair A \
    --partition-dir outputs/partitions_v2_observation_valid \
    --out outputs/figures/v2/partition_A.png
```

The baseline threshold is exploratory. The ranked components are review
candidates, not validated detections. The scripts write reports under
`outputs/audit/`; the technical report records the measurements.

## Georeferenced inference

`scripts/inference/predict_safe_pair.py` runs a trained checkpoint over a `.SAFE` pair and
writes map products. It is the SAFE counterpart of `scripts/inference/predict_folder.py`,
which only accepts paired `.npy` arrays.

```bash
python scripts/inference/predict_safe_pair.py --pair A \
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

`scripts/registers/build_detection_register.py` validates a human-authored register CSV
against `configs/detection_register_schema.yaml` and writes an RFC 7946 GeoJSON
plus a normalised CSV. It never invents a row: `candidate` rows must carry
`change_type=unknown` and are excluded from the detection count.

`scripts/reporting/make_evidence_panel.py` renders one validated record as a four-view
panel: date 1, date 2, the change score and the candidate mask, all on one map
extent and one contrast stretch.

```bash
python scripts/registers/build_detection_register.py \
    --input labels/detection_register.template.csv \
    --output-csv outputs/register/detections.csv \
    --output-geojson outputs/register/detections.geojson
python scripts/reporting/make_evidence_panel.py --pair A \
    --register outputs/register/detections.csv \
    --record-id REC-0001 --output-dir outputs/figures/evidence
```

Panels from `candidate` rows show the score tail, not detected change. Use
`--status smoke_test_not_a_detection` for a pipeline check.

`scripts/registers/components_to_geojson.py --mask <tif> --pair A` converts a prediction
mask to polygons with an erosion/dilation area range and a skeleton-based
length. Area is reported as a range because a raster boundary is quantised to
whole pixels; length comes from the skeleton because a pixel count measures
area, not length.

## Reproducibility manifest

```bash
export SENTINEL_DATA_ROOT=/path/containing/the/SAFE/folders
python scripts/reporting/make_run_manifest.py --output outputs/audit/reproducibility/run_manifest.json \
    --run-dir outputs/baseline_v2_observation_valid
```

It records the git commit, package versions, each product's metadata hash and
band-file inventory, and an inventory of the output directory. It refuses to
overwrite. Reproducibility is described in the technical report.

## Project structure

```
terrashift/
├── src/terrashift/    # the installable package
│   ├── data/          # SAFE reader, manifest, grids, masks, validity, patch index
│   ├── preprocessing/ # normalisation statistics, spectral indices, water state
│   ├── registration.py # ECC alignment of the two acquisition dates
│   ├── models/        # Siamese U-Net and a ResNet-18 variant
│   ├── inference/     # sliding window, ONNX Runtime, Torch session, checkpoints
│   ├── evaluation/    # lattice accumulation, metrics, matching, locked-test protocol
│   ├── geospatial/    # geometry, rendering, component vectors
│   ├── review/        # chip rendering for human review
│   └── training/      # configuration, seeding, precision, checkpoint retention
├── scripts/           # CLI workflows: audit, evaluation, inference, registers,
│                      #   reporting, review, training, utilities, plus demo.sh
├── artifacts/         # frozen models, register, results, evidence and derived inputs
├── configs/           # data manifest, region splits, label and register schemas
├── data/              # annotation GeoPackages (datasets stay local and are ignored)
├── docs/              # report.md, the rendered PDF and figures/
├── labels/            # annotation sets, exclusion lists, register template
├── review/            # review queue and calibration sheets
├── outputs/           # generated local-only outputs (gitignored)
├── tests/             # pytest suite
├── config.yaml        # the configuration the scripts read
├── eval.py            # pixel, object and size-stratified evaluation
├── quantize.py        # ONNX export, parity, INT8 QDQ, benchmark
├── train.py           # training entry point (three dataset modes)
├── train_oscd_original.py, train_local_manual.py  # modules, invoked through train.py
├── pyproject.toml, requirements.txt, environment.yml
└── LICENSE, NOTICE    # Apache-2.0
```

## Data sources

See the technical report for dataset provenance and attribution. The AI usage record is
supplied separately with the submission.

## Documentation

Three documentation artefacts are public in this repository. The AI usage record is
supplied separately with the submission:

| file | content |
|---|---|
| `README.md` | this file: install, run, test, layout |
| `docs/report.md` | Markdown source of the technical report |
| `docs/report_2026-11-22.pdf` | technical report: method, results, limitations — the authoritative rendered version |

Other documentation — decision records, the annotation and measurement-uncertainty notes, the
reproducibility note, the detection-register procedure, internal plans and interim audits — is
maintained separately by the author. It is **not** part of this repository and is not required to
build, test or run anything described here; where the technical report cites those notes, the
citation names a document that lives outside the public tree.

Submission evidence beyond what is committed — full locked-test outputs and the wider review
packages — is likewise held separately. The subset needed to check the recorded numbers is
committed under `artifacts/`.


## Command reference

Run every command from the repository root. `SENTINEL_DATA_ROOT` must point at the
directory that holds the four `.SAFE` products; commands marked *data* fail without it.

| Operation | Command | Inputs | Outputs | Verified |
|---|---|---|---|---|
| Smoke workflow | `PYTHON=python3 bash scripts/demo.sh` | local demo window | `outputs/demo/local_window_A_v1/` | yes |
| Train (OSCD) | `python train.py --dataset oscd_original` | `configs/data_manifest.yaml`, SAFE | `outputs/checkpoints/` | yes |
| Train (manual labels) | `python train.py --dataset local_manual --init-checkpoint <oscd best.pt>` | `data/manual_annotations*.gpkg`, patch index, normalisation | `outputs/checkpoints/local_manual_v2*/` | yes |
| Checkpoint evaluation | `python eval.py --checkpoint <best.pt> --output-dir <dir> --split test` | checkpoint, split | metrics JSON under `<dir>` | yes |
| Locked-test protocol | `python scripts/evaluation/evaluate_locked_test.py --registration <json> --partition-dir <dir> --output-dir <dir>` | frozen registration, partitions, annotations | `test_evaluation_report.md`, `test_metrics.json` | yes |
| Inference | `python scripts/inference/predict_safe_pair.py --pair A --output-dir <dir>` | SAFE, checkpoint, normalisation | georeferenced GeoTIFF | yes |
| Register | `python scripts/registers/build_detection_register.py` | predictions, review records | `detections.csv`, GeoJSON | yes |
| ONNX + INT8 | `python quantize.py --checkpoint <best.pt> --output-dir <dir>` | checkpoint, calibration | `onboard_fp32.onnx`, `onboard_int8.onnx` | yes |
| Report PDF | `python scripts/reporting/build_report_pdf.py --input docs/report.md --output docs/report_2026-11-22.pdf` | `docs/report.md`, figures | `docs/report_2026-11-22.pdf` | yes |

### Entry points that look similar but are not

- `eval.py` — evaluates one checkpoint on a train/validation/test split and reports pixel,
  object and size-stratified metrics.
- `scripts/evaluation/evaluate_locked_test.py` — runs the **frozen** locked-test protocol
  against `outputs/partitions_v2_observation_valid/` and the manual annotations. It must
  not be used for threshold selection; its membership is frozen.
- `train.py --dataset local_manual` fine-tunes the general trainer on local patches from
  the manifest. `train_local_manual.py` is the module that implements the human-reviewed
  fine-tune (checkpoint selection on validated AP); it has no `__main__` block, so it is
  invoked through `train.py --dataset local_manual`, not run directly.

## Running the full pipeline, in order

Every command runs from the repository root. Set the data root once:

    export SENTINEL_DATA_ROOT=/path/to/dir/holding/the/four/SAFE/products

The stages below are sequential; each reads what the previous one wrote, and none
overwrites an existing output directory, so a rerun needs a new path.

    # 1. align the two dates of each pair (ECC on B08)
    python scripts/utilities/run_registration.py \
        --output outputs/audit/registration_v2_observation_valid/registration.json

    # 2. water state and validity audit raster
    python scripts/reporting/make_audit_outputs.py

    # 3. classical baseline: scores, validity, candidate components
    python scripts/utilities/run_classical_baseline.py \
        --output-dir outputs/baseline_v2_observation_valid

    # 4. split each pair into train / guard / validation / test regions
    python scripts/utilities/prepare_local_partitions.py \
        --output-dir outputs/partitions_v2_observation_valid

    # 5. training-window index (256 px patches, stride 128)
    python scripts/utilities/build_dataset.py --root outputs/dataset/v2 \
        --partition-dir outputs/partitions_v2_observation_valid \
        --registration outputs/audit/registration_v2_observation_valid/registration.json index

    # 6. per-pair, per-band statistics from TRAIN pixels only
    #    (must follow the index: the default --source windows counts each
    #     geographic pixel once by reading the window index)
    python scripts/utilities/build_dataset.py --root outputs/dataset/v2 \
        --partition-dir outputs/partitions_v2_observation_valid normalise

    # 7. optional: write the selected patches to disk
    python scripts/utilities/build_dataset.py --root outputs/dataset/v2 \
        --partition-dir outputs/partitions_v2_observation_valid materialise

    # 8a. stage one: train on the original OSCD regions. This produces the checkpoint
    #     that stage two starts from. Skip it if you use artifacts/models/oscd_best.pt.
    #     train_oscd_original.py is a module; train.py is the entry point.
    python train.py --dataset oscd_original --epochs 20 --output-dir outputs/checkpoints/oscd_gpu_v1

    # 8b. stage two: fine-tune on the reviewed local patches. Initialising from the OSCD
    #     checkpoint is what the reported run did; from random weights the curve differs.
    python train.py --dataset local_manual --epochs 20 --pos-weight 0.834 \
        --init-checkpoint artifacts/models/oscd_best.pt \
        --normalisation outputs/dataset/v2/normalisation.json

    # 9. validation-split analysis
    python scripts/evaluation/analyse_validation_thresholds.py \
        --output outputs/evaluation/validation_threshold_analysis

    # 10. locked-test evaluation
    python scripts/evaluation/evaluate_locked_test.py \
        --registration outputs/audit/registration_v2_observation_valid/registration.json \
        --partition-dir outputs/partitions_v2_observation_valid \
        --output-dir outputs/evaluation/final_locked_test_v1

    # 11. georeferenced inference
    python scripts/inference/predict_safe_pair.py --pair A \
        --checkpoint artifacts/models/best.pt \
        --normalisation outputs/dataset/v2/normalisation.json \
        --output-dir outputs/predictions/pair_A_v2

    # 12. registers
    python scripts/registers/build_final_detection_register.py

    # 13. ONNX export, INT8 quantisation and benchmarks
    python quantize.py --checkpoint artifacts/models/best.pt \
        --output-dir outputs/quant/local_manual_v2

    # 14. rebuild the report
    python scripts/reporting/build_report_pdf.py \
        --input docs/report.md --output docs/report_2026-11-22.pdf

    # 15. read-only demonstration
    bash scripts/demo.sh

## Reproducibility, stage by stage

Each stage below was executed at least once during verification and compared against what is
committed. Where a stage was run for one pair only, the table says so; nothing here claims
equivalent coverage across both pairs. The distinction that matters is **whether a stage needs
the original Sentinel-2 imagery**: without it several stages cannot run at all, because the
evaluator, the inference tool and the trainer open the SAFE products, not just the manifest.

| Stage | Required inputs | Verification status |
|---|---|---|
| Register validation | Repository and the committed register artefacts | Reproduced, no SAFE data needed |
| Test suite | Repository and dependencies; project data only for the data-dependent tests | 635 passed, 9 skipped without data; 644 passed with it |
| Locked-test evaluation | The four `.SAFE` products, `SENTINEL_DATA_ROOT`, the committed checkpoint and the committed derived inputs (registration, partitions, labels, normalisation) | Reproduced with the exact recorded metrics |
| Georeferenced inference | `.SAFE` products, the registration summary named by `config.yaml: data.registration_summary`, the checkpoint and the normalisation statistics | Pair A full-scene inference executed. **Pair B full-scene inference was not run**, so the two pairs are not equally verified |
| Two-stage training | Stage one: the extracted OSCD region data. Stage two: `.SAFE` products, the reviewed annotations, the local patch index, the partitions, the normalisation statistics and the stage-one checkpoint | Reproduced in method and metric range, **not bit-identically** |
| ONNX FP32 and INT8 quantisation | Checkpoint, `.SAFE` products, the local patch index, the label rasters and partitions, the normalisation statistics, and the ONNX dependencies. Calibration reads the **train** split; evaluation runs on **validation**; the test split is never touched | Reproduced with byte-identical FP32 and INT8 outputs |

Stages 5 (index), 6 (statistics), 1 (registration), 4 (partitions), 10 (locked test),
12 (registers), 14 (report) and 15 (demo) were executed; partitions, index and statistics
reproduced byte-identically.

**Training is method-reproducible, not bit-identical.** Two runs of the same seed with the
same initialisation follow the same curve but do not land on the same epoch: the recorded run
reached validation AP 0.770226 at epoch 7, the verification run 0.752562 at epoch 2, on a
curve that swings between roughly 0.54 and 0.77. cuDNN selects different convolution
algorithms between runs, so identical checkpoints are not guaranteed. Deterministic stages —
evaluation, ONNX export, quantisation — do reproduce exactly.

**Pair B is substantially weaker than Pair A** on the reviewed locked-test pixels (AP
0.366373 against 0.969537). That is a reported limitation, not a defect that was fixed; the
cause is analysed in the report.

## Limitations

These are the boundaries of what this repository establishes. They are stated in the
technical report and repeated here so no reader has to infer them.

- **TerraShift is a candidate-generation and human-review system, not a fully automated
  validated detector.** The model proposes regions; a person accepts, rejects or marks them
  uncertain, and the register records that decision with its evidence.
- **Pair B performs substantially worse than Pair A.** On the reviewed locked-test pixels,
  AP is 0.366373 for Pair B against 0.969537 for Pair A. The cause is analysed in the report:
  Pair B's reviewed sample is not representative across the score range, and the pair spans
  3.0 years rather than 4.7 months.
- **Thresholds are fixed diagnostic comparison points, not validated operating thresholds.**
  0.05, 0.10 and 0.50 exist to compare models on a common scale. None was selected from the
  locked test, and none is a calibrated operating point.
- **Full-scene score rasters are not confirmed detections.** They are continuous
  probabilities, and at low thresholds the candidate fraction is far larger than any
  defensible detection rate.
- **The reviewed test pixels are not representative of the entire scene.** Metrics are
  computed on pixels a human reviewed, so they describe that sample, not scene-wide accuracy.
- **Training is method-reproducible, but GPU runs are not guaranteed to produce identical
  checkpoints.** Deterministic stages — evaluation, ONNX export, quantisation — do reproduce
  exactly.

## Reproduce the locked-test evaluation from a clone

Two different sets of inputs are involved, and they are not interchangeable.

**Shipped, small (2.1 MB), under `artifacts/derived/`** — four derived evaluation inputs, so
nothing has to be regenerated:

    registration/registration.json      the ECC shift per pair
    partitions_v2_observation_valid/    the train / guard / validation / test split rasters
    labels/                             the reviewed label rasters per pair
    dataset_v2/normalisation.json       the per-pair, per-band statistics

**Not shipped, required at run time** — the four original Sentinel-2 `.SAFE` products. The
derived inputs replace the need to recompute partitions, labels or statistics. They do **not**
replace the imagery: the evaluator opens the `.SAFE` bands for every window it scores.

With both present, the command runs from a fresh clone:

    python scripts/evaluation/evaluate_locked_test.py \
        --checkpoint artifacts/models/best.pt \
        --normalisation artifacts/derived/dataset_v2/normalisation.json \
        --registration artifacts/derived/registration/registration.json \
        --labels-dir artifacts/derived/labels \
        --partition-dir artifacts/derived/partitions_v2_observation_valid \
        --output-dir <new-directory> --batch-size 8 --models torch_fp32

Expect `pooled AP 0.653204  AUC 0.747176 | macro AP 0.667955 | A 0.969537 B 0.366373`.

**This command reads the Sentinel-2 imagery.** The execution path is
`evaluate_locked_test.py` → `quantize.pair_image_loader` → `load_manifest` → `SafeProduct` →
`read_pair_window`: the evaluator opens the `.SAFE` bands for every window it scores. The
committed derived inputs replace the need to regenerate partitions, labels or statistics, but
they do not replace the imagery.

Set the data root before running it:

```bash
# Linux / WSL
export SENTINEL_DATA_ROOT=/path/to/dir/holding/the/four/SAFE/products

# Windows PowerShell
$env:SENTINEL_DATA_ROOT = "D:\path\to\the\SAFE\parent"

# Windows cmd
set SENTINEL_DATA_ROOT=D:\path\to\the\SAFE\parent
```

`SENTINEL_DATA_ROOT` must be the **directory that contains** the four `.SAFE` folders, not a
`.SAFE` folder itself. Without it the command fails while loading the manifest:

    ValueError: unexpanded environment variable in product 'QBB_2022_12_23' path;
                set SENTINEL_DATA_ROOT

Regenerating the inputs instead of using the shipped copies:

    python scripts/utilities/prepare_local_partitions.py --output-dir outputs/partitions_v2_observation_valid
    python scripts/utilities/build_dataset.py normalise --root outputs/dataset/v2
    python scripts/utilities/run_registration.py --output outputs/audit/registration_v2_observation_valid/registration.json

All three were re-run during verification. Partitions, index and statistics reproduced the
committed artefacts byte for byte, so the shipped copies and a regenerated set agree.

## Verification status

The pipeline was re-run from the repository, writing every output outside the working
tree, and the results were compared against the artefacts already in the repository.

Reproduced exactly:

| stage | comparison |
|---|---|
| partitions | partition rasters byte-identical to `outputs/partitions_v2_observation_valid` |
| training index | 343 windows, same offsets, splits and counts (differs only in the recorded path) |
| normalisation statistics | byte-identical, sha256 `8aeea033…` |
| locked test | pooled AP 0.653204, AUC 0.747176, macro 0.667955, A 0.969537, B 0.366373 |
| ONNX FP32 export | byte-identical, sha256 `6d58e309…` |
| INT8 quantisation | byte-identical, 550,071 B, sha256 `326fb3c9…` |
| validation AP (quantize path) | pooled 0.770233, A 0.980576, B 0.743143 |
| demo | `scripts/demo.sh` exits 0 |
| test suite | 635 passed, 9 skipped without data; 644 passed with it |

Measured, but with no shipped counterpart to compare against:

- full-scene inference on Pair A: 10,980 x 10,980 tile, EPSG:32643, 120,540,008 valid
  pixels, 730 s at 165 k px/s, peak RSS 2.9 GiB. The shipped full-tile predictions came
  from the OSCD checkpoint, so they are not a baseline for the local model.
- full 20-epoch fine-tune, seed 43, `pos_weight` 0.834, initialised from the OSCD checkpoint:
  342 training patches, 159 validation windows. Best epoch was 2 with validation AP 0.752562,
  against epoch 7 at 0.770226 in the recorded run. See "Reproducibility, stage by stage" above
  for why training reproduces in method and range rather than bit-identically.

Not claimed: the validation figure of 0.770233 comes from the quantisation comparison
over 159 validation windows. `scripts/evaluation/analyse_validation_thresholds.py`
computes a different quantity on a different pixel set (pooled AP 0.7017) and is not a
contradiction of it.

## Licence

TerraShift is released under the **Apache License 2.0**. The full text is in `LICENSE`.
GitHub identifies the repository licence as Apache-2.0.

`NOTICE` holds the copyright line, a statement that the work was produced for the SkyServe
assignment, the OSCD/OSCD100 attribution with its citation, and a note that the Sentinel-2
imagery is Copernicus data. It does not restate the licence terms.

The Sentinel-2 imagery is Copernicus data and is not redistributed here. OSCD and OSCD100
retain their own terms, described in the technical report.
