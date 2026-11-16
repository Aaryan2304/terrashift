# Reproducibility bundle

The submission bundle must let another person rebuild the numbers in the
report. This document defines what goes in it, and what the repository already
records automatically.

## What the repository records at run time

| Record | Written by | Contains |
|---|---|---|
| `run_config.json` | `train.py`, `train_local_manual.py` | every resolved setting, CLI overrides, seeds, band order, patch geometry, metric basis |
| `best.pt` | both trainers | weights plus the resolved `run_config` and validation metrics |
| GeoTIFF tags | baseline, partitions, predictions, validity | method, threshold, validity definition, grid, status |
| `multi_seed_summary.csv` | both trainers | per-seed best metrics, NaN-aware mean and population standard deviation (`std_population`, ddof = 0) |
| `quantization_report.json` | `quantize.py` | ONNX sizes and hashes, parity, INT8 comparison, latency with the provider list that produced each figure |
| `registration_v2_observation_valid/registration.json` | `scripts/run_registration.py` | per-window shift, correlation, mask counts, residual statistics |
| `partition_summary.json` | `scripts/prepare_local_partitions.py` | split geometry, per-split valid-land counts, validity source |
| `run_manifest.json` | `scripts/make_run_manifest.py` | git commit, package versions, product fingerprints, output inventory |

A run without a `run_config.json` cannot be reproduced and must not be quoted.

## Producing the manifest

    export SENTINEL_DATA_ROOT=/path/containing/the/SAFE/folders
    python scripts/make_run_manifest.py \
        --output outputs/audit/reproducibility/run_manifest.json \
        --run-dir outputs/baseline_v2_observation_valid \
        --run-dir outputs/partitions_v2_observation_valid \
        --run-dir outputs/audit/registration_v2_observation_valid

The manifest refuses to overwrite an existing file. `dirty: true` means the
working tree differs from the recorded commit; a submission manifest should
record `dirty: false`, so commit first.

## What the manifest fingerprints, and what it does not

- **Hashed in full:** `config.yaml`, `configs/data_manifest.yaml`, every small
  text output (JSON, CSV, Markdown, YAML, GeoJSON) under the inventoried
  directories, and each product's `MTD_MSIL1C.xml`.
- **Recorded by count and size only:** the `.jp2` band and quality files. The
  four products are about 0.5 GB each; hashing them on every run is not worth
  the time, and the metadata hash plus the file inventory is enough to detect a
  swapped product.
- **Not recorded:** the operating system build, the GPU driver version, and the
  CUDA toolkit. Add them by hand to the final report if the submission needs
  them; the package versions alone do not fix a CUDA build.

## ONNX Runtime dependency requirement

ONNX Runtime does not load the CUDA libraries it needs. It expects them to be in
the process already. When it cannot find them it **drops the execution provider,
logs one error line, and builds a CPU-only session** — inference still runs, so a
missing GPU appears as a slow CPU result, not as a failure.

The error, seen on this machine when nothing else had loaded CUDA:

    Failed to load library .../libonnxruntime_providers_cuda.so with error:
    libcublasLt.so.12: cannot open shared object file: No such file or directory

Two ways to satisfy the requirement, in order of preference:

1. **`onnxruntime.preload_dlls()`** before the first session. Available in
   ONNX Runtime 1.24.4 with the signature
   `preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)`. It loads
   CUDA 12 and cuDNN 9 from the `nvidia-*` site packages. **PyTorch does not
   need to be imported**, which is what makes this the right method for a
   deployment script and for the quantisation tool.
2. Import `torch` before creating the session. The PyTorch wheel ships its own
   CUDA 12 libraries and loading them satisfies ONNX Runtime. This works, but it
   ties the inference path to a training dependency and is not used here.

Either way, verify the result. `session.get_providers()` must contain
`CUDAExecutionProvider`; a request is not a guarantee.

`utils/ort_session.py` implements both habits: `preload_cuda_libraries()` and
`create_session(..., require="CUDAExecutionProvider")`, which raises instead of
accepting a fallback. `quantize.py` records the provider list next to every
latency figure, and writes `{"status": "not_measured"}` instead of a timing if
the CUDA session cannot be built.

Verified on 2026-11-15 in a process that never imported torch:

| Step | Result |
|---|---|
| Session without `preload_dlls()` | `['CPUExecutionProvider']` — silent fallback |
| Session after `preload_dlls()` | `['CUDAExecutionProvider', 'CPUExecutionProvider']` |
| Real validation patch inference | completes, 202 of 228 kernels on CUDA (88.6 %) |
| `create_session(..., require=CUDA)` without preload | raises `RuntimeError` naming the provider and the fix |

This environment has the `nvidia-*` pip packages installed
(`nvidia-cublas-cu12` 12.8.4.1, `nvidia-cudnn-cu12` 9.19.0.56), which is where
`preload_dlls()` finds the libraries. A machine without them needs the CUDA 12
runtime and cuDNN 9 on the default loader path instead.

## Bundle layout for submission

    terrashift/
    ├── PLAN.md                     living plan and decision log
    ├── README.md                   exact commands
    ├── config.yaml                 the configuration the code reads
    ├── configs/                    data manifest, splits, schemas
    ├── data/ models/ preprocess/ utils/ scripts/ tests/
    ├── docs/
    │   ├── report.md               the technical report
    │   ├── oscd_data.md            dataset provenance and audit
    │   ├── manual_labeling.md      annotation procedure
    │   ├── detection_register.md   register procedure
    │   ├── onboard_architecture.md ground and onboard split
    │   ├── reproducibility.md      this file
    │   ├── decisions/              one file per decision
    │   └── reviews/                independent review material
    ├── outputs/
    │   ├── audit/                  audit reports and run manifests
    │   ├── baseline_v2_observation_valid/
    │   ├── partitions_v2_observation_valid/
    │   ├── labels/                 reviewed polygons and review layers
    │   ├── register/               detections.csv and detections.geojson
    │   └── figures/
    ├── AI_USAGE.md
    └── requirements.txt, environment.yml

`outputs/` is gitignored. The bundle includes it as an archive, not as commits.

## Reproduction order

1. `conda env create -f environment.yml` (or `pip install -r requirements.txt`).
2. `export SENTINEL_DATA_ROOT=...`, then `python -m data.manifest` to confirm the
   four products resolve.
3. `python scripts/run_registration.py --output <new dir>/registration.json`
4. `python scripts/run_classical_baseline.py A --output-dir <new dir>` and the
   same for `B`.
5. `python scripts/prepare_local_partitions.py --score-dir <new dir> --output-dir <new dir>`
6. Annotate, rasterize, build the index, train, predict.
7. `python scripts/make_run_manifest.py ...` against the final outputs.

### Exact commands for the model stage

Both trainers write a fresh `run_config.json` per seed. Every command below
refuses to overwrite an existing directory, so a repeat run needs a new name.

    export SENTINEL_DATA_ROOT=/path/containing/the/SAFE/folders
    PY=/path/to/python

    # memory test before any training
    $PY scripts/measure_training_memory.py

    # baseline, three seeds
    $PY train.py --dataset oscd_original --epochs 20 --batch-size 8 --base 16 \
        --seeds 42 --workers 0 \
        --output-dir outputs/checkpoints/oscd_gpu_v1/baseline_seed42
    $PY train.py --dataset oscd_original --epochs 20 --batch-size 8 --base 16 \
        --seeds 43,44 --workers 0 \
        --output-dir outputs/checkpoints/oscd_gpu_v1/baseline_seeds_43_44

    # one controlled trial per run: learning rate, then batch size
    $PY train.py --dataset oscd_original --epochs 20 --batch-size 8 --base 16 \
        --lr 0.0005 --seeds 42 --workers 0 \
        --output-dir outputs/checkpoints/oscd_gpu_v1/trial_lr5e4_seed42

    # locked test evaluation, once, on the validation-selected checkpoint
    $PY eval.py --checkpoint <best.pt> \
        --output-dir outputs/eval/oscd_baseline_seed42_test --split test

    # export, parity, QDQ, benchmark
    $PY quantize.py --checkpoint <best.pt> \
        --output-dir outputs/quant/oscd_baseline_seed42_v2

    # three-provider validation benchmark
    $PY scripts/benchmark_validation_providers.py \
        --checkpoint <best.pt> \
        --quant-dir outputs/quant/oscd_baseline_seed42_v2 \
        --output-dir outputs/bench/oscd_validation_providers

    # full-tile SAFE inference smoke
    $PY scripts/predict_safe_pair.py --pair A --checkpoint <best.pt> \
        --output-dir outputs/predictions/transfer_smoke_full_tile_A \
        --stride 128 --device cuda

    # report PDF, from docs/report.md, never overwriting
    $PY scripts/build_report_pdf.py

    # read-only demonstration of the results already on disk
    PYTHON=$PY bash scripts/demo.sh

Every step refuses to overwrite an existing output, so a reproduction always
lands in a fresh directory and the original stays available for comparison.

## Known gaps

- No CI workflow runs the tests.
- `scripts/demo.sh` is still a placeholder.
- The four `.SAFE` products are not redistributable with the bundle; the
  manifest records their identity so a reviewer can request them.
