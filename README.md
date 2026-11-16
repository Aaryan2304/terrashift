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

The data is not copied into this repository. The paths are in
`configs/data_manifest.yaml`.

## Requirements

- Python 3.13 (conda env `gpu` on this machine)
- See `requirements.txt` for the pinned versions.

## Setup

```bash
conda env create -f environment.yml
conda activate skyserve-cd
```

Or use the existing env:

```bash
/home/aaryan/miniconda/envs/gpu/bin/python -c "import rasterio, torch; print('ok')"
```

## Usage

Day 1 audit (data loader, masks, registration, normalisation):

```bash
PY=/home/aaryan/miniconda/envs/gpu/bin/python
$PY -m data.register                    # self-test of the ECC convention
$PY scripts/run_registration.py         # registration + residual
$PY scripts/run_normalization.py        # normalisation comparison
$PY scripts/audit.py                    # consolidated audit
```

The audit is written to `outputs/audit/data_audit.md`.

Training, evaluation and inference commands will be added on Day 2 and Day 3.

## Project structure

```
skyserve_assignment/
├── PLAN.md            # the living plan
├── config.yaml        # all defaults
├── configs/
│   └── data_manifest.yaml
├── data/              # .SAFE loading and patch extraction
├── models/            # Siamese U-Net, onboard model
├── utils/             # metrics, geometry, visualisation
├── scripts/           # audit, patches, labels, benchmarks
├── labels/            # reference labels
├── outputs/           # figures, masks, register, checkpoints
├── docs/              # report and decision records
└── AI_USAGE.md        # AI chat record
```

## License and attribution

The Sentinel-2 data is Copernicus / ESA. See `docs/report.md` for attribution.
