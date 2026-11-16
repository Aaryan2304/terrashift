# Onboard and ground architecture

Work item 3 of the assignment asks for the architecture that would run on the
spacecraft and on the ground. This document states the split, the data flow and
the interfaces. It contains **no measured performance numbers**. Every number
that would describe speed, memory or energy is listed as pending, because no
target board has been benchmarked.

## Why a split exists

The four supplied products are Sentinel-2A MSI Level-1C, about 10,980 x 10,980
pixels per band at 10 m, 13 bands per product, roughly 0.5 GB of imagery per
date. A 4 GB laptop GPU is already the practical ceiling for training here. A
radiation-tolerant flight processor is one to two orders of magnitude weaker
than that laptop GPU. Downlinking the full pair and processing it on the ground
is therefore the only option that keeps the science honest for this data set;
onboard processing is a scope reduction that must be justified, not assumed.

The split below reflects that: the spacecraft does the work that saves
bandwidth, the ground does the work that needs accuracy.

## Ground segment (implemented, tested)

    L1C pair (.SAFE, 13 bands)                     data/safeloader.py
        |
        | DN -> TOA reflectance, DN 0 = NoData
        v
    quality and validity                           data/masks.py, data/validity.py
        |  MSK_CLASSI (cloud, cirrus, snow/ice)
        |  MSK_QUALIT (lost, degraded, defective, nodata, saturated, cross-talk)
        |  NoData, registration coverage
        v
    paired observation-valid mask                  data/validity.py
        |
        +--> registration  (masked ECC, one global shift)     data/register.py
        +--> normalisation (four methods compared)            preprocess/normalize.py
        +--> classical baseline (candidate prior)             scripts/run_classical_baseline.py
        |
        v
    four-band stack B02/B03/B04/B08 at 10 m        data/local_patches.py
        |
        v
    Siamese U-Net, 4 in / 1 out, base 16            models/siamese_unet.py
        |
        v
    sliding-window inference, overlap averaged      utils/inference.py
        |
        v
    georeferenced probability, mask, validity       scripts/predict_safe_pair.py
        |
        v
    human review -> detection register              docs/detection_register.md

## Spacecraft segment (designed, not implemented)

    raw MSI pushbroom frames
        |
        | on-board calibration, co-registration of the two acquisitions
        v
    tile-level change prior
        |
        | 4-band downsample to 20 m, 256 px chips, 4-channel pair
        | INT8 quantised Siamese U-Net (FC-Siam-diff style)
        v
    probability chip + validity chip
        |
        | threshold, connected components, area filter
        v
    candidate polygons + confidence
        |
        | downlink budget: polygons and statistics, not imagery
        v
    ground segment re-runs the same model on the full-resolution pair

The onboard stage is a **routing** stage: it decides which tiles are worth a
full downlink. It cannot confirm a change. The ground stage owns the decision,
the register and the report.

## Interface contract

| Item | Value | Source |
|---|---|---|
| Input bands | B02, B03, B04, B08 | `config.yaml: data.bands` |
| Input radiometry | TOA reflectance, `(DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE` | `data/safeloader.py` |
| Input resolution | 10 m, native | `configs/data_manifest.yaml` |
| Patch size | multiple of 8; 256 used locally, 128 used for OSCD | `run_config.json` of each run |
| Model input | `(N, 4, H, W)` per date, z-scored with training statistics | `train_local_manual.py` |
| Model output | `(N, 1, H, W)` logits; probability = sigmoid | `utils/inference.py` |
| Quantisation target | INT8 static QDQ, ONNX Runtime | not implemented |
| Onboard output | candidate polygons in tile CRS | not implemented |

## What is measured, estimated and pending

| Claim | Status | Evidence |
|---|---|---|
| Ground pipeline runs end to end on the four products | **Measured** | `outputs/baseline_v2_observation_valid/`, `outputs/audit/registration_v2_observation_valid/` |
| Ground inference writes a georeferenced raster | **Measured** | `outputs/predictions/smoke_inference_v1/`, on a 1024 x 1024 window |
| Model accuracy | **Pending** | no trained local model; no reviewed labels |
| INT8 quantisation parity and speed | **Pending** | ONNX export and QDQ not implemented |
| Onboard throughput, power, memory | **Pending** | no target board; no estimates are asserted here |
| Downlink saving from onboard filtering | **Pending** | depends on the candidate rate, which is unmeasured |

## What must not be claimed

- That the onboard stage has been implemented. It has not.
- That quantisation preserves accuracy. It has not been measured.
- That the current candidate rate would reduce downlink by a stated factor.
  The candidate rate is a 99th-percentile tail of an uncalibrated score, not a
  measured detection rate.

## Design risks

1. **Radiation tolerance.** No part of the compute budget can be claimed until
   a target device is chosen and its qualification is documented. Earlier notes
   in this project referenced a Zynq ZU3CG; no source supports calling it
   radiation-tolerant, so that claim was removed and must not return without a
   datasheet or a qualification report.
2. **Band availability.** L1C carries 13 bands; a flight instrument that
   downlinks only four would change the model contract.
3. **Quantisation drift.** The model uses BatchNorm. Static QDQ calibration on
   a small, unrepresentative set can shift the operating point. The threshold
   must be re-checked after quantisation, on validation data only.
