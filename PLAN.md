# Man-Made Change Detection — Project Plan

**Last updated:** 14 November 2026
**Status:** D2.6 partitions, original-OSCD loader, manual-label tooling, local patch indexing and training integration are implemented. Waiting for the user's manual label calls before local label rasters and real local patches can be produced. D2.4 OSM is deferred until that work is complete.
**Owner:** Aaryan Kurade
**Deadline:** 18 November 2026 (Wednesday)
**Schedule:** a five-day sprint, 13 to 17 November 2026. This includes the weekend. It is not five ordinary working days.
**Buffer:** 17 November is the final test day. Submit on 18 November.
**Compute:** ASUS TUF F15, i7-12650H, RTX 3050 Ti (4 GB VRAM), 16 GB RAM, WSL2, Ubuntu 24.04
**Interpreter:** Python 3.11 (see `environment.yml`)
(torch 2.11.0+cu128, rasterio 1.4.4, geopandas 1.2.0, shapely 2.2.0, pyproj 3.7.2,
onnx 1.21.0, onnxruntime-gpu 1.24.4, weasyprint 69.0, pdflatex, lualatex)

---

## 0. How to use this file

This file is the single source of truth for the project. Update it after each day.

Rules:

1. Every number in this file must come from a file in this repository, or from a reference in section 7.
2. When a measurement changes, update this file in the same commit.
3. When a decision is made, add it to section 11.
4. When a risk becomes a problem, record the date and the fix.

---

## 1. What I know for sure

### 1.1 The assignment

Title: "Interview Assignment : Man Made Change Detection".

The task: build a reproducible system that detects, localises and analyses structural man-made changes in satellite images from different dates. The system must separate these changes from natural surface variation and from differences caused by image acquisition. Every conclusion must have quantitative results and before/after evidence.

Seven work items:

1. Data requirements and execution plan.
2. Preprocessing and registration.
3. Architecture for onboard and ground execution.
4. Training and hyperparameter tuning.
5. Change analysis and validation.
6. Model quantization.
7. Reproducibility and decision records.

Plus: an AI chat record (AI_USAGE.md) and a submission package.

The evaluation criteria say: "Every reported conclusion must be traceable to data, an experiment or an explicitly stated assumption. A visually convincing change map without supporting analysis is insufficient."

### 1.2 The data

Four Sentinel-2A MSI Level-1C products. Two bi-temporal pairs.

| Pair | Tile | Date 1 | Date 2 | Gap | Orbit | Baseline | Archiving |
|------|------|--------|--------|-----|-------|----------|-----------|
| A | T43QBB | 2022-12-23 | 2023-05-12 | 4.7 months | 005 | 05.10 | S2RP |
| B | T43RGM | 2023-10-06 | 2026-10-22 | 3.0 years | 105 | 05.10 / 05.13 | S2RP / 2APS |

All products: EPSG:32643 (WGS84 / UTM zone 43N), descending orbit, product type S2MSI1C, TOA reflectance, uint16.

Raster grid: 10,980 x 10,980 px at 10 m = 109.8 km x 109.8 km. The nominal MGRS tile is 100 km x 100 km. The Copernicus documentation says each tile has a surface of about 110 x 110 km to provide overlap with the neighbouring tiles.

| Pair | Longitude | Latitude | Region |
|------|-----------|----------|--------|
| A (T43QBB) | 72.152 to 73.183 | 18.882 to 19.887 | Mumbai, Navi Mumbai, Thane, Kalyan, Vasai-Virar |
| B (T43RGM) | 77.032 to 78.176 | 27.922 to 28.891 | Delhi-NCR south: Faridabad, Palwal, Jewar |

Pair B is NOT Agra. Agra city is at 27.19 N, about 75 km south of the tile. Mathura city is at about 27.49 N, also south of the tile. Nuh town is at 77.01 E, about 3 km west of the tile.

Sites checked against the tile bounds:

| Site | Longitude | Latitude | Inside |
|------|-----------|----------|--------|
| Navi Mumbai International Airport | 73.030 | 18.980 | T43QBB |
| Mumbai Trans Harbour Link (Atal Setu) | 72.950 | 18.970 | T43QBB |
| Mumbai Coastal Road (Worli) | 72.820 | 18.985 | T43QBB |
| Jewar / Noida International Airport | 77.570 | 28.240 | T43RGM |
| Faridabad | 77.320 | 28.410 | T43RGM |
| Ballabhgarh | 77.320 | 28.220 | T43RGM |
| Nuh | 77.010 | 28.100 | outside |

Bands (read from MTD_MSIL1C.xml):

| Band | Resolution | Central wavelength (nm) |
|------|-----------|-------------------------|
| B01 | 60 m | 442.7 |
| B02 | 10 m | 492.7 |
| B03 | 10 m | 559.8 |
| B04 | 10 m | 664.6 |
| B05 | 20 m | 704.1 |
| B06 | 20 m | 740.5 |
| B07 | 20 m | 782.8 |
| B08 | 10 m | 832.8 |
| B8A | 20 m | 864.7 |
| B09 | 60 m | 945.1 |
| B10 | 60 m | 1373.5 |
| B11 | 20 m | 1613.7 |
| B12 | 20 m | 2202.4 |
| TCI | 10 m | 3-band true colour (uint8, visualisation only) |

Native pixel grids: 10 m bands 10980 x 10980; 20 m bands 5490 x 5490; 60 m bands 1830 x 1830.

### 1.3 Corrections applied in this version

| # | Error | Correction |
|---|-------|------------|
| C1 | Reflectance from the raw digital number | Apply the per-band radiometric offset. See 1.4.1. |
| C2 | MSK_CLASSI described as one band, all zero | It is a 3-band single-bit raster. See 1.4.2. |
| C3 | DynamicEarthNet described as Sentinel-2 | It is PlanetScope / Planet Fusion. Use OSCD. |
| C4 | Architecture called FC-EF | It is FC-Siam-diff. Use "FC-Siam-diff-inspired". |
| C5 | A union water mask | A union mask removes reclaimed land. Use three masks. |
| C6 | OSM treated as a label source | OSM is candidate context. Verify every label. |
| C7 | Deadline unknown | Deadline is 18 November 2026. Five-day sprint. |
| C8 | Zynq called radiation-tolerant | No source. Removed. |
| C9 | All bands resampled to 10 m | Upsampling creates no detail. Keep native resolution. |
| C10 | Full-tile float32 arrays | 13 bands x 2 dates is about 11.7 GiB. Use windowed I/O. |
| C11 | Sea as a stable target | The sea changes. Use land PIFs. |
| C12 | Index definitions not stated | Every formula is stated. See 4.3. |
| C13 | Pair B called "less reliable" | Re-measured. It is consistent. See 1.4.3. |
| C14 | Pair B called the "Agra pair" | It is Delhi-NCR south / Jewar. |
| C15 | Cut Phase 7 if short | Phase 7 is mandatory. Cut scope, not requirements. |
| C16 | MSK_CLASSI band order not stated | Confirmed: opaque, cirrus, snow/ice. |
| C17 | Baseline to PSD mapping missing | PB 05.11 is PSD 15.0. PB 05.13 is PSD 15.2. |
| C18 | Nuh listed as inside | It is 3 km west of the tile. Outside. |
| C19 | B11 reflectance 0.0153 | Wrong. Correct value is 0.0139. The 20 m band was read with a 10 m window. See 1.4.1. |
| C20 | Scenes called "cloud-free" | Too strong. The mask does not cover haze or cloud shadow. See 1.4.2. |
| C21 | NDBI grid not stated | Compute at 20 m. See 4.3. |
| C22 | Pair A window count "4 of 4" | It is 4 of 5. One window did not converge. See 1.4.3. |
| C23 | Invalid as a model class | Invalid is not a class. Use a validity mask. See 3.1. |
| C24 | Stage 2 undefined | Stage 2 is analyst-reviewed. See 3.2. |
| C25 | Agriculture rule too deterministic | NDVI and field shape are not proof. See 3.3. |
| C26 | OR-based multi-date merge | Use temporal persistence. See 4.10. |
| C27 | Area and length uncertainty | Use boundary sensitivity and a skeleton graph. See 4.11. |
| C28 | CUDA INT8 assumed | Provider availability is not op support. Verify. See 4.7. |
| C29 | CPU called eight cores | It is 10 cores (6P + 4E), 16 threads. See 4.6. |
| C30 | Jetson latency 1 ms per patch | No benchmark. Removed. |
| C31 | Offline inference as a requirement | It is a self-imposed constraint. See 4.14. |
| C32 | Hyperparameter search not scheduled | Added to Day 3. See 5. |
| C33 | config.yaml not in the checklist | Added. See 5 and 8. |
| C34 | OSCD split usage not stated | The original OSCD splits by region. OSCD100 cannot be split by location. See 4.2. |
| C35 | Mathura in the Pair B description | Removed. It is south of the tile. |

### 1.4 Measurements (corrected)

#### 1.4.1 Radiometric conversion

Read from MTD_MSIL1C.xml for every product and every band:

- QUANTIFICATION_VALUE = 10000
- RADIO_ADD_OFFSET = -1000 for all 13 bands in all four products

The official formula (ESA, PB 04.00 and later):

    reflectance = (DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE
    DN = 0 is NO_DATA

So: `reflectance = (DN - 1000) / 10000`.

Apply this only to valid pixels. Do not apply it to DN = 0.

Example, Pair A, Arabian Sea window. Each band uses its own native window. The 20 m band uses half the row and column indices of the 10 m window.

| Band | Resolution | Raw DN | Corrected reflectance |
|------|-----------|--------|-----------------------|
| B03 | 10 m | 2111 | 0.1111 |
| B04 | 10 m | 1751 | 0.0751 |
| B08 | 10 m | 1343 | 0.0343 |
| B11 | 20 m | 1139 | 0.0139 |

The old value 0.0153 was wrong. It came from reading the 20 m band with a 10 m window. The correct value is 0.0139.

Whole-tile water fraction. These are outputs of a chosen index and threshold, not independent measurements of real water coverage.

| Product | NDWI > -0.2 | NDWI > -0.1 | NDWI > 0.0 | NDWI > 0.1 | NDWI > 0.2 |
|---------|-------------|-------------|------------|------------|------------|
| QBB 2022-12-23 | 75.93% | 67.64% | 62.28% | 61.35% | 61.08% |
| QBB 2023-05-12 | 75.34% | 63.00% | 60.72% | 59.21% | 32.33% |
| RGM 2023-10-06 | 9.73% | 2.18% | 0.36% | 0.15% | 0.02% |
| RGM 2026-10-22 | 39.86% | 6.22% | 0.49% | 0.20% | 0.07% |

The threshold 0.0 is the physical boundary of the McFeeters index and is the most stable across dates: 62.28% vs 60.72% for Pair A. The threshold 0.2 is not stable: 61.08% vs 32.33%.

A combined rule, NDWI > 0.0 and B08 < 0.15, gives 62.27% vs 60.30% for Pair A and 0.36% vs 0.49% for Pair B. The audit uses NDWI > 0.0 as the water threshold.

#### 1.4.2 Quality masks

MSK_CLASSI (60 m, 1830 x 1830, 3 bands, one bit each):

| Band | Meaning |
|------|---------|
| 1 | Opaque clouds |
| 2 | Cirrus clouds |
| 3 | Snow and ice |

MSK_QUALIT (one file per band, 8 bands):

| Band | Meaning |
|------|---------|
| 1 | Ancillary lost data |
| 2 | Ancillary degraded data |
| 3 | MSI lost data |
| 4 | MSI degraded data |
| 5 | Defective pixels |
| 6 | Pixel nodata, inside the detectors |
| 7 | Partially corrected, cross-talk |
| 8 | Saturated pixels |

MSK_DETFOO (one file per band, 4 bits, 12 detector footprints). Not a cloud mask.

Measured:

- Product 2 has 259 pixels with value 1 in MSK_CLASSI band 2 (cirrus). That is 0.0077%, which matches the cloudy pixel percentage in the tile metadata.
- The other three products are all zero in all three MSK_CLASSI bands.
- All MSK_QUALIT files are zero in all four products.

Precise statement: the supplied MSK_CLASSI masks contain no opaque-cloud or snow/ice flags. One product contains 259 cirrus-flagged pixels, which will be masked. This does not establish that every pixel is free of haze or cloud-shadow contamination. A visual quality check and a separate invalid mask are still required.

#### 1.4.3 Registration

ECC on B08 at 10 m, 5 land windows per pair. The median global translation is applied and the residual is measured on the same windows.

| Pair | Initial dx | Initial dy | Initial correlation | Residual dx | Residual dy | Residual RMSE |
|------|-----------|-----------|---------------------|------------|------------|---------------|
| A | -1.18 m | -6.39 m | 0.831 | 0.06 m | -0.24 m | 0.6526 |
| B | +0.13 m | -1.19 m | 0.507 | 0.15 m | -0.09 m | 1.0128 |

Pair A: one window did not converge. The other four are consistent.

Pair B: the shift is small and consistent. The low correlation in some windows may come from real surface change, radiometric difference or unsuitable windows. This is not yet demonstrated.

After one global translation, the residual displacement is below 0.25 m for both pairs. That is below one tenth of a 10 m pixel. One global translation is sufficient for these two pairs.

The residual RMSE is not zero. It measures the whole image difference, including real surface change and the remaining radiometric difference, not only misregistration.

Do not report the estimated transform as the registration error. A single translation cannot correct local distortion.

#### 1.4.4 Normalisation status

Measured on held-out stable land pixels. The fit set and the validation set are spatial halves of the stable pixels.

| Pair | Method | MAE before | MAE after | Reduction | Change score before | Change score after |
|------|--------|-----------|-----------|-----------|---------------------|--------------------|
| A | none | 0.02403 | 0.02403 | 0.0% | 3.1900 | 3.1900 |
| A | global | 0.02403 | 0.00731 | 69.6% | 3.1900 | 3.4479 |
| A | pif | 0.02403 | 0.00777 | 67.7% | 3.1900 | 2.5262 |
| A | histogram | 0.02403 | 0.00796 | 66.9% | 3.1900 | 1.0782 |
| B | none | 0.01411 | 0.01411 | 0.0% | 1.1326 | 1.1326 |
| B | global | 0.01411 | 0.00765 | 45.8% | 1.1326 | 2.5002 |
| B | pif | 0.01411 | 0.00775 | 45.1% | 1.1326 | 1.4134 |
| B | histogram | 0.01411 | 0.00775 | 45.1% | 1.1326 | 0.9830 |

Findings:

- Every method reduces the mean absolute error on stable land by 45% to 70%.
- The global fit increases the change score for both pairs. It makes the stable pixels less consistent. Do not use it.
- The PIF fit reduces the change score for Pair A but increases it for Pair B.
- Histogram matching gives the lowest change score for both pairs: 1.0782 for A and 0.9830 for B.

The change score measures the spread of the difference on stable pixels. A lower value is better. The effect on real construction change is not yet measured. That check comes with the detection stage.

#### 1.4.5 Processing baselines

| Processing baseline | PSD version | Status |
|---------------------|-------------|--------|
| 05.10 | 14.9 (inferred) | Used by products 1, 2 and 3 |
| 05.11 | 15.0 | Confirmed by ESA |
| 05.13 | 15.2 | Confirmed by ESA. Used by product 4 |

The PB 05.13 L1C change that matters: the reflectance range is extended to -0.1 to 3.1767. The other PB 05.13 changes are L2A only.

Measured effect: the minimum DN is above 1000 for most bands in both baselines. A small fraction is below 1000 (B11, B12, B8A: 0.01% to 0.16%). The practical effect is small, but it must be checked on the dark pixels.

Do not claim that the baseline difference causes the radiometric offset. The offset is a product-level correction that applies to both. The remaining inter-date difference comes from the atmosphere, the sun angle and the surface state.

### 1.5 Environment

Now available: rasterio, geopandas 1.2.0, shapely 2.2.0, pyproj 3.7.2, numpy, scipy, scikit-image, opencv, matplotlib, PIL, pandas, torch, onnx, onnxruntime-gpu, tifffile, imageio, weasyprint 69.0.

ONNX Runtime execution providers on this machine:

    TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider

Provider availability does not prove that every INT8 operator is supported on CUDA. The actual operator support must be checked on the exported graph.

ONNX Runtime quantization API in this build: `quantize_static`, `quantize_dynamic`, `QDQQuantizer`, `QuantFormat.QDQ`. There is no function called `quantize_qdq`.

Also installed: pdflatex and lualatex. The report will be markdown to HTML to PDF with weasyprint.

---

## 2. Open decisions

| # | Decision | Options | Default |
|---|----------|---------|---------|
| D1 | Deadline | 18 Nov 2026 | Confirmed |
| D2 | Onboard hardware | laptop proxy / Jetson proxy / space estimate | Laptop measured. Others are estimates. |
| D3 | Labels | OSM + manual / manual only | OSM as candidate context, then manual verification |
| D4 | Public training data | OSCD / DynamicEarthNet / LEVIR-CD | OSCD first |
| D5 | Base bands | four 10 m bands / add 20 m | Four 10 m bands for the baseline |
| D6 | Report format | markdown + weasyprint / lattedit | markdown + weasyprint |

---

## 3. Target classes

### 3.1 Stage 1 — candidate change

The model outputs two classes:

| ID | Class | Meaning |
|----|-------|---------|
| 0 | unchanged | No apparent change |
| 1 | candidate_change | A spectral change that may be man-made |

A separate validity mask is applied to the loss and the metrics:

| Value | Meaning |
|-------|---------|
| 1 | valid |
| 0 | invalid: cloud, cirrus, NoData, saturated, unreliable registration |

Do not train the model to identify cloud, NoData or registration failure as change classes.

### 3.2 Stage 2 — event classification

Stage 2 is analyst-reviewed. A person examines each accepted candidate against the before/after images, the spectral evidence, the spatial context and the available reference information. The person assigns one event type:

| ID | Class | Type |
|----|-------|------|
| 3 | new_building | target |
| 4 | new_road | target |
| 5 | bridge_extension | target |
| 6 | demolition | target |
| 7 | land_reclamation | target |
| 8 | other_infrastructure | target |
| 9 | agriculture | non-target |
| 10 | water | non-target |
| 11 | vegetation | non-target |
| 12 | erosion_deposition | non-target |
| 13 | uncertain | uncertain |

The detection register records the source of each event type: model-predicted, analyst-assigned or uncertain.

Do not claim an automatic 13-class classifier unless it is implemented and evaluated.

### 3.3 Agriculture

Agriculture is class 9. It is non-target.

A candidate is classified as agriculture only when the evidence supports it. The evidence is: a regular field shape, a seasonal NDVI change, the image context and the available land-use information.

NDVI change and field shape alone are not proof. Soil preparation, harvesting, crop rotation, vegetation clearing and construction on agricultural land can look similar. If the cause cannot be established, use class 13.

The same caution applies to vegetation removal. Do not classify every vegetation change as non-target if there is evidence of a construction site.

---

## 4. Approach

### 4.1 Model

One architecture first. A shared-weight Siamese encoder with absolute feature differences and a U-Net decoder. This is FC-Siam-diff-inspired.

Do not train ChangeFormer at the same time. The local label set is small and the GPU has 4 GB.

### 4.2 Public data

Two related datasets exist. They are not the same.

OSCD (original):

- 24 pairs of multispectral Sentinel-2 images, taken between 2015 and 2018.
- 13 bands, registered pairs, 10 m / 20 m / 60 m.
- 14 training regions and 10 test regions.
- Pixel-level binary change labels. The changes are urban: new buildings and new roads.
- Images archive: 512,716,711 bytes (489 MiB) from the HuggingFace mirror `hkristen/oscd`.
- Sources: https://rcdaudt.github.io/oscd/ and IEEE DataPort.

OSCD100 (the dataset downloaded for this project):

- A subset of OSCD: 100 pre-cropped 256 x 256 pairs.
- Splits: 60 train, 20 val, 20 test.
- Images archive: 230,045,775 bytes (219 MiB) from the HuggingFace mirror `hkristen/oscd100`.
- Verified on the downloaded files: all 100 samples carry an identical
  `dates.txt` (date_1 20150801, date_2 20170801). The 100 crops come from ONE
  acquisition pair. The split is a random crop split, not a geographic split.
- The rasters have no CRS and an identity transform. They cannot be
  georeferenced or partitioned by location.
- The dataset card says OSCD100 is "designed for tutorials and demonstrations,
  not benchmarking".

Licence:

- The original OSCD project page does not state a licence.
- The HuggingFace OSCD100 card says "Same as the original OSCD dataset" and
  carries a CC-BY-NC-SA-4.0 tag added by the mirror uploader.
- Treat the licence as UNCONFIRMED. Do not redistribute the data.

Split usage:

- OSCD100 cannot be split at the geographic scene level. Its train/val/test
  are random crops of one region, so the validation and test scores are NOT
  independent of training. Use OSCD100 only as a pipeline smoke test, and say so.
- For a real geographic split, use the original OSCD 14 train / 10 test regions.
- Keep the local SkyServe geographic holdout untouched during fine-tuning.

DynamicEarthNet is PlanetScope / Planet Fusion. Not Sentinel-2. Extra experiment only.

LEVIR-CD is 0.5 m RGB from Google Earth. Not Sentinel-2. Extra experiment only.

### 4.3 Spectral indices

Compute after the radiometric conversion and after masking invalid pixels. State the grid for each index.

| Index | Formula | Grid |
|-------|---------|------|
| NDVI | (B08 - B04) / (B08 + B04) | 10 m |
| NDWI (McFeeters) | (B03 - B08) / (B03 + B08) | 10 m |
| NDBI | (B11 - B08) / (B11 + B08) | 20 m |

For NDBI, resample B08 to the B11 grid at 20 m. Do not upsample B11 to 10 m. The result is a 20 m index.

Add a safe division. Do not divide by zero.

Thresholds are experimental parameters, not universal constants. Record every threshold used.

### 4.4 Water and quality masks

Three separate masks:

1. Stable water core. Persistent water in both dates. Exclude from infrastructure detection.
2. Dynamic shoreline and water transition. Keep for reclamation and water-level analysis.
3. Invalid observation. Cloud, cirrus, NoData, saturated, unreliable registration. Exclude from all metrics.

Do not use a union water mask. It would remove reclaimed land.

### 4.5 Normalisation

Compare four methods on held-out stable land targets:

| Method | Description |
|--------|-------------|
| none | Corrected reflectance only |
| global | One straight line per band on all land pixels |
| pif | Select pseudo-invariant features, then one straight line per band |
| histogram | Histogram matching per band |

Metric: mean absolute difference on held-out stable targets. Also report the variance of the change score on stable targets.

The goal is a reduction of the stable-target difference. A method that also removes real construction change is not acceptable.

### 4.6 Hardware

| Platform | Role | CPU | Memory | Compute | Power | Storage | Latency | Status |
|----------|------|-----|--------|---------|-------|---------|---------|--------|
| ASUS laptop | ground proxy | i7-12650H, 10 cores (6P + 4E), 16 threads | 16 GB RAM | 8 cores visible in WSL | about 45 W, specification | 872 GB free | measured | measured |
| RTX 3050 Ti | ground proxy | — | 4 GB VRAM | 2560 CUDA cores | about 50 W, specification | 4 GB VRAM | measured | measured |
| Jetson Orin Nano 8 GB | onboard proxy | — | 8 GB | up to 40 TOPS | 7 to 15 W, specification | microSD | not measured | estimate, no board |
| Space-qualified device | onboard target | — | not defined | not defined | not defined | not defined | not defined | conceptual only |

WSL reports 8 cores and 16 threads. The physical CPU has 10 cores and 16 threads. Report both.

Do not claim a device is radiation-tolerant or space-qualified without a device-specific source. Do not report a latency that was not measured.

### 4.7 Quantisation

Export the model to ONNX FP32. Check that the ONNX output matches the PyTorch output. Then quantise to INT8 with ONNX Runtime using `quantize_static` and `QuantFormat.QDQ`. Calibrate on training patches only. Evaluate FP32 and INT8 on the same locked test patches.

Before quantising:

1. Export to ONNX FP32.
2. Verify ONNX predictions against PyTorch.
3. Select a calibration subset from the training data only.
4. Quantise with QDQ.
5. Inspect the execution-provider partitioning.
6. Confirm which operations run on CUDA and which fall back to CPU.
7. Benchmark the QDQ model on CPU as well.
8. Do not assume the quantised model is faster on CUDA.

The CPU INT8 route is the initial measurable route. Investigate GPU INT8 only after verifying operator support.

Report: file size, F1, IoU, latency, peak memory, and the smallest detected object.

**Status on 2026-11-15: all eight steps are done, with one divergence.**

| Step | Result |
|---|---|
| 1. Export FP32 ONNX | 1,941,434 B, dynamic batch/height/width, TorchScript exporter |
| 2. Verify against PyTorch | max abs difference 1.52e-06, threshold agreement 1.0000 |
| 3. Calibration subset | training split only, the first 20 batches at batch size 8 |
| 4. QDQ quantise | 550,136 B, QInt8 weights and activations, per-channel |
| 5. Provider partitioning | 202 of 228 kernels on CUDA (88.6 %), 26 Memcpy boundaries |
| 6. CUDA against CPU placement | confirmed on the QDQ graph; see the decision log entry on the silent CPU fallback |
| 7. CPU benchmark | FP32 8.76 ms, INT8 9.94 ms, median of 20 runs |
| 8. Do not assume INT8 is faster | confirmed: INT8 is 13 % slower on CPU |

**Divergence.** This section planned to evaluate FP32 and INT8 on the same
locked test patches. The implementation evaluated both on the validation split
instead: AP 0.3362 to 0.2620, AUC 0.9025 to 0.8895, mean absolute probability
shift 0.1848 over 62 validation windows. The test split was evaluated once, for
the FP32 PyTorch checkpoint selected by validation AP, and was not reopened for
a quantisation variant. The reason is that the INT8 cost is a diagnostic of the
quantisation method, not a model-selection result, and each opening of the test
split weakens its value as a held-out set.

If the submission needs a test-set INT8 figure, that is a deliberate additional
test-set use and needs a decision first. Nothing in this repository currently
claims one.

### 4.8 Requirement traceability

| Assignment requirement | Where |
|------------------------|-------|
| Audit the imagery | Day 1, `outputs/audit/data_audit.md` |
| Define the required bands | Section 1.2 |
| Define the spatial resolution | Section 1.2 |
| Define the metadata | Section 1.2 and 1.4 |
| Define the image quality | Section 1.4.2 |
| Define the reference labels | Section 3 |
| Define the computational resources | Section 4.6 |
| Explain the workflow | Sections 4 and 5 |
| Handle two or more observations | Section 4.10 |
| State the requirements the data does not meet | Section 4.9 |
| Identify and justify extra data sources | Section 4.2 |
| Preprocessing and registration | Section 4.4 and 4.5, Day 1 |
| Measure registration error | Section 1.4.3, Day 1 |
| Explain residual misalignment | Section 6, risk R14 |
| Onboard and ground architecture | Section 4.1 and 4.6 |
| Hardware limits | Section 4.6 |
| Separate measurements from estimates | Section 4.6 |
| Parameter count, training data, loss, augmentation, optimizer | Day 3, `docs/report.md` |
| Reproducible hyperparameter search | Day 3 |
| Compare against a baseline | Day 2 and Day 3 |
| Class imbalance and false positives | Day 3 |
| Metrics: precision, recall, F1, overlap | Day 3 |
| How labels are obtained and checked | Day 2 |
| Stop geographic and temporal leakage | Section 4.2, Day 2 |
| Do not treat a difference map as truth | Section 4.2 |
| Location, type, extent, confidence | Day 4, detection register |
| Measure area or length, state uncertainty | Section 4.11 |
| Rejected apparent changes with evidence | Day 4 |
| Failures and unassessable regions | Day 4 |
| What two dates can and cannot show | Section 4.12 |
| Quantise at least one model | Day 3 |
| Compare original and quantised | Day 3 |
| File size, performance, memory, latency | Day 3 |
| Loss of sensitivity to small changes | Day 3 |
| Calibration data, supported ops, limits | Day 3 |
| Git repository with commits | This repository |
| Assumptions, alternatives, decisions | `docs/decisions/` |
| Version configurations and manifests | `configs/config.yaml`, `requirements.txt` |
| Dependencies and random seeds | `requirements.txt`, `configs/config.yaml` |
| Reproduction instructions | `README.md` |
| Attribute external data, code, weights | Section 4.13 |
| AI chat record | `AI_USAGE.md` |

### 4.9 Data gaps

| Requirement | What the data provides | Gap | Plan |
|-------------|------------------------|-----|------|
| Reference labels | None supplied | No ground truth | Make labels from OSM plus manual work |
| Many change examples | Two pairs | Too few to train from scratch | Pretrain on OSCD, then fine-tune |
| Cloud mask | MSK_CLASSI, 3 bands | Only cloud and cirrus, no water | Build a water mask from the bands |
| Water classification | No L2A SCL | No official water class | Build a mask from NDWI and B08 |
| Multi-date coverage | Two dates per pair | No exact change date | State the limit, section 4.12 |
| Onboard hardware | None | No board to measure | Use the laptop as the proxy |
| Historical labels | OSM is current | OSM is not historical truth | Verify every label in the images |

### 4.10 Handling more than two observations

The system compares two dates. For more than two dates, it compares consecutive pairs and merges the results.

Merge rule: a candidate is accepted only if the change persists. A pixel that changes and then returns to its original state is not a persistent change. Use a baseline-versus-date comparison, not a simple OR.

Event date: with sparse observations, report an event interval between the last observation showing the old state and the first observation showing the new state. Do not report a single exact date.

A third date can show persistence or reversal. It does not automatically give a reliable rate.

### 4.11 Area and length

Measure area in EPSG:32643, in square metres.

Area uncertainty: use a boundary sensitivity analysis. Measure the area after one-pixel erosion and one-pixel dilation. Report the range. Add the registration uncertainty.

Road length: thin the road mask to a one-pixel skeleton. Build a skeleton graph. Count 10 m for horizontal and vertical neighbours and 10 sqrt(2) m for diagonal neighbours. Do not multiply the skeleton pixel count by 10 m, because that underestimates diagonal segments.

State the resolution limit. A 10 m pixel cannot resolve an object smaller than about 10 m.

### 4.12 What two dates can and cannot show

Two dates show that a change occurred between them. They show the location, the type and the extent.

Two dates cannot show:

- The exact date of the change.
- The rate of change.
- The cause, without context.
- Whether the change continued after the second date.

A third date would show the direction and the rate. More dates would show the full timeline.

### 4.13 Attribution

| Source | Use | Licence |
|--------|-----|---------|
| Copernicus Sentinel-2A MSI L1C | Study imagery | Copernicus open data |
| OSCD, Daudt et al. 2018 | Pretraining | Not stated on the project page |
| OpenStreetMap | Candidate labels | ODbL, attribution required |
| PyTorch, rasterio, ONNX Runtime | Software | Their own licences |
| Pretrained encoder weights | Transfer learning | To be recorded with the checkpoint |

### 4.14 Self-imposed constraints

These are not assignment requirements. They are design choices.

- Offline inference. The assignment does not require it. It is a sensible assumption for an onboard system.
- A 10 m base resolution. The assignment asks for localisation. 10 m is the finest common resolution.
- A two-stage design. The assignment asks for an uncertain category, which the two-stage design supports.

---

## 5. Five-day schedule

A five-day sprint, including the weekend. 13 to 17 November 2026.

### Day 1 — Friday 13 November: audit and preprocessing

Tasks:

1. Write `data/safeloader.py`. Read QUANTIFICATION_VALUE and RADIO_ADD_OFFSET. Convert to reflectance. Preserve DN=0 as NO_DATA.
2. Write `data/masks.py`. Decode MSK_CLASSI (3 bands), MSK_QUALIT (8 bands), MSK_DETFOO.
3. Write `data/register.py`. ECC over several land windows. Report the median shift, the spread and the failed windows.
4. Apply the shift. Measure the residual displacement on stable control features. Report the residual RMSE in pixels and metres.
5. Write `preprocess/normalize.py`. Four methods. Held-out stable land targets.
6. Write `preprocess/water.py`. Three masks.
7. Write `scripts/audit.py`. Reproduce every number in section 1.4.
8. Commit.

Outputs: `outputs/audit/data_audit.md`, `outputs/audit/registration.json`,
`outputs/audit/normalization_comparison.csv`, `outputs/audit/water_masks.tif`,
`outputs/figures/audit_*.png`.

Gate: every number in section 1.4 is reproduced. The residual registration error is reported.

### Day 2 — Saturday 14 November: dataset, partitions, labels, patches

The implementation order is D2.6 partitions, D2.5 manual label set, then D2.7
patches and index. D2.4 OSM is deferred until the manual label and patch work is
complete. OSM is candidate context only; it must not create ground-truth labels.

Implemented and checked:

1. D2.6: create geographic train/validation/test stripes for each local pair,
   with a 500 m buffer on each side of split boundaries. Choose orientation from
   valid-land coverage only. The official OSCD test regions remain unchanged.
2. Original OSCD: audit the 24 regions, preserve the official 10-region test
   list, hold out `bordeaux` and `pisa` for validation, and exclude `mumbai`
   because it overlaps local Pair A. Build a windowed OSCD dataset and train
   through `train.py` while preserving the existing OSCD100 loader when it is
   explicitly selected.
3. D2.5: define a human-only GeoJSON label schema, candidate GIS layers,
   validation, and rasterization. The user makes each label decision. No local
   labels have been supplied or created.
4. D2.7: create a local patch index and windowed SAFE patch loader. It enforces
   partition membership, ignores unlabeled pixels in the loss and metrics, and
   can materialize train/validation NPZ patches. It omits test patches unless
   explicitly requested after model selection.
5. Keep OSM out of labels. No Overpass query was made in this phase.

Outputs and verification:

- Original OSCD index: 399 train, 62 validation and 230 official test windows
  at 128 x 128 pixels. Test-label pixels were not read by index construction.
- Original OSCD train-only band means/std were computed. A one-epoch,
  one-batch-per-split CUDA smoke run completed. This is a pipeline check, not
  a benchmark result.
- The existing OSCD100 path still completed a one-epoch smoke run when selected
  with `--dataset oscd100`.
- Local partitions are written under `outputs/partitions/`; the verified
  summary is `outputs/partitions/partition_summary.json`.
- The candidate GIS layers contain 71 Pair A and 181 Pair B component
  bounding boxes. They remain unreviewed candidates, not labels.
- Synthetic-only checks passed for manual-label rasterization, split-boundary
  and automated-source rejection, patch indexing, SAFE window reads, masked
  loss, and local training. These synthetic labels were kept in the Hermes
  scratch directory, not in the project data tree.
- No local manual label raster, project patch archive, or local patch index has
  been generated. Those outputs require the user's annotation decisions.

Gate: wait for reviewed user GeoJSON before generating local labels or patches.
Do not begin D2.4 OSM until D2.5 decisions and D2.7 patches/index are complete.

### Day 3 — Sunday 15 November: training, tuning, evaluation, quantisation

Tasks:

1. Fine-tune on the local labels. Use only the train partition.
2. Run a small controlled hyperparameter search. Keep the architecture fixed. At least three trials plus the baseline. Record the seed and the validation result.
3. Write `eval.py`. Pixel metrics, object metrics, confusion matrix, size-stratified recall.
4. Evaluate the classical baseline and the learned model on the same locked test partition.
5. Write `quantize.py`. ONNX FP32, parity check, INT8 QDQ, calibration.
6. Inspect the execution-provider partitioning. Benchmark CPU INT8 and, if supported, CUDA INT8.
7. Write `scripts/benchmark.py`. File size, latency, peak memory, smallest detected object.
8. Commit.

Outputs: `outputs/eval/metrics.csv`, `outputs/eval/confusion_matrix.png`,
`outputs/hparam/search.csv`, `configs/config.yaml`,
`outputs/quant/quantization_comparison.csv`, `models/onboard_fp32.onnx`,
`models/onboard_int8.onnx`.

Gate: the FP32 and INT8 models are evaluated on the same locked test patches. The hyperparameter search is recorded.

### Day 4 — Monday 16 November: outputs and report draft

Tasks:

1. Write `infer.py`. Full-tile inference with overlap blending. Stitch, threshold, label components.
2. Write `scripts/make_register.py` and `scripts/make_evidence.py`.
3. Write the georeferenced masks and the detection register.
4. Measure area and length. Use the boundary sensitivity and the skeleton graph.
5. Write `docs/report.md`. Cover all seven work items.
6. Commit.

Outputs: `outputs/register/detections.csv`, `outputs/register/detections.geojson`,
`outputs/masks/*.tif`, `outputs/figures/before_after/*.png`, `docs/report.md`.

Gate: every detection has a location, a class, an extent and a confidence.

### Day 5 — Tuesday 17 November: final test and submission package

Tasks:

1. Convert the report to HTML and PDF with weasyprint.
2. Write `docs/decisions/*.md`.
3. Write `AI_USAGE.md`. Every AI chat, tool name, full prompts and responses in time order.
4. Write `README.md`. Exact commands.
5. Write `scripts/demo.sh`. A reproducible demonstration run.
6. Freeze `requirements.txt`.
7. Run the demo from a clean state. Fix any error.
8. Final commits.

Outputs: `docs/report.pdf`, `docs/decisions/*.md`, `AI_USAGE.md`,
`README.md`, `scripts/demo.sh`.

Gate: a fresh clone can run the commands in the README. The demo runs without error.

### Wednesday 18 November: submit

Submit the repository, the report, the model artifacts and the AI usage record.

---

## 6. Edge cases and risks

| # | Risk | Why it matters | Mitigation |
|---|------|----------------|------------|
| R1 | Only two pairs | Not enough to train from scratch | Train on OSCD, then fine-tune |
| R2 | Radiometric offset not applied | Wrong reflectance and indices | Apply RADIO_ADD_OFFSET. Recompute everything |
| R3 | Different processing baselines | A radiometric difference | Normalise. Do not blame the baseline without evidence |
| R4 | Different seasons | A vegetation and soil moisture difference | PIF normalisation on land |
| R5 | Sea covers a large part of Pair A | Water changes with tide and glint | Separate stable water, dynamic shoreline and invalid masks |
| R6 | MSK_CLASSI has 3 bands | A misread mask hides cloud | Decode all 3 bands. Mask the 259 cirrus pixels |
| R7 | Agriculture looks like change | False positives | Class 9. Field shape plus seasonal NDVI plus context |
| R8 | Small changes at 10 m | A small building is a few pixels | Patch size 256. Sensitive loss. Test the quantisation loss |
| R9 | Class imbalance | Few changed pixels | Weighted loss. Report per-class metrics |
| R10 | OSM labels are not historical truth | False labels | Verify every label in the images |
| R11 | Leakage between partitions | Inflated metrics | Split by geography before patching |
| R12 | Quantisation loses small changes | The assignment asks for this | Report the smallest detected object for FP32 and INT8 |
| R13 | Onboard memory and power | The model must fit | A small model. INT8. Report the limits |
| R14 | Residual misregistration | False edges | Measure and report. Mask a 1 px border |
| R15 | No internet at inference | Self-imposed constraint | All weights local |
| R16 | 4 GB VRAM is small | Training is hard | Small patches. AMP. Gradient accumulation |
| R17 | Five days is a short time | The plan may fail | Cut scope, not requirements |
| R18 | The report is mandatory | A missing report fails the submission | Write it on Day 4 and Day 5 |
| R19 | OSCD licence is not stated | A licensing risk | State the source and the citation. Do not redistribute |
| R20 | The OSCD split is small | The model may overfit | Use the 14 training pairs, augment, and fine-tune on local labels |
| R21 | CUDA INT8 may not be supported | A misleading speed claim | Benchmark CPU INT8. Check operator support |
| R22 | Pair A registration has one failed window | A local distortion | Report the failed window. Mask it if needed |

---

## 7. References

Sentinel-2 data and products:

- ESA Sentinel-2 mission: https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2
- Copernicus Data Space documentation, Sentinel-2: https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html
- SentiWiki document library: https://sentiwiki.copernicus.eu/web/document-library
- ESA Sentinel-2 major products upgrade (PB 04.00): https://sentinels.copernicus.eu/-/copernicus-sentinel-2-major-products-upgrade-upcoming
- ESA introduction of PB 05.13: https://sentinels.copernicus.eu/-/introduction-of-copernicus-sentinel-2-processing-baseline-05.13
- Sentinel-2 PSD 15.0 announcement (PB 05.11): https://sentinels.copernicus.eu/-/sentinel-2-product-specification-document-psd-in-version-15.0
- PSD 15.2 (PB 05.13): https://sentiwiki.copernicus.eu/__attachments/a_79d81f1db038da666d7831a1b76a44df399e5a67ba5e5f39f1f7038d75cb1fb9/S2-Sentinel-2%20Products%20Specification%20Document-15.2.pdf
- NASA Earthdata Sentinel-2 MSI: https://www.earthdata.nasa.gov/data/instruments/sentinel-2-msi

Mask band definitions:

- eoreader S2MaskBandNames: https://eoreader.readthedocs.io/stable/api/eoreader.bands.band_names.S2MaskBandNames.html

Change detection models and data:

- OSCD dataset: https://rcdaudt.github.io/oscd/
- OSCD on IEEE DataPort: https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection
- Daudt et al., Fully Convolutional Siamese Networks for Change Detection: https://arxiv.org/abs/1810.08462
- Daudt et al., Urban Change Detection for Multispectral Earth Observation: https://arxiv.org/abs/1810.08468
- DynamicEarthNet: https://arxiv.org/abs/2004.05507
- LEVIR-CD: https://github.com/justchenhao/LEVIR
- ChangeFormer: https://arxiv.org/abs/2201.01293

Quantisation and deployment:

- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/
- NVIDIA TensorRT INT8: https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/
- NVIDIA Jetson modules: https://developer.nvidia.com/embedded/jetson-modules

Radiometric normalisation:

- Canty, Nielsen. Automatic radiometric normalization of multitemporal satellite imagery with the iteratively re-weighted MAD transformation. Remote Sensing of Environment, 2008.

OpenStreetMap:

- Attribution and ODbL: https://www.openstreetmap.org/copyright

---

## 8. Deliverables checklist

State on 2026-11-15. "Pending owner" means the item cannot be finished without
the manual labels or an owner decision.

| Item | Location | Status |
|------|----------|--------|
| Source repository with git history | this repository | written, publication path in `docs/publication_plan.md` |
| README with run instructions | `README.md` | complete |
| Environment specification | `requirements.txt`, `environment.yml` | complete |
| Configuration file | `config.yaml` | complete; PLANNED blocks marked |
| Data manifest | `configs/data_manifest.yaml` | complete |
| Corrected data audit | `outputs/audit/data_audit.md` + `docs/audit_reconciliation.md` | complete, reconciliation written |
| Registration result | `outputs/audit/registration_v2_observation_valid/registration.json` | complete, masked |
| Normalisation comparison | `outputs/audit/normalization_v2_observation_valid/normalization_comparison.csv` | complete; method choice pending owner |
| Water and quality masks | `outputs/audit/water_masks_{A,B}.tif` | complete |
| Labels | `labels/` | **pending owner** manual GeoJSON |
| Label statistics | `outputs/labels/label_stats.csv` | **pending owner** |
| Classical baseline | `outputs/baseline_v2_observation_valid/` | complete |
| Trained model | `outputs/checkpoints/oscd_gpu_v1/` | complete (original OSCD, 3 seeds + 3 trials) |
| Training log | `outputs/checkpoints/oscd_gpu_v1/*/seed_*/train_log.csv` | complete, per-epoch GPU and VRAM |
| Hyperparameter search | three controlled trials, one variable each | complete as trials; no grid search |
| Evaluation metrics | `outputs/eval/oscd_baseline_seed42_test/metrics.json` | complete, locked test |
| Confusion matrix | pixel tp/fp/fn/tn in `metrics.json`; object and size metrics too | complete as numbers; no PNG rendered |
| Quantised model | `outputs/quant/oscd_baseline_seed42_v2/onboard_int8.onnx` | complete |
| Quantisation comparison | `outputs/quant/.../quantization_report.json`, `outputs/bench/oscd_validation_providers/` | complete, three providers |
| Detection register | `outputs/labels/register/` | tooling complete, **register empty pending owner** |
| Georeferenced masks | `outputs/predictions/` | smoke tests only, `transfer_smoke_not_a_result` |
| Before/after evidence | `outputs/figures/` | complete for both pairs, plus evidence-panel tooling |
| Technical report | `docs/report.md`, `docs/report.pdf` | draft complete, PDF rendered |
| Decision records | `docs/decisions/` | 8 records, 1 open |
| AI usage record | `AI_USAGE.md` | present, final organisation pending owner |
| Reproducible demonstration run | `scripts/demo.sh` | complete, read-only, under one minute |

---

## 9. Verification log

Checks done on 13 November 2026.

| Check | Result | Source |
|-------|--------|--------|
| RADIO_ADD_OFFSET present and -1000 | Confirmed | Product metadata |
| Radiometric formula | Confirmed | ESA PB 04.00 page |
| MSK_CLASSI is 3 bands | Confirmed | ESA PB 04.00 page and product raster |
| MSK_CLASSI band order | Confirmed: opaque, cirrus, snow/ice | eoreader, after the PSD |
| MSK_QUALIT band order | Confirmed: 8 bands | eoreader, after the PSD |
| Product 2 cirrus pixels | 259, equals 0.0077% | Product raster and tile metadata |
| DynamicEarthNet sensor | PlanetScope / Planet Fusion | arXiv 2004.05507 |
| OSCD sensor | Sentinel-2, 13-band | OSCD project page |
| OSCD size and split | Original: 24 pairs, 14 train, 10 test, images archive 489 MiB. OSCD100: 100 crops, 60/20/20, images archive 219 MiB | Measured HTTP Content-Length on the HuggingFace mirrors |
| OSCD100 acquisition dates | All 100 crops: 20150801 and 20170801 | The downloaded `dates.txt` files, all identical |
| OSCD100 georeferencing | No CRS, identity transform | rasterio on the downloaded GeoTIFFs |
| OSCD mirror reachable | HTTP 200 | `hf.co/datasets/hkristen/oscd` and `hkristen/oscd100` |
| FC-Siam-diff definition | Shared weights plus absolute difference | arXiv 1810.08462 |
| Zynq ZU3CG radiation tolerance | Not supported by any source | AMD product pages |
| MGRS tile size | 100 km nominal, about 110 km raster | Copernicus Data Space docs |
| PB 05.13 mapping | PSD 15.2 | ESA PB 05.13 page |
| PB 05.11 mapping | PSD 15.0 | ESA PSD 15.0 page |
| PB 05.13 L1C change | Reflectance range extended to (-0.1, 3.1767) | ESA PB 05.13 page |
| Minimum DN | Above 1000 for most bands; a small fraction below | Product raster |
| ONNX Runtime providers | TensorRT, CUDA, CPU | Local install |
| ONNX Runtime quantization API | `quantize_static`, `QDQQuantizer` | Local install |
| Pair A registration | -1.18 m, -6.39 m, consistent, 1 window failed | Local ECC, 5 windows |
| Pair B registration | +0.13 m, -1.19 m, consistent | Local ECC, 5 windows |
| Site locations | 6 of 7 inside the tiles; Nuh outside | UTM conversion |
| B11 sea-window reflectance | 0.0139, not 0.0153 | Corrected local computation |
| CPU | i7-12650H, 10 cores (6P + 4E), 16 threads | lscpu and Intel spec |
| Mathura location | About 27.49 N, south of the tile | Standard reference |

### 9.1 Day 1 code review

On 13 November 2026 I reviewed every file written on Day 1 and re-verified the numbers independently.

Fixed:

- `.gitignore`: the negation patterns for `outputs/audit/*` came before the `outputs/**/*.csv` exclusion, so git applied the last matching pattern and the CSV stayed ignored. The negations now come after.
- `config.yaml`: `water_ndwi` was 0.2, which contradicts the audit. Changed to 0.0. `normalize` was `pif`; the audit shows histogram matching is best, so it is now `histogram`.
- `README.md`: the usage section named `scripts/audit_data.py`, which does not exist. It is `scripts/audit.py`.
- `scripts/audit.py`: the raw DN was recovered with hardcoded constants. It now uses the product's own quantification value and offset.
- `scripts/audit.py`: the band table said "see MTD_MSIL1C.xml" for the wavelength. The wavelengths are now read from the metadata.
- `scripts/audit.py`: removed dead code.
- `data/register.py`: the module docstring described the direction convention ambiguously. It now states that the offsets are the content displacement and that the alignment negates them.
- `scripts/run_registration.py`: added a note that the residual is measured on the same windows as the estimate, so it is a consistency check, not an independent validation set.

Independent verification:

- The sea-window reflectances were recomputed with raw rasterio reads, bypassing the loader. They match: B03 0.1111, B04 0.0751, B08 0.0343, B11 0.0139.
- The `out_shape` transform was checked. It gives 100 m pixels with the correct origin.
- The water fraction depends on the resampling method. Bilinear gives 62.28% and 60.72%; nearest gives 63.10% and 61.01%. The audit now records both.
- The ECC convention was verified with a controlled test using `np.roll`. The self-test now checks the alignment RMSE.

Remaining limitation: the registration residual is not measured on independent control features.

Open items:

- The exact PSD version for PB 05.10. Inferred to be 14.9.
- The exact effect of the PB 05.13 reflectance range change on the dark pixels.
- The OSCD licence. RESOLVED 2026-11-14 from the primary source (IEEE DataPort, DOI 10.21227/asqe-7s69): the imagery is modified Copernicus Sentinel data 2015-2018 (ESA), and the change labels are CC BY-NC-SA. See `docs/reference_analysis.md` §0.
- Whether the four scenes contain enough visible construction at 10 m. This is a data check, not a document check.
- Whether to switch to the original OSCD (14 train / 10 test regions) for a real geographic split. OSCD100 is one acquisition pair and cannot support one.
- The OSM data source is blocked from WSL. The client now sends an identifying User-Agent and stops on HTTP 406/429. No second batch has been attempted.

### 9.2 Day 2 review

On 14 November 2026 I reviewed everything written after Day 1. The review found
that the pipeline ran, but the results were not reportable and the repository
did not describe its own code.

Defects found and fixed:

- D1 `config.yaml` described a ResNet-18 encoder, 64 base filters, 2 classes,
  100 epochs and 2 workers. The code is a custom Siamese U-Net, base 16, one
  output channel. `train.py` now reads `config.yaml`, and every run writes a
  `run_config.json` with the resolved values and the CLI overrides.
- D2 The plan said OSCD has 14 training pairs. OSCD100 has 60 train scenes.
  The plan now separates the original OSCD (24 pairs, 14 train / 10 test
  regions, 489 MiB archive) from OSCD100 (100 crops, 60/20/20, 219 MiB).
- D3 The plan said to define geographic partitions first. OSCD100 has no CRS
  and an identity transform, so it cannot be partitioned by location. The
  partition requirement now applies to the local Sentinel-2 tiles only.
- D4 The output was named `change_mask_A.tif` while marking 32% of land. It is
  now `candidate_mask_A.tif`, tagged
  `exploratory_candidate_mask_not_a_detection`.
- D5 The checkpoint stored only bands, mean, std, base and seed. It now also
  stores the full run configuration.
- D6 The truncated log `outputs/baseline_run.txt` from a killed run is deleted.
- D7 Nine scripts hardcoded local product paths. A new
  `data/manifest.py` loads `configs/data_manifest.yaml`, and all nine scripts
  now resolve paths through it. A grep for the hardcoded path returns nothing.
- D8 The morphology and validity filter was written three times. It is still
  three copies; this is a known drift risk and is not fixed yet.
- D9 `data/dataset.py` asserted an unverified pixel scaling. The docstring now
  states the acquisition dates and marks the scaling as unverified.

Verified during the fix:

- `python -m data.manifest` resolves both pairs and checks every path exists.
- All nine scripts compile. No hardcoded product path remains.
- `train.py` runs from `config.yaml` and writes `run_config.json`.
- The classical baseline re-ran for Pair A after the refactor and produced
  `candidate_mask_A.tif`, `candidate_components_A.csv` and
  `change_score_A.tif` with the correct tags and a 20 m / 5490 x 5490 grid.

New finding, not a code defect but a data defect:

- All 100 OSCD100 samples carry an identical `dates.txt`
  (date_1 20150801, date_2 20170801). The 100 crops come from ONE acquisition
  pair. The 60/20/20 split is a random crop split. Validation and test are
  NOT independent of training. OSCD100 is a pipeline smoke test only.

Still not usable:

- The latest percentile candidate masks remain uncalibrated and unverified. The
  99th-percentile run used thresholds 4.2968 (Pair A) and 5.3881 (Pair B).
  After morphology and component filtering, the masks contain 11,554 and 67,145
  pixels, respectively. These are candidate pixels, not confirmed construction.
- A previous five-epoch OSCD100 run reached validation F1 0.5017 at epoch 4 and
  0.1824 at epoch 5 with BatchNorm and batch size 1. It is not a benchmark.
  The later three-seed smoke run used only one batch per split and is also not a
  performance result.

### 9.3 Reference-technique adoption review

On 14 November 2026 I implemented the selected techniques from
`mjsushanth/oscd-sentinel2-change-detection` and the related folder-inference
reference. I did not add the original-OSCD loader or label tooling.

Verified code and tests:

- Undefined normalized differences return NaN. `python -m preprocess.indices`
  checks zero denominators, real zero-valued indices and uint16 arithmetic.
- `utils/metrics.py` returns NaN for undefined metrics and passes the checked AP
  and ROC-AUC examples, tie cases and degenerate cases.
- The classical baseline reads the 99th-percentile prior from `config.yaml`.
  Pair A and B ran. Their GeoTIFF tags record the percentile, threshold value
  and `percentile_prior_not_calibrated` status. Both remain exploratory masks.
- Training selects checkpoints by validation AP, records AP/AUC and computes
  `pos_weight` from training masks only. A CUDA one-epoch smoke run completed.
- The configured seed list `[42, 43, 44]` is now used by default. A smoke run
  with one train and one validation batch per seed wrote per-seed metrics and
  mean/std rows for AP, AUC, F1 and IoU in
  `outputs/checkpoints/adoption_smoke_final/multi_seed_summary.csv`.
- `utils/inference.py` averages overlapping patch probabilities. Its self-check
  covers image edges, small images, overlap, and rejects gaps from stride larger
  than patch size.
- `scripts/predict_folder.py` processed two synthetic four-band NPY pairs and
  wrote masks, comparison figures and `summary.csv` under the Hermes scratch
  directory. This does not read SAFE products or produce georeferenced output.
- `data/augment.py` applies the same flips and quarter-turn rotations to both
  dates and the mask. Sixteen deterministic transform combinations and the
  random-sample interface passed self-checks. Training-only augmentation is
  enabled in `config.yaml` and stored in each run configuration.

The official PyTorch 2.11 `BCEWithLogitsLoss` documentation confirms that
`pos_weight = N_negative / N_positive` is the class-balancing example, and
that values above one increase recall:
https://docs.pytorch.org/docs/2.11/generated/torch.nn.BCEWithLogitsLoss.html

The external API guidance used for paired segmentation transforms was read from
PyTorch's official TorchVision transforms documentation:
https://docs.pytorch.org/vision/stable/transforms.html
The implementation uses native tensor operations and adds no TorchVision
runtime dependency.

Historical limits from the reference-technique adoption pass: OSCD100 smoke
metrics were not a benchmark. At that point no geographic benchmark existed.
Folder inference had no georeferencing, and the percentile threshold had no
label-based calibration. Those limits remain where applicable. Do not report
smoke-test metrics as model performance.

### 9.4 Original OSCD, partitions, manual labels and local patches

On 14 November 2026 I implemented the original-OSCD loader and the D2.6/D2.5/
D2.7 tooling. The user's requested order is preserved: partitions first,
manual labels second, patches/index third. D2.4 OSM remains deferred.

Original OSCD loader and training integration:

- `scripts/audit_oscd_original.py` audited 24 regions and 13 bands. It confirmed
  14 official training regions and 10 official test regions.
- A separate scan of B02/B03/B04/B08 in both dates of all 24 rectified regions
  found zero zero-valued input pixels. This does not replace the product QA
  audit.
- The label TIFFs use 1=no change and 2=change in all 24 audited regions. This
  differs from the downloaded label README, which documents 0/1. The loader
  checks each complete mask and accepts only the observed 1/2 or documented
  0/1 encodings.
- `data/oscd_original.py` validates official lists, checks the stable-hash
  validation rule, indexes rectified 128 x 128 windows, computes train-only
  band statistics and loads only the requested split.
- The local split uses 11 training regions and validation regions `bordeaux`
  and `pisa`. It excludes `mumbai` because its footprint falls inside local
  Pair A. All 10 official test regions remain intact.
- The generated index has 399 train, 62 validation and 230 test windows.
  Indexing reads test B02 metadata for dimensions, but it does not read test
  image pixels or test-label pixels. The training branch does not construct a
  test dataset.
- `train.py` uses original OSCD by default. `--dataset oscd100` preserves the
  existing smoke-test loader and writes to a separate default output directory;
  local training also uses a separate name. `--dataset local_manual` selects
  the local branch after human labels exist.
- The original-OSCD one-epoch/one-batch-per-split CUDA run completed. It reported
  train change fraction 0.023919 and `pos_weight` 40.8075. Validation AP 0.0070
  came from one validation batch and is a pipeline check, not a result.
- The OSCD100 one-epoch smoke path also completed after the training dispatch
  change. This checks compatibility, not generalisation.

D2.6 partitions:

- `configs/local_partitions.yaml` defines 70/15/15 stripes and 500 m guard bands.
  The selected orientation uses valid-land coverage only. Labels and candidate
  scores do not select the split.
- Pair A and Pair B have separate 10 m and 20 m partition rasters under
  `outputs/partitions/`. The 10 m rasters exactly match 2× nearest-neighbour
  replication of the 20 m codes, including buffer value 0. The output summary
  records valid-land counts and retained fractions. The official OSCD test list
  is unchanged.

D2.5 manual labels and D2.7 local patches:

- `configs/manual_label_schema.yaml` and
  `labels/manual_labels.template.geojson` define human-only review records.
  The rasterizer rejects automated evidence sources, invalid geometry,
  conflicting overlaps, split/buffer crossings, and pixels outside the
  baseline valid-land mask. It writes 0=no change, 1=change, 255=uncertain or
  unreviewed. Its `manual_labels_<pair>_audit.csv` records the label, reviewer,
  date, evidence and confidence for each polygon.
- Candidate GIS layers contain 71 Pair A and 181 Pair B component bounding
  boxes. They are unreviewed candidates, not labels. The candidate boxes come
  from the classical score ranking; no human label decisions have been made.
  No OSM-derived label has been used.
- `data/local_patches.py` builds windows only inside one split and loads
  reflectance patches from the `.SAFE` products on demand. By default, it does
  not read test-label pixels. Its `--include-test` option writes metadata that
  causes the local trainer to reject that index. It preserves a valid-pixel
  mask. `DiceBCELoss` and `run_epoch` ignore pixels marked unknown. The trainer
  supports strict checkpoint initialization with `--init-checkpoint`; without
  it, weights start randomly. `scripts/materialize_local_patches.py` writes
  compressed train/validation patches. Test materialization requires an
  explicit flag.
- Final smoke checks after the masked-loss changes passed for original OSCD,
  OSCD100, local partial-label patches, and checkpoint loading. The checkpoint
  fixture was synthetic; its metrics are not model results. An all-invalid
  batch returns zero loss and NaN ranking metrics. A test-inclusive local index
  is rejected by the trainer.
- Scratch-only tests passed for manual rasterization, automated-source and
  partition-boundary rejection, real SAFE window reading, patch-index split
  safeguards, masked loss, and local training. The synthetic labels remained
  under Hermes scratch.
- The review pass added guards for duplicate OSCD split entries, local index
  split-code and test-row consistency, local training metadata and patch shape,
  and accidental overwrite of training, partition, manual-label,
  materialization, candidate-export and local-index outputs. The OSCD audit
  now checks every rectified raster's identity transform; it reports that
  property instead of calling pixel units a physical resolution. Sliding-window
  inference now includes probabilities
  exactly equal to its threshold, matching the trainer's confusion counts. The
  manual-label guide now requires one GeoJSON per pair. The candidate exporter
  and blank template omit the legacy top-level `crs` member, which RFC 7946
  removed; positions remain WGS84 longitude/latitude. Focused negative tests
  and fresh local/original-OSCD/OSCD100 pipeline smokes passed after these
  changes. All smoke metrics remain pipeline checks.
- Independent audit response (P0, paired observation validity). A new
  `data/validity.py` builds one date's observation-valid mask from source
  NoData, `MSK_CLASSI` and `MSK_QUALIT` on the analysis grid, warps the
  registration coverage mask, and combines them into `paired_validity`.
  `data/masks.py` now fails on a missing or unexpected band count instead of
  padding with zeros. `data/grids.py` holds the grid contract used by every
  consumer. The baseline, the normalisation comparison, the partition
  generator and the manual-label rasterizer all use the paired mask; the
  rasterizer refuses to run without `observation_valid_<PAIR>.tif`.
  `reference_scaled_change_score()` is now the single score definition and
  requires an explicit valid mask. The candidate mask writes 255 for
  ineligible pixels. All baseline outputs go to a new `--output-dir`; nothing
  is replaced.
- Real-data check of the validity builder on both pairs (20 m grid,
  5490 x 5490): Pair A date 1 invalid 0.00 %, date 2 invalid 0.01 %
  (2,331 cirrus pixels at 20 m, 40 saturated pixels in B08), paired valid
  99.99 %. Pair B date 1 invalid 0.00 % (19 saturated), date 2 invalid 0.00 %
  (22 saturated), paired valid 100.00 %. No NoData pixel in B02 or B08 of any
  product. The strict band-count checks passed against all four real products.
- Mask-aware registration. `window_validity()` builds a per-window mask from
  source NoData, `MSK_CLASSI` and `MSK_QUALIT` on the window transform;
  `scripts/run_registration.py` now passes it as `ref_valid`/`mov_valid`, so
  ECC sees an `inputMask`. The re-run is in
  `outputs/audit/registration_v2_observation_valid/registration.json`. The
  masked estimate reproduces the earlier one: Pair A median -0.122/-0.639 px
  against -0.118/-0.639 px before (delta -0.004 px = -0.04 m in x, 0.000 px
  in y); Pair B +0.013/-0.119 px both times (delta 0.000 px). Residual RMSE
  mean fell from 0.6526 to 0.6523 (A) and from 1.0128 to 1.0120 (B). The shift
  sign and the transform convention are unchanged.
- Partitions rebuilt from the corrected score into
  `outputs/partitions_v2_observation_valid`. Pair A keeps 95.7 % of valid land
  outside the guard bands, Pair B 98.2 %. Per-split valid pixels: A
  train 2,809,724 / validation 3,591,773 / test 4,142,947; B train 20,611,963
  / validation 4,226,697 / test 4,328,122.
- Candidate review layers regenerated from the corrected run:
  70 candidates for Pair A and 179 for Pair B (previously 71 and 181), all
  tagged `unreviewed_candidate_components`. The ranking CSVs are in
  `outputs/audit/v2_observation_valid/` and the GeoJSON in
  `outputs/labels/v2_observation_valid/`.
- Water policy resolved. Observation validity and water state are separate.
  The annotation rasterizer reads no water mask and cannot reject a polygon
  because it touches water; a water-to-land transition stays a valid `change`.
  The schema rule that said "must not cover invalid or water pixels" was
  wrong and is replaced. The rasterizer message now says "not valid
  observations".
- Final partition and candidate-recall check (read-only; nothing was retuned).
  Candidate pixels by region, at 20 m: Pair A 11,535 px in total, train 4,423
  (38.3 %), validation 2,343 (20.3 %), test 4,172 (36.2 %), guard band 597
  (5.2 %). Pair B 67,206 px, train 38,214 (56.9 %), validation 14,729
  (21.9 %), test 11,248 (16.7 %), guard band 3,015 (4.5 %). The guard band
  therefore hides about 5 % of the candidate area in both pairs.
- Orientation check. The generator considered four stripes. For Pair A the
  valid-land shares are x low-to-high 25.5/32.6/37.6 (minimum 25.5 %), y
  low-to-high 78.0/12.0/8.6 (8.6 %), x high-to-low 100/0/0 (0 %), y
  high-to-low 62.0/16.7/19.0 (16.7 %). The chosen orientation is the only one
  that leaves a usable validation and test share, because the Pair A land is a
  narrow strip at one edge of a mostly-water tile. Pair B is insensitive to
  the choice: all four orientations give 13.9-14.2 % to the smallest split.
- Morphology sensitivity. At the 99th percentile the raw threshold yields
  110,176 px in 50,103 components for Pair A. A 3x3 opening alone reduces this
  to 10,963 px in 546 components; the 5x5 closing returns it to 11,536 px in
  483 components; the 4-pixel minimum then removes one pixel and one
  component. The minimum-size filter is therefore almost inert and the
  opening does nearly all of the suppression. A 5x5 opening would leave
  1,989 px in 34 components. Of the raw Pair A components, 44,615 of 50,103
  are smaller than 4 px and 485 are elongated with a bounding-box aspect of at
  least 3.
- Candidate blind spot. The score is defined only on land, so 19,122,527
  Pair A pixels (63.4 % of the grid) and 426,171 Pair B pixels (1.4 %) never
  enter the candidate search. That set is water plus the dynamic shoreline,
  which is exactly where a reclamation change would appear. The annotation
  tool accepts those areas, but the baseline cannot propose them. A reviewer
  must inspect the shoreline without a candidate hint.
- Independent registration controls. Five windows that do not overlap the
  estimation windows were measured after applying the stored shift. Pair A
  residual displacement median 0.263 px (2.63 m), p95 0.555 px (5.55 m), max
  0.595 px (5.95 m). Pair B median 0.084 px (0.84 m), p95 0.196 px (1.96 m),
  max 0.211 px (2.11 m). These are larger than the same-window residuals
  (0.130 px for A, 0.056 px for B) and still sub-pixel.
- The failing Pair A estimation window is open water: land fraction 0.0 %,
  B08 standard deviation 0.0023 against 0.045-0.073 in the four successful
  windows. ECC hit its iteration cap on a textureless region. The median shift
  is computed from the four converged windows.
- Overlapping-patch statistics and metrics (audit P1). `utils/unique_metrics.py`
  adds coverage-weighted training statistics and a stitched validation
  evaluator. `train_local_manual.py` now takes the band mean and standard
  deviation, the class balance and every validation metric from unique
  geographic pixels; the patch overlap remains only where it helps training.
  On a synthetic 4x4-patch layout over a 6x6 grid the evaluator reports 36
  unique pixels against 64 patch slots. `tests/test_unique_metrics.py` holds 14
  checks, including one that the old patch-weighted mean differs from the
  unique-pixel mean.
- `scripts/run_normalization.py` ran end to end into
  `outputs/audit/normalization_v2_observation_valid/`. On the corrected basis a
  global linear fit gives the largest stable-land error reduction for both
  pairs (A 39.6 %, B 32.0 %), ahead of histogram matching (A 36.5 %, B 31.5 %).
  This contradicts the comment previously in `config.yaml`; the value is left
  at histogram and the conflict is recorded as an open decision.
- `scripts/predict_safe_pair.py` writes a georeferenced probability, mask and
  validity raster from a `.SAFE` pair in row strips with a halo. A 1024 x 1024
  smoke run on Pair A produced three rasters that share the 10 m grid, CRS
  EPSG:32643 and the tile origin, with 1,047,552 of 1,048,576 pixels valid.
  The checkpoint was randomly initialised, so those numbers are a pipeline
  check and not a detection result; the rasters and the config carry
  `status=pipeline_output_not_a_result`.
- Deliverables prepared: `configs/detection_register_schema.yaml`,
  `docs/detection_register.md`, `labels/detection_register.template.csv`,
  `docs/onboard_architecture.md`, `docs/reproducibility.md`,
  `scripts/make_run_manifest.py`, `docs/report.md` (draft) and `AI_USAGE.md`.
  The register itself is empty by design; no polygon has been reviewed.
- Positive controls for normalisation. `scripts/evaluate_normalization_positive_controls.py`
  scores the 13 OSCD development regions before and after each correction
  against the dataset authors' labelled change, using AUROC and a preservation
  ratio on labelled-change pixels. Output:
  `outputs/audit/normalization_positive_controls_v1/`. Neither method destroys
  change: AUROC improves in 12 of 13 regions for both. Histogram preserves the
  change amplitude better in 13 of 13 regions (median 0.988 against 0.947,
  worst 0.887 against 0.837); global ranks marginally better (median delta
  AUROC +0.034 against +0.029) and reduces stable-land error more. The
  configuration keeps histogram, and `config.yaml` now records both views. The
  OSCD test regions and `mumbai` were not read.
- Georeferenced inference re-run against the masked registration. With
  `data.registration_summary` pointing at
  `registration_v2_observation_valid/registration.json`, the smoke run applied
  shift (-0.122358, -0.639057) px, matching the masked estimate and differing
  from the earlier run's (-0.118042, -0.639199) by 0.004 px. Output:
  `outputs/predictions/smoke_inference_v2_masked_registration/`, still tagged
  `pipeline_output_not_a_result` (the tag this and earlier smoke runs carry; the
  full-tile runs below use `transfer_smoke_not_a_result`).
- Full-tile transfer smoke, both pairs, with the original-OSCD seed 42
  checkpoint. Pair A: 22 row strips, 120,560,400 pixels, 120,540,008 valid
  (99.9831 %), 357.9 s, peak RSS 3,496 MiB, peak GPU 211 MiB. Pair B: 22 strips,
  120,560,299 valid (99.9999 %), 370.7 s, peak RSS 3,511 MiB, peak GPU 210 MiB.
  Both produce zero pixels at or above 0.5; the maximum probability is 0.3711,
  so the fixed threshold is never reached. Grids verified at 10,980 x 10,980,
  EPSG:32643, 10 m, with nodata aligned exactly to the invalid pixels. Outputs
  `outputs/predictions/transfer_smoke_full_tile_{A,B}_v2/`, tagged
  `transfer_smoke_not_a_result`. No register row was created. Repeating Pair A
  reproduced its statistics to six decimals, so the path is deterministic.
- Three-provider validation benchmark: ONNX FP32 CUDA, ONNX INT8 CUDA and ONNX
  INT8 CPU over the 62 validation windows. AP 0.33626 / 0.33148 / 0.26196;
  median 9.16 / 11.80 / 59.96 ms per batch of eight; GPU peak 231 / 233 MiB.
  The same INT8 file differs by 0.070 AP between providers (rank correlation
  0.725), so the earlier 0.0743 AP loss is a CPU-only figure. Output
  `outputs/bench/oscd_validation_providers/`.
- The external code-review workers could not run because the provider returned
  HTTP 401. This is a manual review, not an independent review. The committed
  `tests/test_day2_safety_guards.py` covers split/index/metadata/output guards.
  Full Rasterio-to-SAFE integration fixtures used for smoke checks remain in
  Hermes scratch; they are not a permanent end-to-end test suite.
- No full OSCD training, real checkpoint fine-tuning, user-reviewed local labels,
  project local patch index, or project local patch files exist yet. The user
  must make the annotation calls before those local outputs can be produced.
  D2.5 tooling is ready; the label set itself is not complete.

Sources read for these changes:

- OSCD authors' project page: https://rcdaudt.github.io/oscd/
- TorchGeo OSCD dataset documentation:
  https://docs.torchgeo.org/en/stable/api/datasets/oscd.html
- PyTorch 2.11 data loading documentation:
  https://docs.pytorch.org/docs/2.11/data.html
- PyTorch 2.11 `torch.load` documentation:
  https://docs.pytorch.org/docs/2.11/generated/torch.load.html
- Rasterio vector rasterization and windowed I/O:
  https://rasterio.readthedocs.io/en/stable/topics/features.html
  https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html
- Rasterio coordinate transform used for GeoJSON:
  https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html#rasterio.warp.transform_geom
- IETF GeoJSON standard (RFC 7946):
  https://www.rfc-editor.org/rfc/rfc7946.html

---

## 10. Decision log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-11-13 | Apply RADIO_ADD_OFFSET = -1000 to every band | The ESA formula requires it. |
| 2026-11-13 | Decode MSK_CLASSI as a 3-band mask | The ESA page defines three classification bands. |
| 2026-11-13 | Use OSCD as the primary public dataset | It is Sentinel-2, 13-band. |
| 2026-11-13 | Call the model FC-Siam-diff-inspired | The description matches FC-Siam-diff. |
| 2026-11-13 | Remove the Zynq radiation-tolerance claim | No source supports it. |
| 2026-11-13 | Use three separate water masks | A union mask removes reclaimed land. |
| 2026-11-13 | Treat OSM as candidate context only | OSM is not historical ground truth. |
| 2026-11-13 | Keep native band resolution | Upsampling creates no detail. |
| 2026-11-13 | Use windowed I/O | Full-tile float32 arrays are about 11.7 GiB. |
| 2026-11-13 | Use land PIFs for normalisation | The sea is not invariant. |
| 2026-11-13 | Deadline is 18 November 2026 | Confirmed by the user. |
| 2026-11-13 | Cut scope, not requirements | Quantisation and the report are mandatory. |
| 2026-11-13 | Use the laptop as the measured proxy | No Jetson board is available. |
| 2026-11-13 | Five-day sprint, 13 to 17 November | Confirmed by the user. |
| 2026-11-15 | Do not install TensorRT; do not modify the CUDA/ONNX environment | The runtime library is absent. The existing benchmark already answers the question: CUDA FP32 2.43 ms, CPU FP32 8.76 ms, CPU INT8 9.94 ms with a 0.0743 AP loss on validation. Installing TensorRT would touch a working environment for no change in conclusion. Confirmed by the user. |
| 2026-11-15 | State the standard deviation convention in every table | The trainers write `np.nanstd`, the population standard deviation (ddof=0), under the row label `std_population`. Reports quote the sample standard deviation (ddof=1) where a spread is described. At n = 3 the two differ by a factor of 1.22, so an unlabelled figure is ambiguous. |
| 2026-11-15 | Quantisation AP/AUC are validation-split results | AP 0.3362 to 0.2620 and AUC 0.9025 to 0.8895 are measured on `bordeaux` plus `pisa`, 62 windows. They are separate from the locked official-test evaluation and must not be read as test results. The 0.2620 endpoint came from the defective CPU optimiser path; the corrected CPU value is 0.33198 and the CUDA value is 0.33148. |
| 2026-11-15 | Record the CUDA provider's silent CPU fallback | With no CUDA consumer already in the process, ORT fails to resolve `libcublasLt.so.12` and quietly builds a CPU-only session. `onnxruntime.preload_dlls()` fixes it without importing torch, and `create_session(..., require=...)` in `utils/ort_session.py` raises instead of accepting the fallback. Deployment scripts must assert `session.get_providers()`. |
| 2026-11-15 | CPU INT8 numbers must not use ORT's default graph optimisation | ORT 1.24.4's QDQ selector fuses Conv into QLinearConv at EXTENDED and above. The fused kernel applies `x_scale * w_scale` to the int32 bias, so the ONNX invariant requires the bias to have been quantised with that scale. 9 of 17 fused convs conform; the 8 of the second Siamese branch do not (ratio 0.18 to 1.95), because the two branches share one bias but were quantised with different weight scales. Repairing only those bias quantisations moves the fused path from r 0.678 to r 0.9905. This is upstream issue 24711, fixed in PR 28229 (ORT 1.27). Workaround applied: CPU QDQ sessions pinned to ORT_ENABLE_BASIC, costing about 2.7x CPU latency. Diagnosis in `docs/decisions/009-cpu-qdq-optimiser-defect.md`. |
| 2026-11-13 | Report in markdown to PDF with weasyprint | Matches the sar2eo workflow. |
| 2026-11-13 | Correct the B11 reflectance to 0.0139 | The 20 m band was read with a 10 m window. |
| 2026-11-13 | Do not call the scenes cloud-free | The mask does not cover haze or cloud shadow. |
| 2026-11-13 | Compute NDBI at 20 m | B11 is 20 m. Do not upsample. |
| 2026-11-13 | Stage 1 outputs two classes | Invalid is a validity mask, not a class. |
| 2026-11-13 | Stage 2 is analyst-reviewed | No automatic 13-class classifier is claimed. |
| 2026-11-13 | Use temporal persistence for multi-date merge | An OR rule creates false events. |
| 2026-11-13 | Use boundary sensitivity for area uncertainty | One pixel is not a full uncertainty estimate. |
| 2026-11-13 | Use a skeleton graph for road length | Counting pixels underestimates diagonals. |
| 2026-11-13 | Benchmark CPU INT8 first | CUDA INT8 op support is not established. |
| 2026-11-13 | Report the CPU as 10 cores (6P + 4E), 16 threads | The i7-12650H specification. |
| 2026-11-13 | Remove the Jetson 1 ms latency estimate | No benchmark supports it. |
| 2026-11-13 | Label offline inference as a self-imposed constraint | The assignment does not require it. |
| 2026-11-13 | Add the hyperparameter search to Day 3 | The assignment requires it. |
| 2026-11-13 | Add `configs/config.yaml` to the deliverables | The assignment requires versioned configurations. |
| 2026-11-13 | Split OSCD at the scene level | Prevent leakage. |
| 2026-11-14 | Correct the OSCD split decision: OSCD100 cannot be split by location | All 100 OSCD100 crops carry the same dates.txt (20150801 / 20170801) and the rasters have no CRS. Its 60/20/20 split is a random crop split of one acquisition pair, so validation and test are not independent. |
| 2026-11-14 | Use OSCD100 only as a pipeline smoke test | The dataset card says it is not for benchmarking. A real geographic split needs the original OSCD regions. |
| 2026-11-14 | Make `configs/data_manifest.yaml` the only source of product paths | Nine scripts used local product paths. `data/manifest.py` now expands `SENTINEL_DATA_ROOT` from the environment and every script resolves through the manifest. |
| 2026-11-14 | Build one paired observation-valid mask and use it for every statistic | The L1C quality masks existed but the baseline never applied them, and finite reflectance is not a valid observation. Source: independent audit P0; Sentinel-2 product specification for `MSK_CLASSI` / `MSK_QUALIT`; OpenCV `findTransformECC` `inputMask`. |
| 2026-11-14 | Fail closed on missing or unexpected quality-mask bands | Padding a mask with all-zero rows turns missing information into "valid". |
| 2026-11-14 | One change-score definition with an explicit valid mask | Two definitions existed, and the normalisation script standardised each date independently, which makes the score depend on both scene-wide distributions. |
| 2026-11-14 | Baseline refuses to overwrite and writes to a versioned `--output-dir` | The previous candidates must stay available so a corrected run can be compared against them. |
| 2026-11-14 | Pass quality masks into ECC instead of estimating on every pixel | A flagged pixel is not an observation. The masked estimate is now the stored one; it reproduces the unmasked shift within 0.004 px, so the convention is unchanged. |
| 2026-11-14 | Water is not observation validity | Water answers "what is on the ground", validity answers "is the measurement trustworthy". A water-to-land transition can be real reclamation, so excluding water from annotation would delete a target class. |
| 2026-11-14 | Validation metrics use unique geographic pixels | Patches overlap by 50 %, so a reviewed pixel appeared in up to four validation samples. Metrics are now computed on the stitched grid; training keeps the overlap. |
| 2026-11-14 | Normalisation method kept at histogram, on positive-control evidence | Stable-land error favours a global fit, but a correction can lower that error by shrinking real change. On the 13 OSCD development regions, histogram preserved the labelled change amplitude better in 13 of 13 regions (median 0.988 against 0.947, worst 0.887 against 0.837) with near-equal ranking (median delta AUROC +0.029 against +0.034). Attenuating change is the worse failure for this task. |
| 2026-11-14 | Make `train.py` read `config.yaml` and write `run_config.json` | The checkpoint alone did not record the hyperparameters, so a run could not be reproduced from it. |
| 2026-11-14 | Rename the baseline output to `candidate_mask_*.tif` | A mask marking 32% of land must not be called a change mask. The GeoTIFF carries `exploratory_candidate_mask_not_a_detection`. |
| 2026-11-14 | Fix the B11 window read in `scripts/verify_offset.py` | It read the 20 m B11 band with 10 m row/col numbers, the same error the plan had already corrected. The sea-window B11 mean was 0.0153; it is now 0.0139, matching the plan. The land window now returns values instead of NaN. |
| 2026-11-14 | Record the OSCD licence as CC BY-NC-SA for the labels | The IEEE DataPort page (DOI 10.21227/asqe-7s69) states it directly. The imagery is modified Copernicus data 2015-2018. |
| 2026-11-14 | Study four public reference projects and record what transfers | See `docs/reference_analysis.md`. |
| 2026-11-14 | Adopt selected reference techniques before OSCD loader/label tooling | Metrics, percentile thresholding, AP/AUC selection, training-derived `pos_weight`, three configured seeds, overlap-averaged inference, change-area summaries and paired augmentation were implemented and smoke-tested before the loader work began. |
| 2026-11-14 | Treat the configured 99th-percentile score as an uncalibrated prior | Pair A and B runs write candidate masks only. GeoTIFF tags record the percentile, threshold and `percentile_prior_not_calibrated` status. |
| 2026-11-14 | Implement the original OSCD loader; preserve the official test regions | OSCD100 is a single-pair smoke dataset, while original OSCD provides the region split needed for geographic evaluation. |
| 2026-11-14 | Preserve OSCD TIFF label classes 1=no-change and 2=change | The local audit found this encoding in all 24 masks, although the downloaded README documents 0/1. The loader verifies values before mapping. |
| 2026-11-14 | Exclude Mumbai from local OSCD training and validation | Its footprint overlaps the local Pair A tile T43QBB. |
| 2026-11-14 | Keep local labels human-reviewed and do not derive them from OSM | OSM is current-state context and is not historical ground truth; the user makes each label call. |
| 2026-11-14 | Generate local patches only after labels and partitions | D2.7 windows must remain inside one D2.6 split, and unlabeled pixels must be ignored. |
| 2026-11-14 | Exclude test-label pixels from local index construction by default | The indexer requires an explicit `--include-test` flag; local training rejects an index whose metadata records test-label reads. |
| 2026-11-14 | Refuse to overwrite non-empty generated output directories by default | Protect existing runs, partitions, manual-label rasters and materialized patches; use a new output path for a new run. |
| 2026-11-14 | Reject duplicate OSCD region-list entries and inconsistent local index metadata | Prevent duplicate training windows and metadata claims that do not match the indexed rows. |
| 2026-11-14 | Defer D2.4 OSM until D2.5 and D2.7 are complete | User requested the OSM step only after the manual label set and patches/index. |
| 2026-11-13 | Do not accept Pair A's exploratory threshold mask as candidate sites | The first score marked 34.7% of land and had a 659.2 km² component. After changing to reference-land scaling and reapplying validity after morphology, 32.1% remained positive and the largest component was 116.3 km²; the threshold is still uncalibrated. |
| 2026-11-13 | Save the per-pixel change-score raster | The mask alone is not enough to audit threshold behavior. |
| 2026-11-13 | Scale each band difference by the reference date's land standard deviation | Standardising both dates independently can create or suppress apparent differences; the threshold remains exploratory. |
| 2026-11-13 | Use raster-wide component statistics and reapply validity after morphology | A per-component full-raster scan is too slow, and closing must not mark invalid or water pixels. |
| 2026-11-13 | Do not treat OSM query errors as empty map data | The 20 initial queries returned 2 HTTP 406 and 18 network errors; there were 0 successful replies. |
| 2026-11-13 | Remove Mathura from the Pair B description | It is south of the tile. |

---

## 11. Next action

Ground segment is complete and verified: validity chain, masked registration,
partitions, classical baseline, review layers, georeferenced inference,
evaluation, quantisation, register schema and the audit outputs. Original-OSCD
GPU training has run (3 seeds, 3 tuning trials, one locked-test evaluation).

Blocked on the user: the manual polygons. Everything local depends on them.

Next, in order:

1. Review the annotated GeoJSON, rasterize it, build the local patch index and
   materialise patches. The trainer and the unique-pixel metrics are ready.
2. Fine-tune from the OSCD checkpoint on the local labels, then repeat the
   quantisation comparison, because the final onboard model must be the local
   one.
3. Decide the normalisation method: histogram (change preservation) against
   global (stable-land error). See `docs/report.md` section 8.
4. Publish the repository. Local and remote share no ancestor, so the work must
   go out on a new branch from the published root. The safe procedure and its
   pre-flight checks are in `docs/publication_plan.md`; the route and the licence
   choice are the owner's decision.
