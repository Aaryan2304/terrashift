# Reference project analysis

Date: 2026-11-14

Purpose: record what four public projects do, what is transferable to this
project, and what is not. Nothing here is copied. Every factual claim about an
external project is labelled VERIFIED (read the code or the primary page) or
UNVERIFIED.

## 0. A licence fact these projects helped settle

The OSCD licence was an open item. It is now settled from the primary source.

IEEE DataPort, OSCD page, DOI 10.21227/asqe-7s69, Instructions section:

> Copyright: Sentinel Images: This dataset contains modified Copernicus data
> from 2015-2018. Original Copernicus Sentinel Data available from the European
> Space Agency. Change labels: Change maps are released under
> Creative-Commons BY-NC-SA. For commercial purposes, please contact the authors.

So:

- Imagery: modified Copernicus Sentinel data 2015-2018, ESA.
- Labels: CC BY-NC-SA. Non-commercial, share-alike.

The same page lists the file sizes: images 488.96 MB, train labels 134.64 KB,
test labels 81.65 KB. These match the measured HTTP Content-Length values
(512,716,711 / 137,873 / 83,614 bytes) on the HuggingFace mirrors.

The same page also confirms the dataset layout: `imgs_1_rect` and `imgs_2_rect`
are the `imgs_1` and `imgs_2` images resampled to 10 m, and the official split
is in `train.txt` and `test.txt`.

Status: VERIFIED. Source: https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection

## 1. mjsushanth/oscd-sentinel2-change-detection

https://github.com/mjsushanth/oscd-sentinel2-change-detection

The findings and gaps below describe what the reference project showed. Current
implementation status is in section 5; use that section where it differs from
the pre-adoption assessment.

What it is: classical baselines plus a ~269k-parameter Siamese U-Net on the
ORIGINAL OSCD (24 pairs, 14 train cities / 10 test cities), run on Apple
Silicon. 135 tests, of which 130 pass and 5 are deliberate `xfail` tripwires.

Headline: classical floor F1 0.237, Siamese U-Net F1 0.378 (1.60x), on the 10
held-out cities, every threshold chosen without the test split.

### Transferable, and worth adopting

1. **Split by whole pair or city, never by tile.** Their measured
   `P(neighbour changed | changed)` is 0.765-0.933 against base rates of
   0.3-3.8%, an autocorrelation lift of 24x-270x. With overlapping tiles a
   random tile split puts the same pixels on both sides. This is the strongest
   argument in the document, and it confirms what we found independently in
   OSCD100: 100 crops from one acquisition pair, no CRS, a random crop split.

2. **Percentile thresholds, not absolute cutoffs.** Their train split averages
   2.30% changed pixels and their test split 5.17%, a 2.25x shift. A threshold
   tuned on train cost 24.2% of achievable F1 on test. Our baseline uses an
   absolute 2.0 on a score whose units are arbitrary. A percentile is invariant
   to score units, but it sets a scene-relative prior and does not transfer a
   calibrated operating point. Do not tune it on test data.

3. **AUC alongside F1 and IoU.** AUC judges the score alone; F1 and IoU judge
   the score and the threshold. They recorded a pair with AUC 0.939 and F1
   0.087. We compute no AUC.

4. **Weighted BCE plus soft Dice.** `pos_weight = (1-p)/p` equalises the class
   contribution, and Dice adds a scale-invariant region objective. Our
   `DiceBCELoss` uses plain BCE with no `pos_weight`. At OSCD100's measured
   4.85% change rate, `pos_weight` would be about 19.6. Their side-effect note
   matters: a large `pos_weight` inflates the probabilities, so 0.5 is not a
   meaningful threshold.

5. **Select the best epoch on average precision, not F1 at 0.5.** They measured
   val F1 0.253 at threshold 0.5 versus 0.458 at the tuned threshold. We
   currently select on F1 at 0.5.

6. **Report per-pair spread, not a mean.** Their baseline per-pair F1 std is
   +/-0.150 on a mean of 0.237, which is 63% of the mean. Their seed-to-seed
   range (0.066 F1) is about half the model's entire improvement over the
   baseline, so a single run is "close to meaningless".

7. **A label-free radiometric offset estimate.** `median(date2) - median(date1)`
   per band, using no labels. They measured that it matches the
   ground-truth-informed estimate to within 23 DN (correlation 0.99982) and
   raised mean AUC from 0.660 to 0.713. This is a simpler and cheaper
   alternative to the histogram matching we use.

8. **A log band-ratio scorer.** Under a multiplicative illumination change,
   differencing gives a nuisance proportional to brightness, while a log ratio
   makes it a constant that one threshold can remove scene-wide. Our baseline
   uses a plain absolute difference.

9. **The epistemic conventions.** VERIFIED / UNVERIFIED on every claim, and "a
   number is not a fact until it names its arm" (its configuration and its seed
   count). Their landmine list records that four of their own published claims
   were overturned by their own later work, and they kept the failures.

### Not transferable

- Apple Silicon / MPS. We are on CUDA with 4 GB.
- Their tiling and stitching workflow. We work on whole images and on 256 x 256
  pre-cropped OSCD100 samples.
- Their quantification estimators (Adjusted Count, Median Sweep). Interesting,
  but they report three failed attempts and call the validation split the
  binding constraint. Not worth the time for us.
- Their cloud heuristic. It never fired, then found sand. Abandoned.

### Defects in our code that this project exposed

- Our `ndwi` returns 0.0 where the denominator is zero. Their landmine 4 is
  exactly this: 0.0 is a legitimate index value meaning "the two bands are
  equal", so returning it for 0/0 makes invalid pixels indistinguishable from
  real measurements. Undefined results should be NaN.
- Our `metrics_from_counts` returns 0.0 for degenerate cases (nothing predicted,
  nothing changed). Their metrics return NaN, because a measurement that does
  not exist is not 0.
- Our baseline threshold is an absolute number. A percentile is invariant to
  score units, but it sets a scene-relative prior and does not transfer a
  calibrated operating point.

## 2. etienne912/sentinel-2-data-processing

https://github.com/etienne912/sentinel-2-data-processing

What it is: an AOI-driven pipeline that finds the best cloud-free Sentinel-2
L2A images for two time windows through a STAC API, crops the AOI at 10 m,
masks clouds with the Scene Classification Layer, normalises to [0, 1], tiles
to a fixed patch size, and writes a manifest.

### Transferable

1. **The tile grid builder.** `build_tile_grid` projects the AOI into the
   product CRS, snaps the origin to the stride grid so tiles align, and emits
   one `TileSpec` per tile with its bounds and affine transform. Cleaner than
   what we would otherwise write for `make_patches.py`.

2. **The manifest.** A JSON file linking the T1 and T2 source product IDs, the
   AOI as GeoJSON, the band list, and the tile paths. Our
   `configs/data_manifest.yaml` covers the product level; this is the patch
   level. The pairing is by tile grid position, which is the right key.

3. **Per-band GeoTIFF tags.** Each band carries `band_name`, `source_item` and
   `acquisition_date`, with NaN as nodata. We already write some tags; this is
   a good pattern to extend.

4. **The annotation workflow section.** Select samples with a high probability
   of change so annotators do not see mostly "no change"; use inter-annotator
   agreement on a shared subset; provide a "low confidence" label option rather
   than forcing a guess; attach the patch ID and metadata to every task. This
   is directly relevant to our label set, which does not exist yet.

### Not transferable

- **L2A only.** The pipeline depends on the SCL band. Our products are L1C,
  which has no SCL; L1C provides MSK_CLASSI / MSK_QUALIT / MSK_DETFOO in
  QI_DATA instead. Their invalid-class set `{0,1,3,8,9,10,11}` is an L2A SCL
  encoding and does not apply to our masks.
- **No co-registration.** Their design notes say they assume L2A processing is
  sufficient. We measured a real 1-6 m shift between our dates and correct it.
- **Small-AOI assumption.** They assume an AOI under 1 km that falls in one
  MGRS tile. Our tiles are 110 km.
- **Only 3 bands in the worked example.** We need the four 10 m bands.

## 3. riku-1825/AI-Based-Geospatial-Change-Detection-for-Intelligence-Analysis

https://github.com/riku-1825/AI-Based-Geospatial-Change-Detection-for-Intelligence-Analysis

What it is: a Siamese U-Net with a shared ImageNet-pretrained ResNet-18 or
ResNet-34 encoder, a U-Net decoder over multi-scale absolute feature
differences, trained on LEVIR-CD.

### Transferable

1. **Sliding-window inference with overlap averaging.** `sliding_window_predict`
   pads small images and averages overlapping patches over large ones. Our
   Sentinel-2 tiles are 10980 x 10980, so any patch-based model needs this.
   We do not have it yet.

2. **The changed-area summary.** `changed_area_stats` returns changed pixels,
   total pixels, changed percent and unchanged percent. The README's example
   output is `Changed Area: 14.7% | Unchanged Area: 85.3%`. Our assignment
   asks for a quantitative change summary; this is the shape of it.

3. **A folder summary CSV.** One row per pair with changed/unchanged percent
   and pixel counts, plus a mean across the folder. Cheap and useful.

4. **Albumentations with `additional_targets`.** Geometric transforms are
   applied identically to both dates and the mask, so they stay aligned. We
   have no augmentation at all.

5. **A pretrained ResNet encoder is a real option.** Our `config.yaml`
   originally claimed `encoder: resnet18, pretrained: true` and I changed it to
   `custom_cnn, pretrained: false` because the code did not match. Repo 3 shows
   the claim was not absurd: a shared pretrained ResNet with multi-scale
   feature differencing is a standard, compact design. With only 60 training
   scenes, pretrained features may beat a from-scratch CNN. The cost is that
   ImageNet weights expect 3 channels, so the first convolution must be adapted
   for 4 bands.

### Not transferable

- **LEVIR-CD.** 0.5 m RGB, 3 bands. Not Sentinel-2. Their own README says the
  task is framed on LEVIR-CD, WHU-CD and DSIFN-CD.
- **Batch size 32 and 24 workers.** We have 4 GB of VRAM and 16 GB of RAM.
- **Overall accuracy in the reported metrics.** Their `evaluate.py` prints it.
  At a 4.85% change rate a do-nothing detector scores 95.2% accuracy with zero
  recall. We should not report it.
- **No co-registration and no radiometric normalisation** beyond ImageNet
  scaling. We need both.

## 4. Kaggle: Training/Evaluation of SpaceNet 7 Change Detection

https://www.kaggle.com/code/amerii/training-evaluation-of-spacenet-7-change-detection

What it is: a chip-level change classifier on SpaceNet 7. A ResNet-50 takes a
stacked 6 x 64 x 64 chip and classifies it as change or no change, with
`nn.CrossEntropyLoss`. Best val F1 0.495. Runtime 6 h 34 m on a P100.

### Transferable

1. **The split key.** They split by `im_name`, the location folder, so the same
   location never appears in two splits. Their stated reason is exactly our
   reason: "By having the same location in the same set, we ensure that we are
   avoiding data leakage." Same principle as repo 1.

2. **A balancing sampler.** `BalancingSampler` oversamples change chips to a
   target fraction of each batch. Their change rate is 17.6%. An alternative to
   `pos_weight` for the class imbalance.

3. **Per-epoch precision, recall and F1 with a classification report.** Simple
   and sufficient for monitoring.

4. **`ReduceLROnPlateau`.** A reasonable scheduler. Our config says
   `scheduler: "none"` because it is not implemented.

### Not transferable

- **Chip-level classification, not pixel-level segmentation.** The output is one
  label per chip. Our assignment needs a change mask and localised change.
- **`nn.CrossEntropyLoss` on two classes.** We need pixel-wise BCE with logits.
- **Batch size 512 on a P100.** Not possible on 4 GB.
- **The Slack monitoring bot.** Irrelevant here.
- **`torch.save(model, ...)` of a whole model.** Not portable; a state dict and
  a config are better. We already do the right thing.

## 5. Adoption status after implementation

The recommendations below are now checked against the current working tree and
actual execution. A smoke test confirms the pipeline path only; it does not
establish model quality or justify a benchmark claim.

| Recommendation | Status | Evidence |
|---|---|---|
| Undefined NDWI values are NaN, not zero | Adopted | `preprocess/indices.py`; `python -m preprocess.indices` passes zero-denominator, valid-zero and uint16 checks. |
| Degenerate precision/recall/F1/IoU are not silently set to zero | Adopted | `utils/metrics.py`; self-check passes; undefined denominators return NaN. |
| Use a percentile threshold and mark it uncalibrated | Adopted | `scripts/run_classical_baseline.py` reads `change_score_percentile` from `config.yaml`; Pair A/B runs used 99.0 and GeoTIFF tags say `percentile_prior_not_calibrated`. |
| Report AP and AUC; select epoch by AP | Adopted | `train.py`; AP/AUC self-checks pass; checkpoint comparison uses validation AP. |
| Use training-derived `pos_weight` | Adopted | `data/dataset.py`, `models/siamese_unet.py`, `train.py`; CUDA smoke run computed weight from train masks and completed. |
| Repeat at least three seeds and report spread | Adopted | Configured seeds 42/43/44 are now the default; 1-epoch, 1-batch-per-split smoke run produced all three rows plus mean/std for AP, AUC, F1 and IoU. This is not a performance result. |
| Sliding-window inference with overlap averaging | Adopted | `utils/inference.py`; synthetic self-check passed odd image edges, overlap, small inputs, and rejects stride > patch to prevent gaps. |
| Changed-area and folder CSV summary | Adopted | `utils/metrics.py`, `utils/inference.py`, `scripts/predict_folder.py`; two synthetic four-channel pairs produced masks, figures and a CSV. |
| Paired geometric augmentation | Adopted | `data/augment.py`; 16 flip/rotation combinations preserve exact date and mask alignment; enabled for training only and recorded in run config. |

## 6. Limits of this implementation

- The current dataset class still reads OSCD100. It is suitable for pipeline
  smoke tests only. No original-OSCD loader or label tooling was added in this
  adoption step.
- Folder inference accepts matching array files. The synthetic run used CHW
  `.npy` arrays. It does not yet read the four local `.SAFE` products or write
  georeferenced prediction rasters.
- The baseline percentile encodes a prior, not a calibrated operating point.
  The candidate masks are not detections and have not been confirmed against
  independent labels.
- The three-seed run used one training batch and one validation batch per seed.
  Its metrics only verify control flow and output recording.
- A pretrained encoder, log-ratio scoring and label-free median radiometric
  correction remain research options. This adoption pass did not implement
  them, because they need separate evaluation and decision work.

## 7. Not recommended

- Repo 1's quantification estimators. Three failed attempts, and the author
  calls the validation split the binding constraint.
- Repo 2's SCL masking. L2A only; our data is L1C.
- Repo 3's and the Kaggle notebook's overall accuracy. Misleading under this
  class imbalance.
- Chip-level classification. The task needs a pixel mask and a location.
