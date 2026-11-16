# Verification pass — evaluation, area filtering, metric reconciliation

Answers five questions with measurements, not assertions. Everything here is
validation-split work on original OSCD. The official test split was not re-read.

Artefacts: `results/verification.json`, `results/across_seeds.csv`,
`results/across_seeds_summary.json`, `results/*/thresholds.csv`, `results/comparison.csv`.

---

## 1. Outstanding runs, and across-seed comparison

### 1.1 The three retrains that were in flight when this pass started

All three finished normally (exit 0) before this verification began.

| Run | Architecture | Loss | Seed | Epochs | Batch | Peak VRAM | Train time |
|---|---|---|---|---|---|---|---|
| `loss_bce_seed42` | SiameseUNet, base 16 (0.48 M) | BCE only | 42 | 20 | 8 | 0.17 GiB | 11.1 s/epoch |
| `loss_focal_seed42` | SiameseUNet, base 16 | focal, gamma 2 | 42 | 20 | 8 | 0.17 GiB | 12.0 s/epoch |
| `arch_resnet18_seed42` | ResNet-18 encoder (14.23 M) | BCE + Dice | 42 | 20 | 8 | 0.35 GiB | 6.5 s/epoch |

Every run printed `official test labels were not read`. Training time is not a
model comparison: the job is data-bound at 10-20 % GPU utilisation and the epoch
time tracked machine load.

### 1.2 Across seeds

Scored identically for every run: scene-level average precision, then pixel and
object metrics at a **fixed** operating point of threshold 0.90 and 64-pixel
minimum component, so the seeds are comparable rather than each flattered by its
own threshold choice.

| Run | Seed | AP | Pixel F1 | Object F1 | Object TP / FP / FN |
|---|---|---|---|---|---|
| Siamese U-Net (baseline) | 42 | 0.2806 | 0.3862 | 0.2124 | 12 / 24 / 65 |
| Siamese U-Net (baseline) | 43 | 0.2168 | 0.2839 | 0.1085 | 7 / 45 / 70 |
| Siamese U-Net (baseline) | 44 | 0.2149 | 0.3094 | 0.1515 | 10 / 45 / 67 |
| **Siamese U-Net, mean ± SD** | 3 | **0.2375 ± 0.0374** | 0.3265 ± 0.0536 | **0.1575 ± 0.0522** | — |
| ResNet-18 encoder | 42 | 0.3067 | 0.3521 | 0.0839 | 6 / 60 / 71 |
| ResNet-18 encoder | 43 | 0.2179 | 0.2878 | 0.0656 | 4 / 41 / 73 |
| ResNet-18 encoder | 44 | 0.2420 | 0.3195 | 0.1274 | 10 / 70 / 67 |
| **ResNet-18, mean ± SD** | 3 | **0.2555 ± 0.0459** | 0.3198 ± 0.0326 | **0.0923 ± 0.0317** | — |

Single-seed references at the same operating point: BCE-only AP 0.2666, object
F1 0.1644; focal AP 0.2580, object F1 0.0250.

**Reading, and the trap in it.** Seed 42 is the best seed for *both* architectures
— 0.2806 and 0.3067 AP against means of 0.2375 and 0.2555. Reporting seed 42 for
either would overstate expected performance by 0.04-0.05 AP and by 0.04 object F1.
The across-seed means are the usable numbers.

On the means, ResNet-18 leads on AP by 0.018 and trails on object F1 by 0.065.
Neither is established at n = 3: the AP gap is 0.4 of a pooled standard
deviation, and the object-F1 gap is about 1.5. What can be said is directional —
the pretrained encoder does not pay for its 30× parameter count on this dataset,
and its object-level behaviour is consistently worse.

Focal at the fixed 0.90 operating point collapses (1 true object), because its
own optimum is 0.75. That is a property of comparing at a fixed threshold; at its
own threshold it reached object F1 0.19 after filtering. Its AP, which is
threshold-free, is still the lowest of the four.

The spread matters more than the ordering. Both families have a between-seed
standard deviation of roughly 0.02-0.05 AP, which is the same size as the gap
between them. **No seed is selected as the expected performance**; the table
reports mean and sample standard deviation over all three seeds.

---

## 2. Scene-level evaluation, verified

### 2.1 Overlapping patches are reconstructed into one scene

`collect_predictions.py` writes each window into a per-city canvas at its own
row and column offset. Two quantities matter, and both are measured in
`results/verification.json`:

| City | Canvas | Windows | Pixels never written | Pixels written twice |
|---|---|---|---|---|
| bordeaux | 517 x 461 | 20 | 0 | 70,524 (29.6 %) |
| pisa | 776 x 718 | 42 | 0 | 112,960 (20.3 %) |

Every canvas pixel is written at least once, and a fifth to a third is written
twice. The overlap is large because `patch_origins` appends the edge origin after
the last stride-aligned one: for a 517-row canvas the origins are
0, 128, 256, 384, **389**, so the final window overlaps its predecessor by 123 of
its 128 rows.

**Reconstruction rule: last writer wins.** The windows are written in index
order, so a pixel covered twice holds the value from the later window. This was
verified by re-running 8 edge windows for bordeaux and 12 for pisa and comparing
the canvas against the re-run prediction wherever no later window covers it:
maximum absolute difference **3.07e-04** (bordeaux) and **2.43e-04** (pisa)
in probability units. The residual is not zero because the collect pass ran
batches of eight and the verification ran single windows, and the CUDA kernels
are not bit-identical across batch shapes. It is three orders of magnitude below
the differences the evaluation measures.

An alternative rule — averaging the overlapping windows — would change at most
those second-write pixels. It was not implemented, because the two windows agree
to 3e-4 and averaging would not move any metric at the reported precision.

### 2.2 Components are extracted after reconstruction

`evaluate_patch` is called once per **full-scene** array, not per window. The
scene-level numbers in this pass therefore label components on the reconstructed
canvas.

### 2.3 Ground-truth objects are not duplicated or split

This is the direct check. Counting connected components of the truth mask on the
full scene, then summing the per-window counts using the baseline's own framing:

| City | Truth objects, scene | Truth objects, per patch | Ratio |
|---|---|---|---|
| bordeaux | 26 | 40 | 1.54x |
| pisa | 51 | 88 | 1.73x |
| total | **77** | **128** | **1.66x** |

Scene-level extraction yields 77 objects. Patch framing yields 128 from the same
labels, because an object crossing a patch edge becomes two, and the overlapping
edge patches count some objects twice. That ratio — not the model — accounts for
a large part of the historical object counts.

### 2.4 Conventions, stated exactly

| Item | Convention |
|---|---|
| Connectivity | **4-connectivity** — `scipy.ndimage.label` default, cross-shaped structure. Two change areas touching only at a corner are two objects. Verified with a two-pixel diagonal test: 2 components. |
| Matching | one-to-one, greedy, highest IoU first |
| Match threshold | IoU >= 0.5 |
| Unmatched prediction | false positive |
| Unmatched truth | false negative |
| Framing | scene-level in this pass; per 128 px patch in the historical run |

An earlier version of `docs/report.md` described the object convention as
8-connected. That was wrong and is corrected: the evaluator uses the default
4-connectivity.

### 2.5 Full counts at the recommended operating point

Baseline checkpoint, validation, threshold 0.90, 64-pixel minimum area:

| Level | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Object | 12 | 24 | 65 | 0.333 | 0.156 | 0.212 |
| Pixel | 3,596 | 3,515 | 7,914 | 0.506 | 0.312 | 0.386 |

Pixel counts are over both cities. For reference, without the 64 px filter at the
same threshold the pixel counts are 4,137 / 5,814 / 7,373, giving precision
0.416, recall 0.359, F1 0.386 — the filter trades 541 true pixels for 2,299 false
ones, which leaves pixel F1 unchanged at three decimals and object F1 much
improved. See `results/baseline_seed42/thresholds.csv` and `postprocess.csv` for
every threshold tested.

---

## 3. Component-area filtering, in physical units

### 3.1 What each threshold means on the ground

| min area (px) | at 20 m (OSCD) | at 10 m (SkyServe grid) |
|---|---|---|
| 4 | 1,600 m² | 400 m² |
| 8 | 3,200 m² | 800 m² |
| 16 | 6,400 m² | 1,600 m² |
| 32 | 12,800 m² | 3,200 m² |
| **64** | **25,600 m² (0.0256 km²)** | **6,400 m² (0.0064 km²)** |
| 128 | 51,200 m² | 12,800 m² |
| 256 | 102,400 m² | 25,600 m² |
| 512 | 204,800 m² | 51,200 m² |

**64 pixels is not a portable setting.** It is 25,600 m² on the OSCD grid and
6,400 m² on a 10 m grid — a factor of four apart in area. The same physical
filter on SkyServe 10 m output would be 256 px; the same pixel count would filter
a quarter of the area.

### 3.2 Effect on detections, and on genuine objects

Baseline checkpoint, validation, threshold 0.90, scene framing.

| min px | m² @20 m | TP | FP | FN | P | R | F1 | Truth objects smaller than the filter |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 16 | 175 | 61 | 0.084 | 0.208 | 0.119 | 0 |
| 4 | 1,600 | 16 | 140 | 61 | 0.103 | 0.208 | 0.137 | 0 |
| 8 | 3,200 | 16 | 117 | 61 | 0.120 | 0.208 | 0.152 | 0 |
| 16 | 6,400 | 16 | 88 | 61 | 0.154 | 0.208 | 0.177 | 4 |
| 32 | 12,800 | 14 | 55 | 63 | 0.203 | 0.182 | 0.192 | 18 |
| **64** | **25,600** | **12** | **24** | **65** | **0.333** | **0.156** | **0.212** | **34** |
| 128 | 51,200 | 6 | 10 | 71 | 0.375 | 0.078 | 0.129 | 53 |
| 192 | 76,800 | 2 | 9 | 75 | 0.182 | 0.026 | 0.045 | 61 |
| 256 | 102,400 | 1 | 6 | 76 | 0.143 | 0.013 | 0.024 | 66 |
| 512 | 204,800 | 1 | 2 | 76 | 0.333 | 0.013 | 0.025 | 73 |

**The F1 peak at 64 px is real, but it is bought by declining to detect small
objects.** Of 77 true objects, 34 are smaller than 25,600 m² and cannot be
recovered at all once the filter is applied; recall falls from 0.208 to 0.156
while precision rises from 0.084 to 0.333. If small-buildings detection matters —
and for the SkyServe target it does — 64 px at 20 m is too aggressive, and the
honest trade is 16 px (6,400 m²), where recall is unchanged at 0.208 and object
F1 is 0.177 rather than 0.212.

### 3.3 Raw outputs are preserved

Every run keeps its stitched probability canvas as `.npz`
(`experiments/oscd_improvement/predictions/*.npz`), plus the per-threshold CSVs.
Post-processing can be re-chosen without re-running a model. Nothing is baked in.

---

## 4. Reconciling the historical metrics

Two measurement sets exist. Neither is wrong; they answer different questions.

| | Historical | This pass |
|---|---|---|
| Checkpoint | `outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt` | same |
| Split | **test** — the 10 official regions, 230 windows | **validation** — bordeaux + pisa, 62 windows |
| Threshold | 0.5 (from `config.yaml`) | 0.90 (selected on validation) |
| Object framing | **per 128 px patch** | **scene-level** |
| Post-processing | none | 64 px minimum area |
| Implementation | `eval.py`, `evaluate_patch` per patch | same function, called on full-scene arrays |
| AP | 0.5068 | 0.2806 |
| Pixel P / R / F1 | 0.2446 / 0.7618 / 0.3703 | 0.333 / 0.232 / 0.273 |
| Object P / R / F1 | 0.0361 / 0.1240 / 0.0559 | 0.333 / 0.156 / 0.212 |
| Object TP / FP / FN | 126 / 3,368 / 890 | 12 / 24 / 65 |

Why they differ, in order of size:

1. **Split.** The historical numbers are on the 10 official test regions, which
   are not the same scenes as bordeaux and pisa. Nothing here isolates that
   difference, and re-reading the test split to measure it would spend the one
   held-out set on a diagnostic.
2. **Framing.** Measured on validation, patch framing inflates AP from 0.2806 to
   0.3326 (+0.052) and truth object counts from 77 to 128. The historical AP
   carries the same inflation.
3. **Threshold.** 0.5 against 0.90 moves pixel F1 by more than a factor of two
   and object F1 by an order of magnitude on the same data.
4. **Post-processing.** The 64 px filter is why the new object precision (0.333)
   is an order of magnitude above the historical 0.0361.

**Consequence for reading either set:** the historical test metrics remain the
only test numbers this project has, and they are patch-framed, at threshold 0.5,
with no filtering. They must not be compared with the validation numbers, and the
"3,368 false objects" figure is inflated by framing as well as by the operating
point. Nothing historical was overwritten; `outputs/eval/` is untouched.

---

## 5. Final recommendation

**Recommended OSCD checkpoint:** `outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt`.
It is the seed-42 baseline model, kept for continuity with every measurement
taken so far. It is *not* representative: seed 42 is the best of three, and the
expected performance of this configuration is the mean, AP 0.2375 ± 0.0374 with
object F1 0.1575 ± 0.0522 at the fixed operating point. Quote the mean, not the
checkpoint.

**Recommended architecture:** the existing Siamese U-Net, base 16, 0.48 M
parameters. ResNet-18's +0.018 mean AP is inside the seed noise and costs 30×
the parameters, 2× the VRAM (0.35 against 0.17 GiB) and half the object F1.

**Recommended loss:** BCE + Dice, `bce_weight` 1.0, `dice_weight` 1.0,
`pos_weight` 40.81. BCE-only and focal were not better, and both were measured on
a single seed.

**Validation-selected threshold:** 0.90. Selected on bordeaux + pisa by pixel F1;
object F1 peaks at the same value. This threshold is a consequence of
`pos_weight` 40.81 inflating every probability — it is not a calibrated operating
point and should not be assumed to transfer to the SkyServe tiles.

**Recommended OSCD post-processing:** connected-component minimum area of
**64 px at 20 m** (25,600 m²) if only large structures are in scope, giving
object F1 0.212 on seed 42. If small buildings matter, **16 px** (6,400 m²),
which preserves recall 0.208 at object F1 0.177. No morphology: opening and
closing added nothing beyond the area filter.

**Results across seeds:** section 1.2 — three seeds per architecture, scored at
one fixed operating point, reported as mean ± sample SD.

**For the SkyServe grid, do not reuse the pixel count.** At 10 m, 64 px is
6,400 m², a quarter of the area it removes on the 20 m OSCD grid. Choose the
filter from the physical size of the target objects, then convert: 16 px at 10 m
is 1,600 m², 64 px is 6,400 m², 256 px is 25,600 m². Re-select on local
annotations, not on this table.

**Known limitations:** section 6 below.

---

## 6. Limitations and unresolved issues

1. **One validation split.** bordeaux and pisa only. Every threshold and every
   post-processing parameter was chosen on 77 true objects. That is a small
   sample, and the F1 optimum will move on other data.
2. **Three seeds, one architecture each.** The seed spread is measured for the
   two architectures, not for the losses; the loss comparison remains single-seed
   and therefore inconclusive.
3. **No SkyServe validation exists.** The manual annotations are still being
   prepared. Nothing here has been tested on the target tiles, and the 10 m
   conversion in section 3 is arithmetic, not measurement.
4. **Framing defect remains in `eval.py`.** The historical evaluator still counts
   objects per patch. It was not changed, so that published numbers stay
   reproducible. A corrected evaluator exists only in this experiment directory.
5. **Overlap rule is last-writer-wins.** Averaging was not implemented; the two
   windows agree to 3e-4, so it should not matter, but it is unverified at larger
   disagreements.
6. **Class imbalance handling is unresolved.** `pos_weight` 40.81 inflates every
   probability, which is why the tuned threshold is 0.90 rather than 0.5. A
   calibration step, rather than a hand-picked threshold, would be the principled
   fix and was not attempted.

---

## 7. Overlapping-window aggregation

The saved canvases hold only the last-writer-wins result, so this re-ran the
model once over the 62 validation windows and kept every window prediction, then
built three canvases from the same per-window outputs:

| Rule | Definition |
|---|---|
| `last` | current implementation: the later window in index order overwrites the earlier |
| `mean` | arithmetic mean of every window covering the pixel |
| `blend` | weighted mean with a raised-cosine taper, 1.0 at the patch centre and 0.1 at the border, so the centre of a patch dominates |

Identical scene reconstruction, identical object matching, identical operating
point (threshold 0.90, 64 px minimum area).

| Method | AP | AUC | Pixel P | Pixel R | Pixel F1 | Object P | Object R | Object F1 |
|---|---|---|---|---|---|---|---|---|
| `last` | 0.2806 | 0.8818 | 0.506 | 0.312 | 0.386 | 0.333 | 0.156 | 0.212 |
| `mean` | **0.2834** | 0.8818 | **0.513** | **0.321** | **0.395** | **0.361** | **0.169** | **0.230** |
| `blend` | 0.2822 | 0.8824 | 0.510 | 0.316 | 0.390 | 0.343 | 0.156 | 0.214 |

Full counts: object TP/FP/FN are 12/24/65 (`last`), 13/23/64 (`mean`) and
12/23/65 (`blend`). Pixel TP/FP/FN are 3,596/3,515/7,914, 3,695/3,507/7,815 and
3,635/3,496/7,875.

### Differences inside the overlap regions

| City | Overlap pixels | last vs mean MAE | last vs blend MAE | mean vs blend MAE | Flips last→mean | Flips last→blend |
|---|---|---|---|---|---|---|
| bordeaux | 76,797 | 0.0264 | 0.0253 | 0.0081 | 196 | 187 |
| pisa | 118,960 | 0.0127 | 0.0131 | 0.0060 | 153 | 174 |

"Flips" counts pixels whose detection state at threshold 0.90 differs between
rules — one flip is a pixel appearing or vanishing from the mask.

The two windows covering the same pixel disagreed by up to **0.59** (bordeaux, 11
overlapping pairs) and **0.50** (pisa, 15 pairs) in probability, before any
aggregation. That is the root of it: a patch border has less context, so the same
pixel is scored differently depending on which window it lands in, and which of
those two values survives is decided by write order under the current rule.

### Verdict: not materially different, so the current rule stays

Every metric moves by less than the between-seed spread of the same checkpoint
(AP SD 0.037, object F1 SD 0.052). Mean aggregation is nominally better on all
eight metrics — AP +0.003, pixel F1 +0.009, object F1 +0.018 — but those are a
fraction of the noise. **The current last-writer-wins implementation is retained.**

Its limitation, documented rather than fixed: about **0.13-0.26 % of overlap
pixels, 349 pixels in total, change detection state depending purely on write
order**. For a change map that is negligible. For a register of discrete
detections it is a small arbitrariness — an object near a patch border can
appear or vanish because of an implementation detail — and if the inference path
is ever reworked, `mean` is the better default: it uses every window's evidence,
has no arbitrary winner, and costs one extra accumulator.

One correction in reporting this: the first run of this comparison produced a
nonsense `mean` result (AP 0.052, recall 0.006) because the unweighted mean was
divided by a kernel-weighted accumulator. The numbers above are from the
corrected run; the bug was caught by the collapse in recall, not by a test.

---

## 8. Closing the OSCD experiment phase

This closes the OSCD improvement work. What it established:

- The compact Siamese U-Net with BCE + Dice remains the baseline.
- The evaluation, not the model, was the largest source of misleading numbers:
  patch framing inflated AP by about 0.05 and object counts by 1.66x.
- The operating point mattered more than any architectural change: threshold
  0.90 and a 64 px minimum area took object F1 from 0.012 to 0.212 on seed 42.
- Neither alternative loss nor the pretrained encoder beat the baseline beyond
  seed noise.
- Scene aggregation is a solved non-issue at the precision this project measures.

Next: integration with the manually annotated SkyServe data. Nothing in this
phase transfers automatically — the threshold, the area filter and the expected
performance all need re-measurement on the target tiles.
