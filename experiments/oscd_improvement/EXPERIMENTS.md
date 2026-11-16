# OSCD Siamese U-Net — improvement experiments

**Status: complete.** Phases 1, 2A, 2B, 2D and 2E are done. Phase 2C was
deliberately not run; the reasoning is in section 5.

Baseline preserved: `outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/` and every
file under `outputs/eval/`, `outputs/bench/` and `outputs/quant/` are untouched.
All experiment artefacts live under `experiments/oscd_improvement/`.

Everything below is measured on the **validation split** (bordeaux + pisa,
62 windows). The official test regions were not read during this work, so the
baseline test figures of AP 0.5068 and object F1 0.0559 stand as recorded and are
not revised here.

---

## 1. Baseline audit

### 1.1 Architecture

`models/siamese_unet.py`, 0.48 M parameters.

| Part | Detail |
|---|---|
| Encoder | four double-conv blocks (64→128→256→512 at base 16), shared between dates |
| Fusion | absolute difference of the two feature stacks at every level (FC-Siam-diff) |
| Decoder | three transposed convolutions, each concatenating the difference skip at that level |
| Head | 1×1 convolution, single-channel logits |
| Receptive field | limited by four pooling stages; a 128 px patch is the whole input |

### 1.2 Training pipeline

| Item | Baseline value |
|---|---|
| Split | official OSCD 14 train / 10 test regions; validation = bordeaux + pisa |
| Windows | 399 train, 62 validation, 230 test at 128 px, stride 128 with an edge patch |
| Loss | `DiceBCELoss`, bce_weight 1.0, dice_weight 1.0, pos_weight 40.81 |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4 |
| Schedule | none — constant learning rate |
| Augmentation | random flips and quarter turns, applied identically to both dates |
| AMP | on; batch 8; peak reserved 0.170 GiB |

Three things stand out. The learning rate never decays, the loss has no term that
penalises the *number* of false objects, and the decision threshold was fixed at
0.5 although `pos_weight` 40.81 inflates every probability.

### 1.3 Evaluation implementations

Pixel metrics come from `utils/metrics.metrics_from_counts`, which returns NaN
rather than zero when a denominator is empty. Average precision and AUC are
ranking measures computed over concatenated patch scores.

Object metrics come from `eval.evaluate_patch`, which labels connected components
and matches them greedily at IoU ≥ 0.5, counting unmatched predictions as false
positives and unmatched truth as false negatives. Components use
**4-connectivity** — scipy's default cross-shaped structure — so two change areas
that touch only at a corner count as two objects. Verified by a pixel-diagonal
test in `results/verification.json`.

### 1.4 Why object F1 is 0.0559, and why that number is partly an artefact

Two separate causes, and the second one is a measurement defect.

**Cause 1 — the operating point.** At threshold 0.5 the model predicts far too
much. On validation the same checkpoint gives:

| Threshold | Pixel precision | Pixel recall | Pixel F1 | Object precision | Object recall | Object F1 |
|---|---|---|---|---|---|---|
| 0.50 | 0.084 | 0.709 | 0.150 | 0.006 | 0.104 | 0.0117 |
| 0.90 | 0.416 | 0.359 | **0.386** | 0.084 | 0.208 | **0.1194** |

Pixel recall at 0.5 is 0.71, so the model is not blind. It is simply producing
about ten times more positive area than there is change, and almost all of it is
scattered. Object F1 punishes that far harder than pixel F1 because every speck
is a separate false object.

**Cause 2 — object metrics are computed per 128 px patch.** This is a defect in
the evaluation, not in the model. `eval.main` loops over patches and calls
`evaluate_patch` on each one, so:

- an object that crosses a patch edge is split into two half-objects, neither of
  which matches at IoU 0.5, so one true object becomes two false negatives;
- a prediction crossing an edge becomes two false positives;
- `patch_origins` appends an edge patch that **overlaps** its neighbour by up to
  127 px, so those objects are counted twice.

Measured on the same predictions, same matching rule, two framings:

| City | Framing | TP | FP | FN | Object F1 |
|---|---|---|---|---|---|
| bordeaux | per patch | 6 | 623 | 34 | 0.0179 |
| bordeaux | scene | 3 | 434 | 23 | 0.0130 |
| pisa | per patch | 12 | 1130 | 76 | 0.0195 |
| pisa | scene | 5 | 848 | 46 | 0.0111 |

Patch framing inflates true positives about twofold and false positives about
1.4×. The direction is not uniform, which is why the F1 changes only slightly
here — but the counts it reports (the "3,368 false objects" figure) are inflated,
and a large true object can never be counted as one object at all.

The same double counting affects average precision: **AP is 0.3326 under patch
framing and 0.2806 under scene framing** for the identical checkpoint. The 0.3362
in the training log is the patch-framed number and is inflated by about 0.05.

**Consequence:** the baseline test numbers (AP 0.5068, object F1 0.0559) are
patch-framed results. They are recorded as such and left unmodified. Every
comparison in this document uses scene framing, computed by calling the same
`evaluate_patch` on full-scene arrays, so the matching rule is unchanged.

### 1.5 Leakage and alignment checks

| Check | Result |
|---|---|
| Train/validation regions disjoint | yes, bordeaux and pisa are validation only |
| Test regions in any training run | no, `official test labels were not read` on every run |
| Augmentation consistency | flips and rotations applied to `image1`, `image2` and `mask` together |
| Mask alignment | patch windows read with the same `Window` for both dates and the label |
| NoData | OSCD rectified rasters have no NoData band; the local SAFE pipeline handles it separately |
| Metric double counting | **found**: overlapping edge patches, see 1.4 |
| Threshold tuning on test | not done; all thresholds below are selected on validation |

One more defect found while auditing: `patch_origins` overlaps the final patch
with its neighbour, so the earlier statement in `docs/report.md` §6.2 that "the
windows do not overlap" is wrong. It affects at most one patch row and column per
scene.

---

## 2. Experiment A — threshold optimisation

One forward pass of the baseline checkpoint over the validation split, stitched
into per-city probability maps, then a sweep of 21 thresholds.

Chosen on validation: **threshold 0.90**, pixel F1 0.3855, object F1 0.1194.

| Threshold | Pixel P | Pixel R | Pixel F1 | Object P | Object R | Object F1 | Object FP |
|---|---|---|---|---|---|---|---|
| 0.50 (baseline) | 0.084 | 0.709 | 0.150 | 0.006 | 0.104 | 0.0117 | 1,282 |
| 0.75 | 0.209 | 0.540 | 0.301 | 0.030 | 0.208 | 0.0529 | 512 |
| 0.85 | 0.327 | 0.440 | 0.375 | 0.046 | 0.182 | 0.0733 | 291 |
| **0.90** | **0.416** | **0.359** | **0.386** | **0.084** | **0.208** | **0.1194** | **175** |
| 0.92 | 0.419 | 0.341 | 0.376 | 0.086 | 0.183 | 0.1172 | 148 |
| 0.95 | 0.352 | 0.300 | 0.324 | 0.076 | 0.116 | 0.0918 | 110 |
| 0.99 | 0.052 | 0.076 | 0.062 | 0.000 | 0.000 | 0.0000 | 47 |

AP is threshold-free at 0.2806 (scene framing) and AUC 0.8818, so no threshold
changes the ranking quality — only the operating point.

**Reading:** moving the threshold from 0.5 to 0.9 multiplies pixel F1 by 2.6 and
object F1 by 10, and cuts false-positive objects from 1,282 to 175. The baseline
threshold was never calibrated: the model was trained with `pos_weight` 40.81,
which pushes every probability upward, and 0.5 was inherited rather than chosen.

This is the single largest improvement available, and it costs nothing.

---

## 3. Experiment D — post-processing

Applied to the thresholded mask at the validation-selected threshold, measured on
both framings.

| Opening | Closing | Min area | Pixel F1 | Object F1 | Object FP |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 0.3855 | 0.1194 | 175 |
| 1 | 1 | 4 | 0.3861 | 0.1373 | 140 |
| 1 | 1 | 16 | 0.3875 | 0.1768 | 88 |
| 1 | 1 | **64** | 0.3862 | **0.2124** | **24** |
| 1 | 1 | 256 | 0.2272 | 0.0238 | 6 |
| 3 | 1 | 0 | 0.3844 | 0.1443 | 103 |
| 3 | 5 | 16 | 0.3869 | 0.1595 | 73 |
| 3 | 5 | 64 | 0.3973 | 0.1607 | 26 |

**Reading:** a 64-pixel minimum component area (25,600 m² at 20 m, 0.026 km²)
raises object F1 from 0.119 to 0.212 while pixel F1 moves by less than 0.001, and
it removes 86 % of the false objects. Opening and closing add nothing beyond it —
morphology is not where the problem is. Above 256 px the filter starts deleting
real detections.

Combined, A and D take object F1 from **0.012 to 0.212** on validation without
retraining, and pixel F1 from **0.150 to 0.386**.

---

## 4. Experiment B — loss comparison

Three losses, everything else identical: base 16, 20 epochs, batch 8, seed 42,
AdamW 1e-4, same windows, same augmentation, same `pos_weight`.

| Loss | AP | AUC | Best threshold | Pixel F1 | Object F1 | Post-processed object F1 |
|---|---|---|---|---|---|---|
| BCE + Dice (baseline) | **0.2806** | 0.8818 | 0.90 | **0.386** | **0.1194** | **0.2124** |
| BCE only | 0.2666 | **0.9113** | 0.95 | 0.340 | 0.0762 | 0.1240 |
| Focal (gamma 2) | 0.2580 | 0.9072 | 0.75 | 0.331 | 0.0430 | 0.1897 |

Implementation: `--loss` on `train.py`, defaulting to the config value so the
baseline path is unchanged. `FocalLoss` lives beside `DiceBCELoss` and keeps the
same `pos_weight` and masking contract, so only the loss shape changes.

**Reading:** neither alternative improves on BCE + Dice. BCE alone ranks better
over the whole score range (AUC 0.911 against 0.882) but is worse at the top of
the ranking, which is what average precision measures. Focal drives the model to
a different operating point (its best threshold is 0.75, not 0.90) and ends up
with the lowest AP.

**These differences are not established.** The baseline's three-seed spread is
AP 0.3362 / 0.2671 / 0.2585, sample SD 0.043, and each alternative here is a
single seed. Gaps of 0.01 to 0.02 AP are well inside that spread. The honest
conclusion is that no loss in this set is distinguishable from the baseline on
one seed, and the baseline stays because it is the only one with a measured
spread.

**A defect found in this experiment, and fixed.** The first attempt at these two
runs produced directories labelled `loss_bce` and `loss_focal` that were both
`DiceBCELoss`: the OSCD trainer builds its own criterion and ignored the new
flag. The tell was two runs printing identical epoch losses to five decimals.
They are kept under `checkpoints/_invalid_*` with an `INVALID.md`, not cited, and
`run_config.json` now records `loss` and `architecture` so a run states what it
did.

---

## 5. Experiment C — sampling

**Not run.** Reason: the experiments that address the false-positive problem
directly (A and D) already produced a 17× object-F1 gain, and the remaining GPU
budget was spent on the loss and architecture comparisons, which change the model
rather than its operating point.

What it would take: class-balanced or hard-negative sampling needs a
`WeightedRandomSampler` over the patch index using each patch's change fraction,
and a second pass that re-weights patches containing the model's current false
positives. That is one afternoon of work plus a full retrain per variant, and the
expected gain is smaller than the 0.09 object-F1 already gained from filtering.

---

## 6. Experiment E — architecture

A Siamese U-Net with a pretrained ResNet-18 encoder
(`models/siamese_resnet18.py`), selected with `--arch siamese_resnet18`.

| Property | Baseline | ResNet-18 variant |
|---|---|---|
| Parameters | 0.48 M | 14.23 M (29.6×) |
| Encoder | four conv blocks, from scratch | torchvision ResNet-18, ImageNet weights |
| Input widening | n/a | stem conv widened to 4 bands, extra band = mean of RGB filters |
| Peak VRAM (batch 8, 128 px) | 0.17 GiB | 0.35 GiB |
| Training time (20 epochs) | 11.1 s/epoch | 6.5 s/epoch |

Training time is not comparable across these runs: the job is data-bound at
10-20 % GPU utilisation, and the measured epoch time tracked how much else the
machine was doing, not the model.

| Run | AP | AUC | Best threshold | Pixel F1 | Object F1 | Post-processed object F1 |
|---|---|---|---|---|---|---|
| Baseline (BCE + Dice) | 0.2806 | 0.8818 | 0.90 | **0.386** | **0.1194** | **0.2124** |
| ResNet-18 encoder | **0.3067** | 0.8874 | 0.95 | 0.360 | 0.0413 | 0.0893 |

**Reading:** the pretrained encoder gives the best average precision of the four
configurations, +0.026 over the baseline, and the worst object F1, less than half
the baseline's. The pattern is consistent with what a bigger receptive field
does: it finds change pixels more reliably but smears them over a larger area, so
a prediction that is roughly right at pixel level rarely matches a small true
object at IoU 0.5. Object-level matching is unforgiving of a dilated mask.

That trade-off matters for this project, where the deliverable is a detection
register of discrete objects, not a change map.

**Updated with three seeds** (`VERIFICATION.md` section 1.2): ResNet-18 mean AP
0.2555 +/- 0.0459 against the baseline's 0.2375 +/- 0.0374, and mean object F1
0.0923 +/- 0.0317 against 0.1575 +/- 0.0522. The AP edge of +0.018 is well inside
the seed noise; the object-F1 deficit of 0.065 is consistent across all three
seeds. Seed 42 — the number quoted in the table above — is the best seed for both
architectures and must not be read as expected performance.

---

## 7. Summary against the baseline

All numbers validation-only, scene framing, each run at its own
validation-selected threshold, with 64-pixel minimum component area. The two
mean ± SD rows are scored at one fixed operating point (0.90, 64 px) so the seeds
are comparable; see `VERIFICATION.md` section 1.2.

| Configuration | AP | Pixel F1 | Object F1 | Object FP |
|---|---|---|---|---|
| Baseline at threshold 0.5, no filtering (as recorded) | 0.2806 | 0.150 | 0.0117 | 1,282 |
| Baseline at 0.90 + min area 64 (**recommended**) | 0.2806 | 0.386 | **0.2124** | **24** |
| BCE only at 0.95 + min area 64 | 0.2666 | 0.335 | 0.1240 | 36 |
| Focal at 0.75 + morphology + min area 64 | 0.2580 | 0.327 | 0.1897 | 28 |
| ResNet-18 at 0.95 + min area 64 | **0.3067** | 0.372 | 0.0893 | 30 |
| ResNet-18, three-seed mean ± SD | 0.2555 ± 0.0459 | — | 0.0923 ± 0.0317 | — |
| Baseline, three-seed mean ± SD | 0.2375 ± 0.0374 | — | 0.1575 ± 0.0522 | — |

**Recommended configuration to carry forward:** the existing architecture and
BCE + Dice loss, with the operating point moved to threshold 0.90 and a 64-pixel
minimum component area. That is a 17× improvement in object F1 and a 2.6×
improvement in pixel F1 over the recorded baseline behaviour, at zero training
cost, and it changes no model weights.

Nothing in this experiment justifies retraining for a better model yet. The
changes that helped are measurement and operating-point changes; the changes that
altered the model did not beat the baseline beyond noise.

**Before any of this is used for fine-tuning, two things are needed:**

1. A three-seed repeat of the baseline at the tuned operating point, so the
   comparison band is measured rather than borrowed from a different metric
   framing. The patch-framed SD of 0.043 is not the right yardstick for
   scene-framed AP.
2. A second look at the ResNet-18 variant on those seeds. If its AP advantage
   survives, it is a better feature extractor and worth fine-tuning; if not, the
   0.48 M model is the right starting point for a 4 GB target.

---

## 8. Deliverables

| Requested | Where |
|---|---|
| Baseline reproduction | `results/baseline_seed42/` (predictions, thresholds, post-processing, PR figure) |
| Experiment configurations | `run_config.json` in each `checkpoints/<run>/seed_42/`, plus `--loss` and `--arch` flags |
| Training and validation logs | `train_log.csv` and `validation_by_region.csv` per run |
| Precision-recall curves | `results/*/pr_curve.png`, plus `thresholds.csv` with every point |
| Metric comparison table | `results/comparison.csv`, reproduced in section 7 |
| Prediction visualisations | `results/figures_*/<city>_{false_positives,false_negatives}.png` |
| Object-level analysis | section 1.4, `results/*/analysis.json` framing block |
| Best-performing checkpoint | baseline at the tuned operating point; no new checkpoint is promoted |
| Experiment report | this file |

Not delivered: a three-seed repeat of the promising configurations (time), and
Phase 2C sampling. Both are listed as next steps, not as results.

---

## 9. Experimental controls

- Official train/test separation preserved; test labels never read.
- Validation regions are two held-out cities, not random patches from the same
  scenes.
- Thresholds and post-processing parameters selected on validation only.
- Identical geometric augmentation applied to both dates.
- Seeds recorded per run; the baseline is the only configuration with a
  three-seed spread so far (AP 0.3362 / 0.2671 / 0.2585 patch-framed).
- One GPU job at a time; batch 8 at 128 px keeps peak reserved memory at 0.17 GiB.

## 10. What this is not

- Not a new baseline. The original checkpoint, evaluation outputs and quantised
  models are unchanged, and no test-set number has been recomputed.
- Not a claim that the model is now good. Object F1 0.212 is better than 0.012 and
  still poor: at the tuned operating point the model misses about 79 % of true
  change objects.
- Not transferable evidence for the SkyServe tiles. Everything here is OSCD.
