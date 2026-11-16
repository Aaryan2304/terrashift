# 006 — GPU training configuration on a 4 GB laptop GPU

**Status:** accepted
**Date:** 2026-11-15

## Context

The available GPU is an RTX 3050 Ti Laptop with 4 GB of VRAM. The user approved
CUDA training on it. The question was which batch size, and whether the GPU is
the limiting resource at all.

## Decision

Measured memory on the real model, loss and training step at 128 x 128 with AMP:
batch 2 uses 0.059 GiB reserved, batch 8 uses 0.164 GiB, batch 16 uses 0.332 GiB.

Batch 8 is used for training, chosen for optimizer steps (50 per epoch over 399
windows), not for memory. AMP is on. Gradient accumulation is not implemented
and nothing claims it.

## Alternatives rejected

- **Batch 16 because memory allows it.** It was measured as a trial and gave
  AP 0.2753 against 0.3362: fewer steps per epoch, worse result.
- **Batch 32 or more.** Not needed; the run is data-bound at 15.9 % mean GPU
  utilisation, so a larger batch would reduce the step count without using the
  idle capacity.

## Consequence

Training is data-bound, not compute-bound. The useful optimisation would be
`num_workers` and faster patch decoding, not a smaller model or a bigger batch.
That is a known, unaddressed limit of this run: 5.7 s per epoch at 16 % GPU
utilisation means about 84 % of the wall clock is not the GPU.

Three seeds give AP 0.3362 / 0.2671 / 0.2585, mean 0.2873, sample SD 0.0426. The
single-seed number would have overstated the model.

## Evidence

`scripts/measure_training_memory.py`, `run_config.json` in each checkpoint
directory, `train_log.csv` (per-epoch GPU utilisation and peak reserved memory),
`docs/report.md` section 4.
