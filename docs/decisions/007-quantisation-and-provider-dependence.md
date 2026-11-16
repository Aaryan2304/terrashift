# 007 — Static INT8 QDQ, measured on the provider it will run on

**Status:** accepted
**Date:** 2026-11-15

## Context

Quantisation was on the deliverable list, and the brief required a CPU FP32
against CPU INT8 benchmark. The first measurement showed INT8 losing 0.0743 AP
and running slower than FP32 on the CPU, which was reported as "INT8 is not a
win".

A later full-split benchmark added the CUDA path and showed the same model losing
only 0.0048 AP there, with a higher AUC. The two providers disagree on ranking
(Spearman 0.725) and on level (mean probability 0.465 on CPU against 0.321 on
CUDA) for the same file.

## Decision

- **Method:** static QDQ, QInt8 weights and activations, per-channel,
  `reduce_range` off, calibrated on the first 20 batches of the training split at
  batch 8 — training data only, never validation or test.
- **Reporting:** every quantisation accuracy figure is reported with the
  execution provider that produced it. A single provider-independent "INT8
  accuracy" is not reported, because it does not exist for this graph.
- **TensorRT:** not installed, by decision. The runtime library is absent, the
  install would touch a working environment, and the existing measurements
  already answer the deployment question.

## Alternatives rejected

- **Quantisation-aware training.** The right tool if INT8 were the target, but it
  needs a training budget that the deadline does not have, and it would need the
  local labels that do not yet exist.
- **Report the CPU number as the INT8 result.** It is the pessimistic half of a
  provider-dependent measurement, and it was already quoted once before the GPU
  number existed. Both are now reported side by side.
- **Install TensorRT to settle the question.** Refused by the user. The CUDA FP32
  path is already the fastest measured configuration.

## Consequence

- INT8 buys size (28.3 % of FP32) and nothing else here: it is 29 % slower than
  FP32 on CUDA and 6.5 times slower on CPU.
- The CPU and CUDA kernels produce materially different functions from one QDQ
  file. The cause is not diagnosed. Any deployment on a third target must
  re-measure, not assume.
- The comparison must be repeated after local fine-tuning, on the final model.

## Evidence

`quantize.py`, `scripts/benchmark_validation_providers.py`,
`outputs/quant/oscd_baseline_seed42_v2/quantization_report.json`,
`outputs/bench/oscd_validation_providers/validation_benchmark.json`,
`docs/report.md` sections 6.1 to 6.6.
