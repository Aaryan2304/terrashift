# 009 — ONNX Runtime's CPU QDQ fusion corrupts this graph

**Status:** diagnosed, workaround implemented and verified
**Date:** 2026-11-15
**Upstream:** issue [microsoft/onnxruntime#24711](https://github.com/microsoft/onnxruntime/issues/24711),
fix in PR [#28229](https://github.com/microsoft/onnxruntime/pull/28229), released in v1.27.0
**Installed here:** onnxruntime-gpu 1.24.4, so the fix is not present

## Symptom

The same `onboard_int8.onnx`, checkpoint, 62 validation windows and threshold
produced AP 0.33148 on CUDA and 0.26196 on CPU, with Pearson correlation 0.734
between the two score vectors.

## Isolation

Both sessions were fed one frozen batch written to disk once, so the inputs were
byte-identical. The FP32 control agrees across providers to MAE 6.6e-05 with
correlation 1.000000, so preprocessing, ordering and output handling are not
involved.

The split appears only with graph optimisation, and only on CPU:

| Comparison | MAE | r |
|---|---|---|
| CPU optimised against CPU unoptimised | 0.162715 | 0.678 |
| CUDA optimised against CUDA unoptimised | 0.000000 | 1.000 |
| CPU against CUDA, both unoptimised | 0.013740 | 0.994 |

By level: `ORT_ENABLE_BASIC` mean 0.3194, `ORT_ENABLE_EXTENDED` 0.4530,
`ORT_ENABLE_ALL` 0.4530. The jump is exactly at EXTENDED, where the QDQ selector
fuses Conv into QLinearConv.

## Which execution is correct

The ONNX reference evaluator executed the same graph with no ORT transforms:

| Path | MAE against reference | r |
|---|---|---|
| CPU, ORT_ENABLE_BASIC | 0.000000 | 1.000000 |
| CUDA, ORT_ENABLE_ALL | 0.012006 | 0.988153 |
| CPU, ORT_ENABLE_ALL | 0.142675 | 0.750494 |

The unfused CPU path reproduces the graph's semantics exactly. The fused path
does not.

## Root cause: the QLinearConv bias-scale invariant

The ONNX QLinearConv specification requires the int32 bias to be quantised with
``bias_scale[i] == x_scale * w_scale[i]``, because the fused kernel applies the
input and weight scales itself and then reuses the bias integers unchanged.
Upstream PR #28229 adds `CheckConvBiasScale` to skip the fusion when that
relation fails; before it, the selector only checked that the bias dtype was
INT32. This is issue #24711.

All 17 fused convolutions were inspected:

| Group | Count | Weight quantisation | bias_scale / (x_scale x w_scale) | Conforming |
|---|---|---|---|---|
| First Siamese branch and decoder | 9 | per-channel, zero point 0 | exactly 1.0 | yes |
| Second Siamese branch (`*_1`) | 8 | per-tensor, zero point -5 to 0 | 0.1801 to 1.9452 | no |

Details that matter:

- **Every fused convolution has bias zero point 0**, so the zero-point aspect of
  the invariant is not implicated here. Only the scale relation fails.
- **The bias tensor is shared** between the Siamese branches: the twin convs
  reach the same int32 bias through Identity nodes. The float path stays correct
  because each branch dequantises the bias with its own scale. The fused path
  does not, because the kernel replaces that scale with `x_scale * w_scale`, and
  the two branches have different weight scales.
- The non-conforming values come from the model, not from ORT: quantising a
  Siamese export whose branches carry different weight quantisation leaves one
  shared bias unable to satisfy both.

### Causal confirmation

A scratch copy of the model gave each offending convolution its own bias
re-quantised against its canonical scale, with the float bias held identical. The
original ONNX file was not modified. On one frozen batch:

| Case | MAE against the unfused original | r |
|---|---|---|
| Original, fused against unfused | 0.162715 | 0.678 |
| Repaired, fused against unfused | **0.017891** | **0.9905** |

Fusion was still active in the repaired graph (17 QLinearConv). Repairing only
the bias scales removes about 89 % of the error, which identifies the bias-scale
invariant as the dominant cause.

### What it is not

The int8-to-uint8 weight conversion also visible in the fused graph (weight zero
points 123 to 128) is a consequence of converting the twin branch's asymmetric
weights, not an independent cause. With the bias scales repaired the fused path
agrees with the unfused path to r 0.991 while that conversion still happens. The
residual 0.018 MAE is rounding introduced by re-quantising the bias.

An earlier draft of this record blamed weight clipping in the uint8 conversion.
The graph inspection does not support that: clipping occurs in at most 2 of the
17 nodes and accounts for a small part of the error, against 8 nodes with a
non-conforming bias scale and a 9x error reduction when only those are repaired.

## Workaround, implemented

1. `utils/ort_session.create_session` accepts `optimization_level`.
2. `quantize.py` and `scripts/benchmark_validation_providers.py` pin CPU QDQ
   sessions to `ORT_ENABLE_BASIC` through `CPU_QDQ_OPTIMISATION`.
3. CUDA keeps `ORT_ENABLE_ALL`, which agrees with its own unoptimised path.
4. Every benchmark entry records the execution provider and the optimisation
   level together.

No model weights, scales or zero points were changed, and no ONNX file was
altered.

## Corrected numbers

Validation split, 62 windows, batch 8, threshold 0.5:

| Run | Optimisation | AP | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| cuda_fp32 | ORT_ENABLE_ALL | 0.33626 | 0.90253 | 0.0998 | 0.7593 | 0.1764 |
| cuda_int8 | ORT_ENABLE_ALL | 0.33148 | 0.91011 | 0.0854 | 0.8015 | 0.1544 |
| cpu_int8 | ORT_ENABLE_BASIC | 0.33198 | 0.91001 | 0.0857 | 0.8020 | 0.1548 |

Latency, 2 warm-up passes then 3 timed passes over 24 batches:

| Run | median ms/batch | p95 ms/batch | ms/window | windows/s |
|---|---|---|---|---|
| cuda_fp32 | 13.82 | 94.64 | 1.73 | 237.5 |
| cuda_int8 | 16.82 | 96.64 | 2.10 | 220.1 |
| cpu_int8 | 161.47 | 183.31 | 20.18 | 49.1 |

The CPU figure **supersedes AP 0.26196**, which came from the defective fused
path. Evidence: `outputs/bench/oscd_validation_providers_v2/`.

## Tests, implemented

`tests/test_qdq_bias_scale.py`, seven tests:

1. Fusion occurs at EXTENDED and above, and not at BASIC, counted in the dumped
   optimised graph so the test does not rest on numerics alone.
2. A conforming bias scale matches the ONNX reference at BASIC and at ALL.
3. A non-zero bias zero point with a conforming scale matches the reference at
   BASIC.
4. A mismatched bias scale (factor 2) is correct unfused and wrong when fused, on
   ORT versions below 1.27.0.
5. A mismatched bias scale with a non-zero bias zero point is correct unfused.
6. CPU BASIC against CUDA on the project's real quantised model, skipped when no
   CUDA provider is present.
7. The tooling actually uses `ORT_ENABLE_BASIC`, so the workaround cannot be
   dropped silently.

The reproducer is built at opset 21 natively, an opset the bundled ONNX reference
evaluator supports, so no existing model is edited and no opset label is faked.

## Remaining limitations

1. **The model still violates the QLinearConv bias invariant.** The workaround
   avoids the fusion; it does not repair the model. The durable fix is to export
   and quantise so both Siamese branches share one weight quantisation, or to
   give each branch its own bias initialiser. That needs re-export and
   re-quantisation, out of scope here, and due again after local fine-tuning.
2. **The workaround costs CPU speed.** Removing the fusion removes the fused int8
   kernels: CPU median moved from 59.96 ms to 161.47 ms per batch of eight, about
   2.7 times slower. That is the price of correctness until ORT 1.27 or later is
   available, where only the offending convolutions are skipped.
3. **onnxruntime-gpu 1.24.4 predates the fix.** Upgrading is a separate decision
   and was not done.
4. **Absolute latencies on this laptop are not stable between sessions.** The
   identical CUDA FP32 configuration measured 9.16 ms in the first benchmark and
   13.82 ms in the repeat. Ratios within one run are meaningful; absolute
   comparisons across runs are not.
