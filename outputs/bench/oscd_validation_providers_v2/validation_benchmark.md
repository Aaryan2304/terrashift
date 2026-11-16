# Validation benchmark: ONNX FP32 and INT8 across providers

Checkpoint `/home/aaryan/projects/skyserve_assignment/outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt`, sha256 `4d9c381dba370bec`.
Validation split only: 62 windows, batch 8, threshold 0.50.
The official OSCD test split was not opened.

| Run | Providers | Optimisation | AP | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| cuda_fp32 | CUDA + CPU | ORT_ENABLE_ALL | 0.3363 | 0.9025 | 0.0998 | 0.7593 | 0.1764 |
| cuda_int8 | CUDA + CPU | ORT_ENABLE_ALL | 0.3315 | 0.9101 | 0.0854 | 0.8015 | 0.1544 |
| cpu_int8 | CPU | ORT_ENABLE_BASIC | 0.3320 | 0.9100 | 0.0857 | 0.8020 | 0.1548 |

## Latency

| Run | Median ms/batch | p95 ms/batch | Median ms/window | Windows/s |
|---|---|---|---|---|
| cuda_fp32 | 13.82 | 94.64 | 1.73 | 237.5 |
| cuda_int8 | 16.82 | 96.64 | 2.10 | 220.1 |
| cpu_int8 | 161.47 | 183.31 | 20.18 | 49.1 |

## Memory and placement

| Run | GPU peak MiB | Delta MiB | CUDA kernels | CPU kernels | CPU Memcpy events |
|---|---|---|---|---|---|
| cuda_fp32 | 231 | 136 | 66 (100.0 %) | 0 | 0 |
| cuda_int8 | 233 | 134 | 202 (88.6 %) | 26 | 26 |

`cpu_int8` is the reference: no GPU memory, CPU provider only.
