"""Benchmark ONNX FP32 and INT8 QDQ over the whole OSCD validation split.

Three configurations, same checkpoint, same windows, same threshold, same
evaluation code:

    cuda_fp32   ONNX FP32 on CUDAExecutionProvider
    cuda_int8   ONNX INT8 QDQ on CUDAExecutionProvider
    cpu_int8    ONNX INT8 QDQ on CPUExecutionProvider, the reference

CUDAExecutionProvider is required, not requested: ``create_session`` raises if
it was dropped, so a silent CPU fallback cannot produce a number here.

The validation split only. The official test split is never opened by this
script; ``make_dataset(..., "validation", ...)`` is the only split requested.

Outputs go to a fresh directory and are never overwritten.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import onnxruntime as ort
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from quantize import evaluate_ort, load_model, make_dataset
from utils.ort_session import CPU, CUDA, create_session, optimisation_label

# CPU QDQ sessions are pinned below the QLinearConv fusion that corrupts this
# model (docs/decisions/009-cpu-qdq-optimiser-defect.md). CUDA keeps the default
# level, which is numerically identical to its own unoptimised path.
CPU_QDQ_OPTIMISATION = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
CUDA_OPTIMISATION = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_memory_used_mib():
    """Memory in use on GPU 0, from nvidia-smi. None when it is unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


class MemorySampler(threading.Thread):
    """Sample GPU memory in the background and keep the peak."""

    def __init__(self, interval_s=0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.peak_mib = 0.0
        self.baseline_mib = gpu_memory_used_mib() or 0.0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            used = gpu_memory_used_mib()
            if used is not None:
                self.peak_mib = max(self.peak_mib, used)
            self._stop_event.wait(self.interval_s)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)


def run_pass(session, loader, input_names, timed=False):
    """One pass over the loader. Returns per-batch seconds when timed."""
    durations = []
    for batch in loader:
        feed = {input_names[0]: batch["image1"].numpy().astype(np.float32),
                input_names[1]: batch["image2"].numpy().astype(np.float32)}
        start = time.perf_counter()
        session.run(None, feed)
        if timed:
            durations.append(time.perf_counter() - start)
    return durations


def partition_report(model_path, providers, loader, input_names, label):
    """Kernel placements and CPU boundary count, from ORT profiling events."""
    options = ort.SessionOptions()
    options.enable_profiling = True
    session, active = create_session(model_path, providers,
                                     sess_options=options, require=providers[0],
                                     preload_cuda=True,
                                     optimization_level=CUDA_OPTIMISATION)
    batch = next(iter(loader))
    session.run(None, {input_names[0]: batch["image1"].numpy().astype(np.float32),
                       input_names[1]: batch["image2"].numpy().astype(np.float32)})
    profile_path = session.end_profiling()
    with open(profile_path) as handle:
        events = json.load(handle)
    os.unlink(profile_path)

    counts = {}
    memcpy_nodes = 0
    for event in events:
        args = event.get("args") or {}
        provider = args.get("provider")
        if not provider:
            continue
        counts[provider] = counts.get(provider, 0) + 1
        if "Memcpy" in str(event.get("name", "")):
            memcpy_nodes += 1
    total = sum(counts.values())
    return {
        "label": label,
        "active_providers": active,
        "kernel_launches": counts,
        "total_kernel_launches": total,
        "cuda_fraction": counts.get(CUDA, 0) / total if total else 0.0,
        "cpu_memcpy_kernel_events": memcpy_nodes,
    }


def summarise(durations, windows):
    array = np.asarray(durations, dtype=np.float64) * 1000.0     # ms per batch
    per_window = array / windows
    return {
        "batches": int(array.size),
        "batch_ms_median": float(np.median(array)),
        "batch_ms_p95": float(np.percentile(array, 95)),
        "batch_ms_min": float(array.min()),
        "batch_ms_max": float(array.max()),
        "window_ms_median": float(np.median(per_window)),
        "windows_per_second": float(windows / (array.sum() / array.size / 1000.0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--quant-dir", required=True,
                        help="directory holding onboard_fp32.onnx and onboard_int8.onnx")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2, help="warm-up passes")
    parser.add_argument("--runs", type=int, default=3, help="timed passes")
    args = parser.parse_args()

    output_dir = os.path.join(ROOT, args.output_dir)
    json_path = os.path.join(output_dir, "validation_benchmark.json")
    md_path = os.path.join(output_dir, "validation_benchmark.md")
    for path in (json_path, md_path):
        if os.path.exists(path):
            parser.error("refusing to overwrite %s" % path)

    fp32_path = os.path.join(ROOT, args.quant_dir, "onboard_fp32.onnx")
    int8_path = os.path.join(ROOT, args.quant_dir, "onboard_int8.onnx")
    for path in (fp32_path, int8_path):
        if not os.path.isfile(path):
            parser.error("model not found: %s" % path)

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    threshold = float(cfg["evaluation"]["threshold"])

    _, bands, checkpoint = load_model(args.checkpoint, torch.device("cpu"))
    loader = make_dataset(cfg, "validation", checkpoint, args.batch_size)
    windows = len(loader.dataset)
    input_names = [item.name for item in
                   create_session(fp32_path, [CPU])[0].get_inputs()]
    print("validation windows", windows, "batch", args.batch_size,
          "threshold", threshold, "bands", bands)

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "models": {
            "fp32": {"path": fp32_path, "sha256": sha256(fp32_path),
                     "size_bytes": os.path.getsize(fp32_path)},
            "int8": {"path": int8_path, "sha256": sha256(int8_path),
                     "size_bytes": os.path.getsize(int8_path)},
        },
        "split": "validation",
        "official_test_accessed": False,
        "windows": windows,
        "batch_size": args.batch_size,
        "threshold": threshold,
        "band_order": bands,
        "warmup_passes": args.warmup,
        "timed_passes": args.runs,
        "available_providers": ort.get_available_providers(),
        "status": "measured_development_result",
        "runs": {},
    }

    configurations = [
        ("cuda_fp32", fp32_path, [CUDA, CPU], True, CUDA_OPTIMISATION),
        ("cuda_int8", int8_path, [CUDA, CPU], True, CUDA_OPTIMISATION),
        ("cpu_int8", int8_path, [CPU], False, CPU_QDQ_OPTIMISATION),
    ]

    for name, model_path, providers, uses_gpu, level in configurations:
        print("\n===", name)
        session, active = create_session(model_path, providers, require=providers[0],
                                         preload_cuda=uses_gpu,
                                         optimization_level=level)
        print("   providers:", active, "optimisation:", optimisation_label(level))

        sampler = MemorySampler() if uses_gpu else None
        if sampler:
            sampler.start()

        for _ in range(args.warmup):
            run_pass(session, loader, input_names)
        durations = []
        for _ in range(args.runs):
            durations.extend(run_pass(session, loader, input_names, timed=True))

        if sampler:
            sampler.stop()

        metrics, _ = evaluate_ort(session, loader, input_names, threshold)
        entry = {
            "model": os.path.basename(model_path),
            "active_providers": active,
            "optimisation": optimisation_label(level),
            "cuda_required": uses_gpu,
            "metrics": {k: metrics[k] for k in
                        ("ap", "auc", "precision", "recall", "f1", "iou",
                         "tp", "fp", "fn", "tn")},
            "latency": summarise(durations, args.batch_size),
        }
        if sampler:
            entry["gpu_memory"] = {
                "baseline_mib": sampler.baseline_mib,
                "peak_mib": sampler.peak_mib,
                "delta_mib": sampler.peak_mib - sampler.baseline_mib,
            }
        report["runs"][name] = entry
        print("   AP %.4f AUC %.4f F1 %.4f  median %.2f ms/batch  %.1f win/s"
              % (metrics["ap"], metrics["auc"], metrics["f1"],
                 entry["latency"]["batch_ms_median"],
                 entry["latency"]["windows_per_second"]))
        if sampler:
            print("   GPU peak %.0f MiB (delta %.0f)"
                  % (entry["gpu_memory"]["peak_mib"], entry["gpu_memory"]["delta_mib"]))

    report["partitioning"] = {
        "cuda_fp32": partition_report(fp32_path, [CUDA, CPU], loader, input_names,
                                      "cuda_fp32"),
        "cuda_int8": partition_report(int8_path, [CUDA, CPU], loader, input_names,
                                      "cuda_int8"),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2)

    lines = [
        "# Validation benchmark: ONNX FP32 and INT8 across providers",
        "",
        "Checkpoint `%s`, sha256 `%s`." % (report["checkpoint"],
                                           report["checkpoint_sha256"][:16]),
        "Validation split only: %d windows, batch %d, threshold %.2f."
        % (windows, args.batch_size, threshold),
        "The official OSCD test split was not opened.",
        "",
        "| Run | Providers | Optimisation | AP | AUC | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ("cuda_fp32", "cuda_int8", "cpu_int8"):
        entry = report["runs"][name]
        metric = entry["metrics"]
        lines.append("| %s | %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f |"
                     % (name, " + ".join(p.replace("ExecutionProvider", "")
                                         for p in entry["active_providers"]),
                        entry["optimisation"],
                        metric["ap"], metric["auc"], metric["precision"],
                        metric["recall"], metric["f1"]))
    lines += ["", "## Latency", "",
              "| Run | Median ms/batch | p95 ms/batch | Median ms/window | Windows/s |",
              "|---|---|---|---|---|"]
    for name in ("cuda_fp32", "cuda_int8", "cpu_int8"):
        latency = report["runs"][name]["latency"]
        lines.append("| %s | %.2f | %.2f | %.2f | %.1f |"
                     % (name, latency["batch_ms_median"], latency["batch_ms_p95"],
                        latency["window_ms_median"], latency["windows_per_second"]))
    lines += ["", "## Memory and placement", "",
              "| Run | GPU peak MiB | Delta MiB | CUDA kernels | CPU kernels | CPU Memcpy events |",
              "|---|---|---|---|---|---|"]
    for name in ("cuda_fp32", "cuda_int8"):
        entry = report["runs"][name]
        part = report["partitioning"][name]
        memory = entry.get("gpu_memory", {})
        lines.append("| %s | %.0f | %.0f | %d (%.1f %%) | %d | %d |"
                     % (name, memory.get("peak_mib", 0), memory.get("delta_mib", 0),
                        part["kernel_launches"].get(CUDA, 0),
                        100.0 * part["cuda_fraction"],
                        part["kernel_launches"].get(CPU, 0),
                        part["cpu_memcpy_kernel_events"]))
    lines += ["", "`cpu_int8` is the reference: no GPU memory, CPU provider only.", ""]
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines))

    print("\nwrote", json_path)
    print("wrote", md_path)


if __name__ == "__main__":
    main()
