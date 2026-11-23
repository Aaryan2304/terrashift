"""GPU inference evaluation: PyTorch CUDA FP32/FP16 and ONNX Runtime CUDA FP32.

Reuses the established validation protocol end to end: the same canonical windows,
pair-specific normalisation, reviewed masks, lattice aggregation and metric functions as
every earlier report. Adds GPU-specific runtime verification, accuracy parity against the
CUDA FP32 reference, component sensitivity and CUDA-synchronised benchmarking.

    python scripts/evaluation/gpu_inference_evaluation.py --stage eval
    python scripts/evaluation/gpu_inference_evaluation.py --stage bench

No training, no calibration, no test data, no full-scene inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import quantize
from terrashift.evaluation.components import component_recall, identify_known_component
from terrashift.inference.torch_session import TorchSession
from terrashift.inference.gpu import GpuSampler, build_entries

OUT = ROOT / "outputs" / "evaluation" / "gpu_inference_evaluation_v1"
CHECKPOINT = ROOT / "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"
NORMALISATION = ROOT / "outputs/dataset/v2/normalisation.json"
FP32_ONNX = ROOT / "outputs/quant/local_manual_v2_seed43_posweight_0p834_int8_v1/onboard_fp32.onnx"
INT8_ONNX = ROOT / "outputs/quant/local_manual_v2_seed43_posweight_0p834_int8_v1/onboard_int8.onnx"
THRESHOLDS = (0.05, 0.10, 0.50)
EXPECTED = {
    "checkpoint": "dbce89f6811d6cdb6d56dd7d6844e58bf673f5ff65806c5d4939cced0a2554cd",
    "fp32_onnx": "6d58e3094a7f61da19571381f2d076ef9c54b8c5e9944a28618de659b5774844",
    "int8_onnx": "326fb3c9bf417352d7eb4262aa325d68fbbd3546eace9cd8bdfd7f2a271b0d2e",
    "normalisation": "8aeea033eb2bec5012ebbceb7eca825814b6e43bb699f9e2eaee0c5bba955d2a",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_eval(args):
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    entries, bands, checkpoint, normalisation = build_entries(cfg)
    device = torch.device("cuda")

    _, _, checkpoint_raw = quantize.load_model(str(CHECKPOINT), "cpu")
    wrapper32 = quantize.Wrapper(quantize.SiameseUNet(in_channels=len(bands),
                                                      base=int(checkpoint_raw["base"])))
    wrapper32.load_state_dict(
        {"model." + key: value for key, value in
         checkpoint_raw["model_state_dict"].items()}, strict=True)
    wrapper32 = wrapper32.to(device).float().eval()
    wrapper16 = quantize.Wrapper(quantize.SiameseUNet(in_channels=len(bands),
                                                      base=int(checkpoint_raw["base"])))
    wrapper16.load_state_dict(wrapper32.state_dict(), strict=True)
    wrapper16 = wrapper16.to(device).half().eval()

    runtimes = {
        "torch_cuda_fp32": (TorchSession(wrapper32, device, half=False),
                            quantize.INPUT_NAMES,
                            {"backend": "torch", "precision": "fp32"}),
        "torch_cuda_fp16": (TorchSession(wrapper16, device, half=True),
                            quantize.INPUT_NAMES,
                            {"backend": "torch", "precision": "fp16"}),
    }
    import onnxruntime as ort
    ort_session = ort.InferenceSession(str(FP32_ONNX),
                                       providers=["CUDAExecutionProvider",
                                                  "CPUExecutionProvider"])
    active = ort_session.get_providers()
    if "CUDAExecutionProvider" not in active:
        raise SystemExit("ONNX Runtime fell back to %s; refusing to report it as GPU" % active)
    runtimes["ort_cuda_fp32"] = (ort_session, quantize.INPUT_NAMES,
                                 {"backend": "onnxruntime", "precision": "fp32",
                                  "providers": active})

    evaluated = {}
    sampler = GpuSampler()
    sampler.start()
    for label, (session, input_names, info) in runtimes.items():
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats() if label.startswith("torch") else None
        result = quantize.evaluate_sessions({label: (session, input_names)}, entries,
                                            THRESHOLDS, args.batch_size)[label]
        evaluated[label] = {
            "pairs": {pair: {key: result["pairs"][pair][key]
                             for key in ("ap", "auc", "pixels", "positives", "thresholds")}
                      for pair in result["pairs"]},
            "pooled": result["pooled"], "macro": result["macro"],
            "scores": result["scores"], "labels": result["labels"],
            "runtime": info,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        if label.startswith("torch"):
            evaluated[label]["peak_allocated_mib"] = round(
                torch.cuda.max_memory_allocated() / 2**20, 1)
            evaluated[label]["peak_reserved_mib"] = round(
                torch.cuda.max_memory_reserved() / 2**20, 1)
        print("[%s] pooled AP %.6f AUC %.6f (%.1f s)"
              % (label, result["pooled"]["ap"], result["pooled"]["auc"],
                 evaluated[label]["elapsed_seconds"]))
        del result
    sampler.stop_event.set()
    sampler.join(timeout=2)
    device_peak = sampler.peak

    reference = evaluated["torch_cuda_fp32"]
    for label in evaluated:
        if label == "torch_cuda_fp32":
            continue
        differences = {}
        for pair in evaluated[label]["pairs"]:
            base = reference["scores"][pair]
            other = evaluated[label]["scores"][pair]
            if base.shape != other.shape:
                raise SystemExit("%s pair %s score vector length differs" % (label, pair))
            delta = np.abs(base - other)
            differences[pair] = {
                "pixels": int(delta.size),
                "mean_abs_difference": float(delta.mean()),
                "max_abs_difference": float(delta.max()),
                "p95_abs_difference": float(np.percentile(delta, 95)),
                "p99_abs_difference": float(np.percentile(delta, 99)),
                "pearson_r": float(np.corrcoef(base, other)[0, 1]),
                "non_finite": int((~np.isfinite(other)).sum()),
                "decision_changes": {
                    "%.2f" % t: int(((base >= t) != (other >= t)).sum()) for t in THRESHOLDS},
            }
        evaluated[label]["difference_vs_torch_fp32"] = differences

    # component sensitivity against the reviewed change mask
    for label in evaluated:
        components = {}
        for pair in entries:
            labels = evaluated[label]["labels"][pair]
            per_threshold = {}
            for threshold in THRESHOLDS:
                key = "%.2f" % threshold
                payload = component_recall(evaluated[label]["scores"][pair], labels,
                                           entries[pair]["target"], threshold)
                payload["known_component"] = identify_known_component(payload["rows"], pair)
                per_threshold[key] = payload
            components[pair] = per_threshold
        evaluated[label]["components"] = components

    serialisable = {}
    for label, payload in evaluated.items():
        serialisable[label] = {key: value for key, value in payload.items()
                               if key not in ("scores", "labels")}
    manifest = {
        "thresholds": list(THRESHOLDS),
        "threshold_status": "historical comparison points; not validated operating thresholds",
        "batch_size": args.batch_size,
        "providers": {label: payload["runtime"] for label, payload in evaluated.items()},
        "device_peak_vram_mib_during_eval": round(device_peak, 1),
        "peak_process_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "results": serialisable,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gpu_eval_results.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", OUT / "gpu_eval_results.json")


def stage_bench(args):
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    _, bands, checkpoint, _ = build_entries(cfg)
    device = torch.device("cuda")
    wrapper = quantize.Wrapper(quantize.SiameseUNet(in_channels=len(bands),
                                                   base=int(checkpoint["base"])))
    wrapper.load_state_dict({"model." + key: value for key, value in
                             checkpoint["model_state_dict"].items()}, strict=True)
    wrapper = wrapper.to(device).float().eval()
    wrapper16 = quantize.Wrapper(quantize.SiameseUNet(in_channels=len(bands),
                                                      base=int(checkpoint["base"])))
    wrapper16.load_state_dict(wrapper.state_dict(), strict=True)
    wrapper16 = wrapper16.to(device).half().eval()

    import onnxruntime as ort
    ort_session = ort.InferenceSession(str(FP32_ONNX),
                                       providers=["CUDAExecutionProvider",
                                                  "CPUExecutionProvider"])
    if "CUDAExecutionProvider" not in ort_session.get_providers():
        raise SystemExit("ONNX Runtime fell back to CPU; refusing to benchmark it as GPU")

    rows = []

    def record(label, backend, precision, batch, latency_ms, throughput,
               peak_alloc=None, peak_reserved=None, device_peak=None, transfer_ms=None):
        rows.append({
            "runtime": label, "backend": backend, "precision": precision,
            "batch_size": batch, "warmup": args.warmup, "measured": args.runs,
            "median_ms": round(statistics.median(latency_ms), 3),
            "p95_ms": round(float(np.percentile(latency_ms, 95)), 3),
            "min_ms": round(min(latency_ms), 3),
            "throughput_patches_per_s": round(throughput, 2),
            "peak_allocated_mib": peak_alloc,
            "peak_reserved_mib": peak_reserved,
            "device_peak_vram_mib": device_peak,
            "h2d_transfer_median_ms": transfer_ms,
            "measurement": "CUDA-synchronised per-iteration wall clock",
        })

    for batch in args.batches:
        for label, model, half in (("torch_cuda_fp32", wrapper, False),
                                   ("torch_cuda_fp16", wrapper16, True)):
            x1 = torch.rand(batch, len(bands), 256, 256, device=device)
            x2 = torch.rand(batch, len(bands), 256, 256, device=device)
            if half:
                x1, x2 = x1.half(), x2.half()
            with torch.no_grad():
                for _ in range(args.warmup):
                    model(x1, x2)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                for _ in range(args.runs):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    model(x1, x2)
                    torch.cuda.synchronize()
                    timings.append((time.perf_counter() - start) * 1000.0)
                peak_alloc = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                peak_reserved = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            # host-to-device cost for the same batch, measured separately
            host1 = torch.rand(batch, len(bands), 256, 256)
            host2 = torch.rand(batch, len(bands), 256, 256)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = host1.to(device, non_blocking=False)
            _ = host2.to(device, non_blocking=False)
            torch.cuda.synchronize()
            transfer = (time.perf_counter() - t0) * 1000.0
            record(label, "torch", "fp16" if half else "fp32", batch, timings,
                   batch / (statistics.median(timings) / 1000.0),
                   peak_alloc, peak_reserved, None, round(transfer, 3))
            del x1, x2, host1, host2
            torch.cuda.empty_cache()

        feed = {"image1": np.random.rand(batch, len(bands), 256, 256).astype(np.float32)}
        feed["image2"] = feed["image1"].copy()
        sampler = GpuSampler()
        sampler.start()
        for _ in range(args.warmup):
            ort_session.run(None, feed)
        timings = []
        for _ in range(args.runs):
            start = time.perf_counter()
            ort_session.run(None, feed)
            timings.append((time.perf_counter() - start) * 1000.0)
        sampler.stop_event.set()
        sampler.join(timeout=2)
        transfer = None
        t0 = time.perf_counter()
        _ = torch.from_numpy(feed["image1"]).to(device)
        _ = torch.from_numpy(feed["image2"]).to(device)
        torch.cuda.synchronize()
        transfer = (time.perf_counter() - t0) * 1000.0
        record("ort_cuda_fp32", "onnxruntime", "fp32", batch, timings,
               batch / (statistics.median(timings) / 1000.0),
               None, None, round(sampler.peak, 1), round(transfer, 3))
        del feed

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "gpu_latency.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT / "gpu_memory.csv", "w", newline="") as handle:
        fields = ["runtime", "batch_size", "peak_allocated_mib", "peak_reserved_mib",
                  "device_peak_vram_mib", "peak_process_rss_mib"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"runtime": row["runtime"], "batch_size": row["batch_size"],
                             "peak_allocated_mib": row["peak_allocated_mib"],
                             "peak_reserved_mib": row["peak_reserved_mib"],
                             "device_peak_vram_mib": row["device_peak_vram_mib"],
                             "peak_process_rss_mib": round(
                                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                                 1)})
    for row in rows:
        print("%-16s b%d median %7.2f ms p95 %7.2f throughput %7.1f patch/s"
              % (row["runtime"], row["batch_size"], row["median_ms"], row["p95_ms"],
                 row["throughput_patches_per_s"]))
    print("wrote", OUT / "gpu_latency.csv", "and", OUT / "gpu_memory.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["eval", "bench"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    {"eval": stage_eval, "bench": stage_bench}[args.stage](args)


if __name__ == "__main__":
    main()
