"""Validation comparison and end-to-end pipeline benchmark for the integrated GPU path.

Uses the production execution method (model.half() with half inputs via
predict_safe_pair.HalfInputModel) over the 159 canonical validation windows, and times the
pipeline stages separately from the model forward pass.

    python scripts/evaluation/gpu_pipeline_evaluation.py --stage validate
    python scripts/evaluation/gpu_pipeline_evaluation.py --stage bench
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import quantize
from terrashift.inference.gpu import GpuSampler, build_entries
from terrashift.evaluation.components import component_recall, identify_known_component
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "predict_safe_pair", ROOT / "scripts" / "inference" / "predict_safe_pair.py")
_psp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_psp)

OUT = ROOT / "outputs/evaluation/gpu_pipeline_integration_v1"
THRESHOLDS = (0.05, 0.10, 0.50)
CHECKPOINT = ROOT / "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"


def make_model(device, precision):
    _, bands, checkpoint = quantize.load_model(str(CHECKPOINT), "cpu")
    model = quantize.SiameseUNet(in_channels=len(bands), base=int(checkpoint["base"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    if precision == "fp16":
        model = _psp.HalfInputModel(model.half().to(device)).eval()
    return model, bands


class ProductionSession:
    """Mirror the production execution path inside the session.run contract."""

    def __init__(self, model, device, precision):
        self.model = model
        self.device = device
        self.precision = precision

    def run(self, names, feed):
        x1 = torch.from_numpy(feed["image1"]).to(self.device)
        x2 = torch.from_numpy(feed["image2"]).to(self.device)
        with torch.no_grad():
            logits = self.model(x1, x2)
        # The models here are bare SiameseUNet / HalfInputModel, which return logits. The
        # production script applies sigmoid after the forward pass, so do the same here.
        # Skipping this would leave AP/AUC unchanged (they are rank-based) while silently
        # corrupting every threshold-dependent metric.
        return [torch.sigmoid(logits.float()).cpu().numpy()]


def stage_validate(args):
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    entries, bands, checkpoint, _ = build_entries(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [("cpu_fp32_b1", torch.device("cpu"), "fp32", 1)]
    if torch.cuda.is_available():
        configs += [("cuda_fp32_b8", device, "fp32", 8),
                    ("cuda_fp16_b4", device, "fp16", 4),
                    ("cuda_fp16_b8", device, "fp16", 8)]

    results = {}
    for label, run_device, precision, batch in configs:
        model, _ = make_model(run_device, precision)
        session = ProductionSession(model, run_device, precision)
        started = time.perf_counter()
        if run_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        payload = quantize.evaluate_sessions({label: (session, quantize.INPUT_NAMES)},
                                             entries, THRESHOLDS, batch)[label]
        if run_device.type == "cuda":
            payload["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        payload["device"] = str(run_device)
        payload["precision"] = precision
        payload["batch_size"] = batch
        results[label] = payload
        print("[%s] pooled AP %.6f AUC %.6f (%.1f s)"
              % (label, payload["pooled"]["ap"], payload["pooled"]["auc"],
                 payload["elapsed_seconds"]))
        del model
        if run_device.type == "cuda":
            torch.cuda.empty_cache()

    reference = results["cpu_fp32_b1"]
    for label, payload in results.items():
        if label == "cpu_fp32_b1":
            continue
        differences = {}
        for pair in payload["pairs"]:
            base = reference["scores"][pair]
            other = payload["scores"][pair]
            if base.shape != other.shape:
                raise SystemExit("%s pair %s vector length differs" % (label, pair))
            delta = np.abs(base - other)
            differences[pair] = {
                "pixels": int(delta.size),
                "mean_abs_difference": float(delta.mean()),
                "max_abs_difference": float(delta.max()),
                "p95_abs_difference": float(np.percentile(delta, 95)),
                "pearson_r": float(np.corrcoef(base, other)[0, 1]),
                "non_finite": int((~np.isfinite(other)).sum()),
                "decision_changes": {"%.2f" % t: int(((base >= t) != (other >= t)).sum())
                                     for t in THRESHOLDS}}
        payload["difference_vs_cpu_fp32"] = differences

    for label, payload in results.items():
        components = {}
        for pair in entries:
            labels = payload["labels"][pair]
            per_threshold = {}
            for threshold in THRESHOLDS:
                key = "%.2f" % threshold
                data = component_recall(payload["scores"][pair], labels,
                                        entries[pair]["target"], threshold)
                data["known_component"] = identify_known_component(data["rows"], pair)
                per_threshold[key] = data
            components[pair] = per_threshold
        payload["components"] = components

    OUT.mkdir(parents=True, exist_ok=True)
    serialisable = {label: {k: v for k, v in payload.items() if k not in ("scores", "labels")}
                    for label, payload in results.items()}
    (OUT / "pipeline_comparison.json").write_text(json.dumps({
        "thresholds": list(THRESHOLDS),
        "threshold_status": "historical comparison points; not validated operating thresholds",
        "windows": {"A": len(entries["A"]["dataset"]), "B": len(entries["B"]["dataset"])},
        "configurations": serialisable,
        "peak_process_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "test_split_accessed": False,
    }, indent=1) + "\n")
    print("wrote", OUT / "pipeline_comparison.json")


def stage_bench(args):
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    entries, bands, checkpoint, _ = build_entries(cfg)
    device = torch.device("cuda")

    configs = [("cpu_fp32_b1", torch.device("cpu"), "fp32", 1)]
    if torch.cuda.is_available():
        configs += [("cuda_fp16_b4", device, "fp16", 4),
                    ("cuda_fp16_b8", device, "fp16", 8)]

    rows = []
    for label, run_device, precision, batch in configs:
        model, _ = make_model(run_device, precision)
        sampler = GpuSampler()
        sampler.start()
        stages = {"data_s": 0.0, "h2d_s": 0.0, "forward_s": 0.0, "d2h_s": 0.0,
                  "accumulate_s": 0.0}
        windows = 0
        total_start = time.perf_counter()
        for pair, entry in entries.items():
            loader = DataLoader(entry["dataset"], batch_size=batch, shuffle=False,
                                num_workers=0)
            accumulator = quantize.LatticeAccumulator(entry["target"].shape,
                                                      entry["col_offset"])
            # Time the fetch itself: the SAFE reads and the per-pair normalisation happen
            # inside the dataset, so the timer must wrap the retrieval of each batch.
            iterator = iter(loader)
            for _ in range(len(loader)):
                t0 = time.perf_counter()
                batch_data = next(iterator)
                stages["data_s"] += time.perf_counter() - t0
                image1 = batch_data["image1"]
                image2 = batch_data["image2"]
                t1 = time.perf_counter()
                x1 = image1.to(run_device)
                x2 = image2.to(run_device)
                if run_device.type == "cuda":
                    torch.cuda.synchronize()
                stages["h2d_s"] += time.perf_counter() - t1
                t2 = time.perf_counter()
                with torch.no_grad():
                    logits = model(x1, x2)
                if run_device.type == "cuda":
                    torch.cuda.synchronize()
                stages["forward_s"] += time.perf_counter() - t2
                t3 = time.perf_counter()
                probabilities = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
                if run_device.type == "cuda":
                    torch.cuda.synchronize()
                stages["d2h_s"] += time.perf_counter() - t3
                t4 = time.perf_counter()
                valid = batch_data["valid_mask"].numpy()
                for index in range(probabilities.shape[0]):
                    accumulator.add(int(batch_data["row_off"][index]),
                                    int(batch_data["col_off"][index]),
                                    probabilities[index], valid[index],
                                    core_offset=int(batch_data["core_offset"][index]),
                                    core_count=int(batch_data["core_count"][index]))
                stages["accumulate_s"] += time.perf_counter() - t4
                windows += int(probabilities.shape[0])
        if run_device.type == "cuda":
            torch.cuda.synchronize()
        total = time.perf_counter() - total_start
        sampler.stop_event.set()
        sampler.join(timeout=2)
        peak_reserved = (round(torch.cuda.max_memory_reserved() / 2**20, 1)
                         if run_device.type == "cuda" else None)
        rows.append({
            "configuration": label, "device": str(run_device), "precision": precision,
            "batch_size": batch, "windows": windows,
            "total_wall_s": round(total, 3),
            "windows_per_s": round(windows / total, 2),
            "data_read_normalise_s": round(stages["data_s"], 3),
            "h2d_transfer_s": round(stages["h2d_s"], 3),
            "model_forward_s": round(stages["forward_s"], 3),
            "d2h_transfer_s": round(stages["d2h_s"], 3),
            "accumulate_s": round(stages["accumulate_s"], 3),
            "forward_share_of_total": round(stages["forward_s"] / total, 4),
            "peak_reserved_vram_mib": peak_reserved,
            "device_peak_vram_mib": round(sampler.peak, 1) if run_device.type == "cuda" else None,
            "peak_process_rss_mib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        })
        print("%-14s total %6.1f s  %6.1f win/s  forward %5.2f s (%.0f%%)"
              % (label, total, windows / total, stages["forward_s"],
                 100 * stages["forward_s"] / total))
        del model
        if run_device.type == "cuda":
            torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "stage_timings.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT / "memory_usage.csv", "w", newline="") as handle:
        fields = ["configuration", "batch_size", "peak_reserved_vram_mib",
                  "device_peak_vram_mib", "peak_process_rss_mib"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    print("wrote", OUT / "stage_timings.csv", "and", OUT / "memory_usage.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["validate", "bench"])
    args = parser.parse_args()
    {"validate": stage_validate, "bench": stage_bench}[args.stage](args)


if __name__ == "__main__":
    main()
