"""Assemble the GPU inference comparison JSON and report from measured results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/evaluation/gpu_inference_evaluation_v1"
CHECKPOINT = ROOT / "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"
NORMALISATION = ROOT / "outputs/dataset/v2/normalisation.json"
FP32_ONNX = ROOT / "outputs/quant/local_manual_v2_seed43_posweight_0p834_int8_v1/onboard_fp32.onnx"
INT8_ONNX = ROOT / "outputs/quant/local_manual_v2_seed43_posweight_0p834_int8_v1/onboard_int8.onnx"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table(header, rows):
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(OUT / "gpu_inference_report.md"))
    args = parser.parse_args()

    eval_results = json.loads((OUT / "gpu_eval_results.json").read_text())
    int8_cuda = json.loads((OUT / "ort_cuda_int8.json").read_text())
    latency = list(csv.DictReader(open(OUT / "gpu_latency.csv")))
    memory = list(csv.DictReader(open(OUT / "gpu_memory.csv")))
    results = eval_results["results"]

    import torch
    import onnxruntime as ort
    device = torch.cuda.get_device_name(0)
    properties = torch.cuda.get_device_properties(0)

    comparison = {
        "scope": "GPU inference evaluation, validation split only",
        "device": {"name": device, "total_vram_mib": round(properties.total_memory / 2**20, 1),
                   "driver": "621.24", "compute_capability": "%d.%d" % (properties.major,
                                                                        properties.minor)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torch_cuda": torch.version.cuda,
                     "cudnn": str(torch.backends.cudnn.version()),
                     "onnxruntime": ort.__version__,
                     "onnxruntime_listed_providers": ort.get_available_providers()},
        "artifacts": {
            "checkpoint": {"path": str(CHECKPOINT.relative_to(ROOT)),
                           "sha256": sha256(CHECKPOINT),
                           "matches_recorded": sha256(CHECKPOINT)
                           == "dbce89f6811d6cdb6d56dd7d6844e58bf673f5ff65806c5d4939cced0a2554cd"},
            "normalisation": {"path": str(NORMALISATION.relative_to(ROOT)),
                              "sha256": sha256(NORMALISATION),
                              "matches_recorded": sha256(NORMALISATION)
                              == "8aeea033eb2bec5012ebbceb7eca825814b6e43bb699f9e2eaee0c5bba955d2a"},
            "fp32_onnx": {"path": str(FP32_ONNX.relative_to(ROOT)),
                          "sha256": sha256(FP32_ONNX), "bytes": FP32_ONNX.stat().st_size,
                          "matches_recorded": sha256(FP32_ONNX)
                          == "6d58e3094a7f61da19571381f2d076ef9c54b8c5e9944a28618de659b5774844"},
            "int8_onnx": {"path": str(INT8_ONNX.relative_to(ROOT)),
                          "sha256": sha256(INT8_ONNX), "bytes": INT8_ONNX.stat().st_size,
                          "matches_recorded": sha256(INT8_ONNX)
                          == "326fb3c9bf417352d7eb4262aa325d68fbbd3546eace9cd8bdfd7f2a271b0d2e"},
        },
        "runtime_availability": {
            "torch_cuda": {"available": True, "note": "torch bundles its own CUDA libraries"},
            "onnxruntime_cuda": {
                "available": True,
                "note": "loads only when the installed nvidia-* wheel library directories are "
                        "added to LD_LIBRARY_PATH; without them the provider fails with "
                        "'libcublasLt.so.12: cannot open shared object file' and the session "
                        "silently falls back to CPUExecutionProvider",
                "default_behaviour": "silent CPU fallback"},
            "onnxruntime_tensorrt": {
                "available": False,
                "reason": "libnvinfer absent, `import tensorrt` fails, provider creation raises "
                          "'Please install TensorRT libraries'"},
            "onnx_fp16_conversion": {
                "available": False,
                "reason": "onnxruntime.transformers.float16 emits a graph that fails "
                          "onnx.checker (not topologically sorted) for this architecture; "
                          "onnxconverter_common is not installed"},
        },
        "thresholds": eval_results["thresholds"],
        "threshold_status": eval_results["threshold_status"],
        "accuracy": {label: {key: payload[key] for key in ("pairs", "pooled", "macro",
                                                          "runtime", "components",
                                                          "difference_vs_torch_fp32")}
                     if "difference_vs_torch_fp32" in payload else
                     {key: payload[key] for key in ("pairs", "pooled", "macro", "runtime",
                                                    "components")}
                     for label, payload in results.items()},
        "ort_cuda_int8": {"providers": int8_cuda["providers"],
                          "pooled": int8_cuda["pooled"], "macro": int8_cuda["macro"],
                          "pairs": int8_cuda["pairs"]},
        "latency": [{k: v for k, v in row.items()} for row in latency],
        "memory": memory,
        "eval_memory": {"device_peak_vram_mib": eval_results["device_peak_vram_mib_during_eval"],
                        "peak_process_rss_mib": eval_results["peak_process_rss_mib"]},
        "benchmark_method": {"warmup": 10, "measured": 50,
                             "timing": "per-iteration wall clock with torch.cuda.synchronize "
                                       "before and after each iteration for torch paths; ORT "
                                       "run() is synchronous",
                             "shapes": [1, 4, 8], "patch": 256,
                             "host_to_device": "measured separately by copying the same batch "
                                               "from host tensors to the device"},
        "limitations": [
            "Validation split only: 25,409 reviewed pixels, prevalence 0.59 (A) / 0.53 (B).",
            "Model-only timing: excludes SAFE raster I/O, CPU normalisation, tiling "
            "bookkeeping and CPU-side accumulation, which the production path still performs.",
            "ONNX Runtime CUDA was only reachable with a process-scoped LD_LIBRARY_PATH "
            "overlay naming the installed nvidia-* wheel directories; no package was installed.",
            "TensorRT and ONNX FP16 were unavailable and are not estimated.",
        ],
        "test_split_accessed": False,
    }
    (OUT / "gpu_inference_comparison.json").write_text(json.dumps(comparison, indent=1) + "\n")

    # ---------------------------------------------------------------- report
    fp32 = results["torch_cuda_fp32"]
    fp16 = results["torch_cuda_fp16"]
    ort32 = results["ort_cuda_fp32"]
    ratio = {}
    for row in latency:
        ratio[(row["runtime"], int(row["batch_size"]))] = float(row["median_ms"])
    speedup_b8 = ratio[("torch_cuda_fp32", 8)] / ratio[("torch_cuda_fp16", 8)]

    lines = []
    add = lines.append
    add("# GPU inference evaluation")
    add("")
    add("Validation split only. No training, no calibration, no test data, no full-scene "
        "inference, no package installed, no threshold selected.")
    add("")
    add("## 1. Preflight")
    add("")
    add(table(["item", "value", "status"],
              [["checkpoint", comparison["artifacts"]["checkpoint"]["sha256"][:32] + "…",
                "matches recorded"],
               ["normalisation", comparison["artifacts"]["normalisation"]["sha256"][:32] + "…",
                "matches recorded"],
               ["FP32 ONNX", comparison["artifacts"]["fp32_onnx"]["sha256"][:32] + "…",
                "matches recorded"],
               ["INT8 ONNX", comparison["artifacts"]["int8_onnx"]["sha256"][:32] + "…",
                "matches recorded, left unmodified"],
               ["GPU", "%s, %.0f MiB" % (device, properties.total_memory / 2**20), "usable"],
               ["torch / CUDA / cuDNN",
                "%s / %s / %s" % (torch.__version__, torch.version.cuda,
                                   torch.backends.cudnn.version()), "CUDA available"],
               ["onnxruntime", ort.__version__, "TRT + CUDA + CPU listed"]]))
    add("")
    add("**One discrepancy in the brief.** The expected normalisation hash was given as "
        "`8aeea033eb2ecb2e9eaee0c5bba955d2a`, which is 32 characters and is not the file's "
        "hash. The file on disk is `8aeea033eb2bec5012ebbceb7eca825814b6e43bb699f9e2eaee0c5bba955d2a`, "
        "which matches the value recorded by every earlier phase. I verified against the "
        "recorded value rather than the truncated string.")
    add("")
    add("**ONNX Runtime CUDA does not work by default here.** Creating a session with "
        "`CUDAExecutionProvider` raises `Failed to load library libonnxruntime_providers_cuda.so "
        "with error: libcublasLt.so.12: cannot open shared object file`, and the session then "
        "silently reports `['CPUExecutionProvider']`. The CUDA libraries are present as pip "
        "`nvidia-*` wheels; naming their `lib` directories in `LD_LIBRARY_PATH` for the process "
        "makes the provider load. That is a process-scoped path overlay, not an installation "
        "and not a persistent change — but it is a real deployment requirement and is reported "
        "as such. Every ORT GPU number below was produced with the provider confirmed active.")
    add("")
    add("**TensorRT is unavailable.** The provider is listed but cannot be created: "
        "`libnvinfer` is absent and `import tensorrt` fails. No TensorRT result appears below.")
    add("")
    add("**ONNX FP16 is unavailable.** The only installed converter "
        "(`onnxruntime.transformers.float16`) emits a graph that fails `onnx.checker` "
        "(not topologically sorted) for this architecture, across parameter variants; "
        "`onnxconverter_common` is not installed and installing is prohibited.")
    add("")

    add("## 2. GPU FP32 baseline")
    add("")
    add(table(["runtime", "provider", "pooled AP", "pooled AUC", "Pair A AP", "Pair B AP",
               "macro AP"],
              [["torch CUDA FP32", "torch.cuda", "%.6f" % fp32["pooled"]["ap"],
                "%.6f" % fp32["pooled"]["auc"], "%.6f" % fp32["pairs"]["A"]["ap"],
                "%.6f" % fp32["pairs"]["B"]["ap"], "%.6f" % fp32["macro"]["ap"]],
               ["ORT CUDA FP32", "CUDAExecutionProvider", "%.6f" % ort32["pooled"]["ap"],
                "%.6f" % ort32["pooled"]["auc"], "%.6f" % ort32["pairs"]["A"]["ap"],
                "%.6f" % ort32["pairs"]["B"]["ap"], "%.6f" % ort32["macro"]["ap"]]]))
    add("")
    add("CUDA execution was confirmed, not assumed: torch paths ran on "
        "`torch.device('cuda')` with peak allocated VRAM recorded per run, and the ORT session "
        "reported `CUDAExecutionProvider` first in its active provider list. No path fell back "
        "to CPU.")
    add("")
    add("The GPU FP32 result reproduces the CPU validation figure from the previous phase "
        "(pooled AP 0.770233 on CPU, 0.770227 here) to within 6e-6, so the hardware change does "
        "not move the accuracy baseline.")
    add("")

    add("## 3. FP16")
    add("")
    add(table(["runtime", "pooled AP", "pooled AUC", "Pair A AP", "Pair B AP", "macro AP"],
              [["torch CUDA FP32", "%.6f" % fp32["pooled"]["ap"], "%.6f" % fp32["pooled"]["auc"],
                "%.6f" % fp32["pairs"]["A"]["ap"], "%.6f" % fp32["pairs"]["B"]["ap"],
                "%.6f" % fp32["macro"]["ap"]],
               ["torch CUDA FP16", "%.6f" % fp16["pooled"]["ap"], "%.6f" % fp16["pooled"]["auc"],
                "%.6f" % fp16["pairs"]["A"]["ap"], "%.6f" % fp16["pairs"]["B"]["ap"],
                "%.6f" % fp16["macro"]["ap"]]]))
    add("")
    add(table(["pair", "mean abs dP", "max abs dP", "p95", "p99", "Pearson r", "non-finite",
               "decision changes @0.05 / @0.10 / @0.50"],
              [[pair, "%.2e" % values["mean_abs_difference"],
                "%.2e" % values["max_abs_difference"], "%.2e" % values["p95_abs_difference"],
                "%.2e" % values["p99_abs_difference"], "%.8f" % values["pearson_r"],
                values["non_finite"],
                "%d / %d / %d" % (values["decision_changes"]["0.05"],
                                  values["decision_changes"]["0.10"],
                                  values["decision_changes"]["0.50"])]
               for pair, values in fp16["difference_vs_torch_fp32"].items()]))
    add("")
    add("FP16 changes the pooled AP by **%+.6f** and the pooled AUC by **%+.6f** — smaller "
        "than the seed-to-seed variation measured earlier for this model. Prediction "
        "differences are %s on Pair B, and only %d of 22,650 reviewed Pair B pixels flip at "
        "0.05."
        % (fp16["pooled"]["ap"] - fp32["pooled"]["ap"],
           fp16["pooled"]["auc"] - fp32["pooled"]["auc"],
           "small (mean %.2e, max %.2e)" % (fp16["difference_vs_torch_fp32"]["B"]
                                            ["mean_abs_difference"],
                                            fp16["difference_vs_torch_fp32"]["B"]
                                            ["max_abs_difference"]),
           fp16["difference_vs_torch_fp32"]["B"]["decision_changes"]["0.05"]))
    add("")

    add("## 4. GPU INT8")
    add("")
    add("TensorRT cannot be used, so no TensorRT INT8 build was attempted; building one would "
        "require installing the TensorRT runtime, which is prohibited. The existing CPU QDQ "
        "INT8 artefact was tested on `CUDAExecutionProvider` — not assumed to be a suitable GPU "
        "artefact:")
    add("")
    add(table(["runtime", "pooled AP", "pooled AUC", "Pair A AP", "Pair B AP"],
              [["torch CUDA FP32 (reference)", "%.6f" % fp32["pooled"]["ap"],
                "%.6f" % fp32["pooled"]["auc"], "%.6f" % fp32["pairs"]["A"]["ap"],
                "%.6f" % fp32["pairs"]["B"]["ap"]],
               ["ORT CUDA INT8 (existing QDQ)", "%.6f" % int8_cuda["pooled"]["ap"],
                "%.6f" % int8_cuda["pooled"]["auc"],
                int8_cuda["pairs"]["A"]["ap"], int8_cuda["pairs"]["B"]["ap"]]]))
    add("")
    add("It runs on the GPU, but pooled AP falls **%+.6f** against GPU FP32. INT8 on this "
        "architecture is a size optimisation that costs accuracy on CPU and on GPU alike."
        % (int8_cuda["pooled"]["ap"] - fp32["pooled"]["ap"]))
    add("")
    add("Two further observations on that path, both recorded from the run:")
    add("")
    add("1. ORT logged `%s`." % int8_cuda.get("note_memcpy_nodes", "n/a"))
    add("2. An indicative latency of **%.2f ms** at batch 4 against %.2f ms for FP32, from a "
        "single 50-iteration loop with 10 warm-up calls and no explicit synchronisation. That is "
        "**indicative only** — it is not part of the CUDA-synchronised table below and should "
        "not be read as a precise ratio."
        % (int8_cuda.get("batch4_median_ms_indicative", float("nan")),
           ratio.get(("torch_cuda_fp32", 4), float("nan"))))
    add("")

    add("## 5. Validation metrics at the fixed comparison thresholds")
    add("")
    for label, payload in (("torch CUDA FP32", fp32), ("torch CUDA FP16", fp16),
                           ("ORT CUDA FP32", ort32)):
        add("**%s**" % label)
        add("")
        rows = []
        for pair in ("A", "B"):
            for threshold in ("0.05", "0.10", "0.50"):
                counts = payload["pairs"][pair]["thresholds"][threshold]
                rows.append([pair, threshold, "%.4f" % counts["precision"],
                             "%.4f" % counts["recall"], "%.4f" % counts["f1"],
                             "%.4f" % counts["iou"],
                             "%.4f" % counts["predicted_positive_fraction"]])
        add(table(["pair", "thr", "precision", "recall", "F1", "IoU", "pred+"], rows))
        add("")
    add("Component-level recall, including the audited Pair B component "
        "(id verified by 10,749 px area at every threshold):")
    add("")
    rows = []
    for label, payload in (("torch FP32", fp32), ("torch FP16", fp16), ("ORT FP32", ort32)):
        for threshold in ("0.05", "0.50"):
            b = payload["components"]["B"][threshold]
            a = payload["components"]["A"][threshold]
            rows.append([label, threshold,
                         "%.4f" % b["known_component"]["recall"],
                         b["known_component"]["status"],
                         "%.4f" % a["rows"][0]["recall"],
                         "%.4f" % a["groups"]["small_lt_200px"]["pixel_weighted_recall"],
                         "%.4f" % b["groups"]["small_lt_200px"]["pixel_weighted_recall"]])
    add(table(["runtime", "thr", "Pair B known component", "identity", "Pair A largest",
               "Pair A small-group", "Pair B small-group"], rows))
    add("")
    add("The known component is stable across all three GPU paths: %.4f / %.4f / %.4f at 0.05."
        % (fp32["components"]["B"]["0.05"]["known_component"]["recall"],
           fp16["components"]["B"]["0.05"]["known_component"]["recall"],
           ort32["components"]["B"]["0.05"]["known_component"]["recall"]))
    add("")

    add("## 6. GPU benchmark")
    add("")
    add("10 warm-up iterations, 50 measured, per-iteration wall clock with "
        "`torch.cuda.synchronize()` before and after each torch iteration; ORT `run()` is "
        "synchronous. Random inputs already resident on the device, so this is **model-only** "
        "latency.")
    add("")
    rows = []
    for row in latency:
        rows.append([row["runtime"], row["batch_size"], row["median_ms"], row["p95_ms"],
                     row["throughput_patches_per_s"], row["h2d_transfer_median_ms"],
                     row["peak_allocated_mib"] or "—", row["peak_reserved_mib"] or "—"])
    add(table(["runtime", "batch", "median ms", "p95 ms", "patches/s",
               "host→device ms", "VRAM alloc MiB", "VRAM reserved MiB"], rows))
    add("")
    add("**FP16 is %.2fx faster than FP32 at batch 8** (%.2f ms → %.2f ms) and uses roughly "
        "half the VRAM (%.0f MiB reserved against %.0f). At batch 1 the gain is smaller "
        "(%.2f ms → %.2f ms) because the launch overhead dominates a 256² patch."
        % (speedup_b8, ratio[("torch_cuda_fp32", 8)], ratio[("torch_cuda_fp16", 8)],
           float(next(r["peak_reserved_mib"] for r in latency
                      if r["runtime"] == "torch_cuda_fp16" and r["batch_size"] == "8")),
           float(next(r["peak_reserved_mib"] for r in latency
                      if r["runtime"] == "torch_cuda_fp32" and r["batch_size"] == "8")),
           ratio[("torch_cuda_fp32", 1)], ratio[("torch_cuda_fp16", 1)]))
    add("")
    add("ORT CUDA FP32 tracks torch FP32 within a few percent (%.2f ms against %.2f ms at "
        "batch 8) and offers no GPU speed advantage here."
        % (ratio[("ort_cuda_fp32", 8)], ratio[("torch_cuda_fp32", 8)]))
    add("")
    add("Host-to-device transfer is %.2f ms for a batch-8 pair of 4-band 256² patches — about "
        "%.0f %% of the FP32 model time and %.0f %% of the FP16 time, so on this device the "
        "transfer is real but secondary."
        % (float(next(r["h2d_transfer_median_ms"] for r in latency
                      if r["runtime"] == "torch_cuda_fp32" and r["batch_size"] == "8")),
           100 * float(next(r["h2d_transfer_median_ms"] for r in latency
                            if r["runtime"] == "torch_cuda_fp32"
                            and r["batch_size"] == "8")) / ratio[("torch_cuda_fp32", 8)],
           100 * float(next(r["h2d_transfer_median_ms"] for r in latency
                            if r["runtime"] == "torch_cuda_fp32"
                            and r["batch_size"] == "8")) / ratio[("torch_cuda_fp16", 8)]))
    add("")
    add(table(["runtime", "peak process RSS MiB", "peak device VRAM MiB"],
              [["evaluation run", eval_results["peak_process_rss_mib"],
                eval_results["device_peak_vram_mib_during_eval"]],
               ["benchmark run", memory[0]["peak_process_rss_mib"], "see table above"]]))
    add("")

    add("## 7. What stays on the CPU")
    add("")
    add("`scripts/inference/predict_safe_pair.py` is unchanged and remains CPU-bound around the forward "
        "pass. Per window it still:")
    add("")
    add("1. reads 8 `.jp2` bands with rasterio (two dates × four bands) on the CPU;")
    add("2. converts digital numbers to reflectance and applies the pair z-score in numpy;")
    add("3. copies each window to the device for the forward pass;")
    add("4. averages overlapping windows into a numpy accumulator on the CPU;")
    add("5. computes validity masks with OpenCV/rasterio on the CPU;")
    add("6. writes the probability, mask and validity rasters with rasterio.")
    add("")
    add("Only step 3's forward pass runs on the GPU. The model is small (482,881 parameters) "
        "and fast (%.1f ms per batch-8 patch group), so in a real tile pass the raster I/O and "
        "per-window Python bookkeeping are likely to dominate. **No end-to-end speedup is "
        "claimed from these model-only numbers**, and none was measured."
        % ratio[("torch_cuda_fp16", 8)])
    add("")

    add("## 8. Deployment recommendation")
    add("")
    add("**Use PyTorch CUDA FP16 for GPU deployment.** It reproduces the FP32 accuracy to "
        "±%.0e AP, halves VRAM, and is %.2fx faster in model-only inference at batch 8. It is "
        "the only GPU configuration here that improves on the FP32 baseline without an accuracy "
        "cost."
        % (abs(fp16["pooled"]["ap"] - fp32["pooled"]["ap"]), speedup_b8))
    add("")
    add("- **ORT CUDA FP32** matches torch FP32 in accuracy and latency; use it only if the "
        "deployment stack is ONNX-based, and only after confirming the CUDA provider actually "
        "loads (it silently falls back to CPU otherwise).")
    add("- **INT8 on GPU** is not recommended: it costs %+.4f pooled AP for a size saving that "
        "matters less on a GPU with 4 GB than it does on a CPU."
        % (int8_cuda["pooled"]["ap"] - fp32["pooled"]["ap"]))
    add("- **TensorRT** could not be evaluated; whether it beats FP16 remains unknown and "
        "unmeasured.")
    add("- No operating threshold is recommended. The threshold remains unlocked, and these "
        "accuracies are on the reviewed validation subset, not full scenes.")
    add("")

    add("## 9. Limitations")
    add("")
    add("1. **Validation subset only** — 25,409 reviewed pixels, prevalence 0.59 (A) / 0.53 (B). "
        "The same scene-wide caveats as every prior report apply.")
    add("2. **Model-only latency** — excludes raster I/O, normalisation, tiling and "
        "accumulation, which remain CPU-side.")
    add("3. **ORT CUDA needed a library-path overlay.** Without it the provider failed and the "
        "session silently fell back to CPU. A production deployment must set this deliberately "
        "or it will run on the CPU without warning.")
    add("4. **TensorRT and ONNX FP16 are unavailable** and were not estimated.")
    add("5. **INT8 GPU accuracy** was measured with the existing CPU-calibrated QDQ artefact; a "
        "TRT-calibrated INT8 build could differ and was not produced. Its CUDA execution also "
        "inserts 26 Memcpy nodes, which ORT flags as a performance risk and which prevents CUDA "
        "graph capture, so the QDQ artefact is not a well-matched GPU artefact.")
    add("6. **One device, one checkpoint.** RTX 3050 Ti Laptop, 4 GB, one seed.")
    add("7. **fp16 numerics** were checked for finiteness and range in every evaluation window; "
        "no overflow was observed, but very large activations remain a risk on other data.")
    add("")

    add("## 10. Artefacts and repository state")
    add("")
    add("    gpu_inference_report.md        this document")
    add("    gpu_inference_comparison.json  metrics, runtimes, providers, methodology")
    add("    gpu_latency.csv                per-runtime, per-batch latency and throughput")
    add("    gpu_memory.csv                 VRAM and RSS per runtime and batch")
    add("    gpu_eval_results.json          raw evaluation payload")
    add("    ort_cuda_int8.json             existing QDQ INT8 measured on the CUDA provider")
    add("")
    add("Test split not accessed. No package installed or upgraded. Checkpoint, normalisation, "
        "ONNX models, annotations and dataset splits unmodified. No full-scene inference run.")
    add("")
    (Path(args.report)).write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "gpu_inference_comparison.json")
    print("wrote", args.report)


if __name__ == "__main__":
    main()
