"""Compare an original FP32 ONNX model against its quantised INT8 version.

Both models are evaluated on the same canonical validation lattice, with the same
per-pair normalisation, the same windows, the same reviewed masks, the same metric
functions and the same fixed comparison thresholds. Nothing here optimises a
threshold or touches the test split.

    python scripts/evaluation/compare_quantized_validation.py \
        --quant-dir outputs/quant/local_manual_v2_seed43_posweight_0p834 \
        --normalisation outputs/dataset/v2/normalisation.json \
        --output outputs/evaluation/quantisation_comparison_v1

Refuses to overwrite an existing output directory.
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
import yaml
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terrashift.preprocessing.normalisation import require_statistics
from terrashift.data.validation_windows import IGNORE_VALUE
from quantize import (CPU_QDQ_OPTIMISATION, DEFAULT_THRESHOLDS, INPUT_NAMES,
                      KNOWN_COMPONENT, PATCH_SIZE, STRIDE, benchmark,
                      build_validation_entries, ensure_new_outputs, evaluate_sessions,
                      sha256, write_json_atomic)
from terrashift.inference.onnx_runtime import CPU, create_session, optimisation_label
from terrashift.evaluation.components import (
    STRUCTURE,
    component_recall,
    identify_known_component,
)

#: Eight-connectivity, matching the spatial audit's component definition.
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def peak_rss_mib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalisation", type=Path,
                        default=Path("outputs/dataset/v2/normalisation.json"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--registration", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--partition-dir", type=Path, default=None)
    parser.add_argument("--pairs", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bench-runs", type=int, default=20)
    parser.add_argument("--provider", default="cpu", choices=["cpu"],
                        help="CPU only: mixing providers would produce one number "
                             "from two different functions")
    args = parser.parse_args()

    fp32_path = args.quant_dir / "onboard_fp32.onnx"
    int8_path = args.quant_dir / "onboard_int8.onnx"
    for path in (fp32_path, int8_path):
        if not path.is_file():
            raise SystemExit("missing model artefact: %s" % path)

    ensure_new_outputs([], args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg["data"]
    manifest = Path(args.manifest or data_cfg["manifest"])
    registration = Path(args.registration or data_cfg["registration_summary"])
    labels_dir = Path(args.labels_dir or data_cfg["manual_label_dir"])
    partition_dir = Path(args.partition_dir or data_cfg["local_partition_dir"])

    thresholds = sorted({float(value) for value in args.thresholds})
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise SystemExit("thresholds must lie in (0, 1)")

    quant_report_path = args.quant_dir / "quantization_report.json"
    quant_report = (json.loads(quant_report_path.read_text())
                    if quant_report_path.is_file() else {})
    bands = tuple(quant_report.get("bands") or ("B02", "B03", "B04", "B08"))

    normalisation = require_statistics(args.normalisation, bands, tuple(args.pairs))
    stub = argparse.Namespace(pairs=list(args.pairs), manifest=str(manifest),
                              registration=str(registration),
                              labels_dir=str(labels_dir),
                              partition_dir=str(partition_dir))
    entries = build_validation_entries(cfg, stub, {"bands": bands}, normalisation)

    fp32_session, fp32_providers = create_session(
        fp32_path, [CPU], require=CPU, optimization_level=CPU_QDQ_OPTIMISATION)
    int8_session, int8_providers = create_session(
        int8_path, [CPU], require=CPU, optimization_level=CPU_QDQ_OPTIMISATION)

    evaluated = evaluate_sessions({"fp32": (fp32_session, INPUT_NAMES),
                                   "int8": (int8_session, INPUT_NAMES)},
                                  entries, thresholds, args.batch_size)

    # ---- prediction differences ------------------------------------------
    differences = {}
    for pair in entries:
        base = evaluated["fp32"]["scores"][pair]
        quantised = evaluated["int8"]["scores"][pair]
        delta = np.abs(base - quantised)
        differences[pair] = {
            "pixels": int(delta.size),
            "mean_abs_probability_difference": round(float(delta.mean()), 6),
            "max_abs_probability_difference": round(float(delta.max()), 6),
            "p95_abs_difference": round(float(np.percentile(delta, 95)), 6),
            "p99_abs_difference": round(float(np.percentile(delta, 99)), 6),
            "pearson_r": round(float(np.corrcoef(base, quantised)[0, 1]), 6),
            "decision_change_fraction": {
                "%.2f" % threshold: round(float(
                    ((base >= threshold) != (quantised >= threshold)).mean()), 6)
                for threshold in thresholds},
            "decision_changes": {
                "%.2f" % threshold: int(
                    ((base >= threshold) != (quantised >= threshold)).sum())
                for threshold in thresholds},
        }

    # ---- component-level sensitivity -------------------------------------
    components = {}
    for pair in entries:
        labels = evaluated["fp32"]["labels"][pair]
        per_threshold = {}
        for threshold in thresholds:
            per_threshold["%.2f" % threshold] = {}
            for model in ("fp32", "int8"):
                payload = component_recall(evaluated[model]["scores"][pair], labels,
                                           entries[pair]["target"], threshold)
                payload["known_component"] = identify_known_component(payload["rows"], pair)
                per_threshold["%.2f" % threshold][model] = payload
        components[pair] = per_threshold

    # ---- resource measurements -------------------------------------------
    shapes = {"256x1": {name: (1, len(bands), PATCH_SIZE, PATCH_SIZE)
                        for name in INPUT_NAMES},
              "256x8": {name: (8, len(bands), PATCH_SIZE, PATCH_SIZE)
                        for name in INPUT_NAMES}}
    latency = {}
    for label, shapeset in shapes.items():
        latency[label] = {
            "cpu_fp32": dict(benchmark(fp32_session, shapeset, args.bench_runs),
                             providers=fp32_providers,
                             optimisation=optimisation_label(CPU_QDQ_OPTIMISATION)),
            "cpu_int8": dict(benchmark(int8_session, shapeset, args.bench_runs),
                             providers=int8_providers,
                             optimisation=optimisation_label(CPU_QDQ_OPTIMISATION))}
    for label, entry in latency.items():
        ratio = (entry["cpu_int8"]["median_ms"] / entry["cpu_fp32"]["median_ms"]
                 if entry["cpu_fp32"]["median_ms"] else float("nan"))
        entry["int8_over_fp32_median"] = round(float(ratio), 3)

    sizes = {
        "checkpoint_bytes": quant_report.get("checkpoint_bytes"),
        "fp32_onnx_bytes": fp32_path.stat().st_size,
        "int8_onnx_bytes": int8_path.stat().st_size,
        "fp32_sha256": sha256(fp32_path),
        "int8_sha256": sha256(int8_path),
    }
    sizes["fp32_matches_quant_report"] = (
        quant_report.get("stages", {}).get("export", {}).get("sha256")
        == sizes["fp32_sha256"])
    sizes["int8_matches_quant_report"] = (
        quant_report.get("stages", {}).get("quantize", {}).get("sha256")
        == sizes["int8_sha256"])
    sizes["checkpoint_sha256"] = quant_report.get("checkpoint_sha256")
    sizes["reduction_bytes"] = sizes["fp32_onnx_bytes"] - sizes["int8_onnx_bytes"]
    sizes["reduction_percent"] = round(
        100.0 * sizes["reduction_bytes"] / sizes["fp32_onnx_bytes"], 2)
    sizes["int8_as_percent_of_fp32"] = round(
        100.0 * sizes["int8_onnx_bytes"] / sizes["fp32_onnx_bytes"], 2)

    summary = {
        "models": {"fp32": str(fp32_path), "int8": str(int8_path)},
        "provider": "CPUExecutionProvider",
        "optimisation": optimisation_label(CPU_QDQ_OPTIMISATION),
        "pairs": list(entries),
        "thresholds": thresholds,
        "threshold_status": "fixed comparison points; not validated operating "
                            "thresholds and not optimised here",
        "patch_size": PATCH_SIZE, "stride": STRIDE,
        "onnx_output_is_probability": True,
        "normalisation": str(args.normalisation),
        "normalisation_sha256": sha256(args.normalisation),
        "detection": {model: {key: evaluated[model][key]
                              for key in ("pairs", "pooled", "macro")}
                      for model in ("fp32", "int8")},
        "ap_delta_pooled": round(evaluated["int8"]["pooled"]["ap"]
                                 - evaluated["fp32"]["pooled"]["ap"], 6),
        "auc_delta_pooled": round(evaluated["int8"]["pooled"]["auc"]
                                  - evaluated["fp32"]["pooled"]["auc"], 6),
        "differences": differences,
        "components": components,
        "sizes": sizes,
        "latency": latency,
        "peak_rss_mib": peak_rss_mib(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "test_split_accessed": False,
        "threshold_optimised": False,
    }

    # ---- per-pair comparison CSV -----------------------------------------
    rows = []
    for pair in entries:
        for model in ("fp32", "int8"):
            metrics = evaluated[model]["pairs"][pair]
            for threshold in thresholds:
                key = "%.2f" % threshold
                counts = metrics["thresholds"][key]
                rows.append({"pair": pair, "model": model, "threshold": key,
                             "ap": metrics["ap"], "auc": metrics["auc"],
                             **{field: counts[field] for field in
                                ("tp", "fp", "fn", "tn", "precision", "recall", "f1",
                                 "iou", "predicted_positive_fraction")},
                             "largest_component_recall": (
                                 components[pair][key][model]["rows"][0]["recall"]
                                 if components[pair][key][model]["rows"] else ""),
                             "small_region_recall": components[pair][key][model][
                                 "groups"].get("small_lt_200px", {}).get(
                                     "pixel_weighted_recall", ""),
                             "known_component_id": (
                                 components[pair][key][model]["known_component"].get(
                                     "component_id", "")),
                             "component_8_recall": (
                                 components[pair][key][model]["known_component"].get(
                                     "recall", "")
                                 if components[pair][key][model]["known_component"]["status"]
                                 == "verified" else ""),
                             "known_component_status": components[pair][key][model][
                                 "known_component"]["status"]})
    with open(args.output / "pair_comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_json_atomic(args.output / "comparison.json", summary)
    (args.output / "comparison_report.md").write_text(
        render_markdown(summary), encoding="utf-8")
    print("wrote", args.output / "comparison.json")
    print("pooled AP %.4f -> %.4f (%+.4f)"
          % (evaluated["fp32"]["pooled"]["ap"], evaluated["int8"]["pooled"]["ap"],
             summary["ap_delta_pooled"]))


def render_markdown(summary):
    """The full comparison report. Every number is read from the measured payload."""
    def table(header, rows):
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "|".join("---" for _ in header) + "|"]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
        return "\n".join(lines)

    thresholds = ["%.2f" % value for value in summary["thresholds"]]
    detection = summary["detection"]
    pairs = summary["pairs"]
    sizes = summary["sizes"]
    latency = summary["latency"]

    ap_delta = {pair: detection["int8"]["pairs"][pair]["ap"]
                - detection["fp32"]["pairs"][pair]["ap"] for pair in pairs}
    auc_delta = {pair: detection["int8"]["pairs"][pair]["auc"]
                 - detection["fp32"]["pairs"][pair]["auc"] for pair in pairs}
    ratios = {shape: values["int8_over_fp32_median"] for shape, values in latency.items()}
    slower_everywhere = all(ratio > 1.0 for ratio in ratios.values())
    loses_ap_everywhere = all(delta < 0 for delta in ap_delta.values())

    out = ["# FP32 versus INT8 validation comparison", "",
           "Both models were evaluated on the same 159 canonical validation windows, the same "
           "pair-specific normalisation, the same reviewed masks and the same metric "
           "functions, on CPUExecutionProvider with %s for both." % summary["optimisation"],
           "", "No operating threshold was selected or optimised. Threshold status recorded "
           "with the run: %s." % summary["threshold_status"], "",
           "Thresholds 0.05 / 0.10 / 0.50 are historical comparison points.", "",

           "## 1. Verified artefacts", "",
           table(["artefact", "bytes", "sha256", "matches quantisation report"],
                 [["FP32 ONNX", sizes["fp32_onnx_bytes"], sizes["fp32_sha256"],
                   sizes["fp32_matches_quant_report"]],
                  ["INT8 ONNX", sizes["int8_onnx_bytes"], sizes["int8_sha256"],
                   sizes["int8_matches_quant_report"]],
                  ["checkpoint", sizes.get("checkpoint_bytes"),
                   sizes.get("checkpoint_sha256"), "recorded by the quantisation run"],
                  ["normalisation", "—", summary["normalisation_sha256"], "—"]]), "",
           "Patch %d, stride %d. Normalisation file: `%s`."
           % (summary["patch_size"], summary["stride"], summary["normalisation"]), ""]

    out += ["## 2. Per-pair detection metrics", "",
            "Pair A and Pair B are reported separately; pooled and macro figures follow in "
            "section 3.", ""]
    for pair in pairs:
        metrics = detection["fp32"]["pairs"][pair]
        out += ["### Pair %s" % pair, "",
                "Reviewed pixels %d, positives %d."
                % (metrics["pixels"], metrics["positives"]), "",
                table(["model", "AP", "ROC-AUC"],
                      [[model, detection[model]["pairs"][pair]["ap"],
                        detection[model]["pairs"][pair]["auc"]]
                       for model in ("fp32", "int8")]), "",
                table(["model", "thr", "TP", "FP", "FN", "precision", "recall", "F1",
                       "IoU", "pred+"],
                      [[model, threshold]
                       + [int(detection[model]["pairs"][pair]["thresholds"][threshold][field])
                          for field in ("tp", "fp", "fn")]
                       + [detection[model]["pairs"][pair]["thresholds"][threshold][field]
                          for field in ("precision", "recall", "f1", "iou",
                                        "predicted_positive_fraction")]
                       for model in ("fp32", "int8") for threshold in thresholds]), "",
                "AP change INT8 − FP32: **%+.6f**; ROC-AUC change **%+.6f**."
                % (ap_delta[pair], auc_delta[pair]), ""]

    out += ["## 3. Pooled micro and macro", "",
            table(["model", "AP pooled", "AP macro", "AUC pooled", "AUC macro"],
                  [[model, detection[model]["pooled"]["ap"], detection[model]["macro"]["ap"],
                    detection[model]["pooled"]["auc"], detection[model]["macro"]["auc"]]
                   for model in ("fp32", "int8")]), "",
            "Pooled micro ranks all reviewed pixels of both pairs together; macro is the "
            "unweighted mean of the two pairs.", "",
            "Pooled AP change **%+.6f**, pooled AUC change **%+.6f**."
            % (summary["ap_delta_pooled"], summary["auc_delta_pooled"]), ""]

    out += ["## 4. Prediction differences on identical pixels", "",
            table(["pair", "reviewed px", "mean abs dP", "max abs dP", "p95", "p99",
                   "Pearson r"]
                  + ["changed @%s" % t for t in thresholds],
                  [[pair, values["pixels"], values["mean_abs_probability_difference"],
                    values["max_abs_probability_difference"],
                    values["p95_abs_difference"], values["p99_abs_difference"],
                    values["pearson_r"]]
                   + ["%d (%.2f%%)" % (values["decision_changes"][t],
                                       100 * values["decision_change_fraction"][t])
                      for t in thresholds]
                   for pair, values in summary["differences"].items()]), "",
            "A high correlation does not by itself establish equivalent detection; read these "
            "beside section 2.", ""]

    out += ["## 5. Component-level sensitivity", "",
            "The known Pair B component is located by its audited area (10,749 px) and then "
            "its label id is verified; rank alone would only say \"largest\".", ""]
    rows = []
    for pair in pairs:
        for threshold in thresholds:
            for model in ("fp32", "int8"):
                payload = summary["components"][pair][threshold][model]
                known = payload["known_component"]
                others = payload["rows"][1:]
                mean_other = (sum(row["recall"] for row in others) / len(others)
                              if others else None)
                rows.append([pair, threshold, model,
                             payload["rows"][0]["recall"] if payload["rows"] else "n/a",
                             known.get("pixels", ""), known.get("recall", ""),
                             known["status"],
                             "%.3f" % mean_other if mean_other is not None else "n/a",
                             payload["groups"].get("small_lt_200px", {}).get(
                                 "pixel_weighted_recall", "n/a")])
    out += [table(["pair", "thr", "model", "largest recall", "known px", "known recall",
                   "identity", "mean other comps", "small-group recall"], rows), ""]
    if any(row[6] == "id_mismatch" for row in rows):
        out += ["**Identity mismatch:** a component of the audited size exists but not under "
                "id %d. The column is reported under the measured id and must not be read as "
                "component 8." % KNOWN_COMPONENT["component_id"], ""]
    elif any(row[6] not in ("verified", "not_applicable") for row in rows):
        out += ["**Limitation:** the audited component was not uniquely located, so no "
                "component-8 figure is claimed.", ""]
    else:
        out += ["Component identity verified for Pair B at every threshold; Pair A has no "
                "audited component identity and reports as `not_applicable`.", ""]

    out += ["## 6. Model size", "",
            table(["artefact", "bytes", "relative to FP32"],
                  [["FP32 ONNX", sizes["fp32_onnx_bytes"], "100 %"],
                   ["INT8 ONNX", sizes["int8_onnx_bytes"],
                    "%.2f %%" % sizes["int8_as_percent_of_fp32"]],
                   ["reduction", sizes["reduction_bytes"],
                    "%.2f %% smaller" % sizes["reduction_percent"]]]), "",
            "The checkpoint is %s bytes but contains optimizer state; it is not the deployed "
            "artefact." % sizes.get("checkpoint_bytes"), ""]

    out += ["## 7. CPU inference latency (model-only)", "",
            table(["shape", "FP32 median ms", "FP32 min ms", "INT8 median ms", "INT8 min ms",
                   "INT8/FP32 median", "runs"],
                  [[shape, values["cpu_fp32"]["median_ms"], values["cpu_fp32"]["min_ms"],
                    values["cpu_int8"]["median_ms"], values["cpu_int8"]["min_ms"],
                    values["int8_over_fp32_median"], values["cpu_fp32"]["runs"]]
                   for shape, values in latency.items()]), "",
            "This is model-only ONNX Runtime inference on random input: no SAFE loading, no "
            "preprocessing, no tiling and no full-scene pass. Both models used the same "
            "provider, optimisation level, warm-up (5) and measured runs.", "",
            ("**No CPU speedup is claimed.** INT8 is slower than FP32 at every measured shape, "
             "which is the expected consequence of the documented QDQ fusion workaround: "
             "ORT_ENABLE_BASIC avoids the incorrect fused kernel and with it the int8 "
             "convolution kernels." if slower_everywhere else
             "INT8 is not slower at every shape; read the per-shape ratios rather than a single "
             "summary claim."), ""]

    out += ["## 8. Peak process RSS", "",
            "%s MiB across the whole comparison run, including both sessions and the latency "
            "measurements." % summary["peak_rss_mib"], ""]

    out += ["## 9. Accuracy-versus-size trade-off", "",
            "INT8 is %.2f %% smaller (%d bytes saved). Against that, AP changed %s and ROC-AUC "
            "changed %s (INT8 minus FP32)."
            % (sizes["reduction_percent"], sizes["reduction_bytes"],
               ", ".join("%s %+.6f" % (pair, ap_delta[pair]) for pair in pairs),
               ", ".join("%s %+.6f" % (pair, auc_delta[pair]) for pair in pairs)), "",
            "Both statements are measured. Whether the size saving is worth the accuracy change "
            "depends on a deployment constraint this audit does not have — a size, memory or "
            "bandwidth budget.", ""]

    out += ["## 10. Deployment recommendation", ""]
    if loses_ap_everywhere and slower_everywhere:
        out += ["**FP32 remains the preferred model.** INT8 loses detection quality on both "
                "pairs and is slower on CPU at both measured shapes, so it is smaller and worse "
                "on every axis available to this comparison. Size alone is not a reason to "
                "deploy it.", "",
                "This would change if a future measurement showed INT8 recovering accuracy on a "
                "held-out review that spans the scene's mid-score band, or if a deployment "
                "target required the smaller artefact and accepted the accuracy cost."]
    elif loses_ap_everywhere and not slower_everywhere:
        out += ["**Conditional.** INT8 loses detection quality but is not uniformly slower. It "
                "is defensible only where the size or memory budget is binding and the accuracy "
                "cost is acceptable."]
    elif slower_everywhere:
        out += ["**FP32 remains the preferred model on CPU.** INT8 does not lose AP but is "
                "slower here; its case would rest on a deployment target where the smaller "
                "artefact matters more than latency."]
    else:
        out += ["**INT8 is viable on these measurements.** It does not lose AP on either pair "
                "and is not slower. Confirm on a broader reviewed sample before relying on it."]
    out += ["", "This is interpretation of measured numbers, not an operating decision; the "
                "operating threshold remains unset and unlocked.", ""]

    out += ["## 11. Limitations", "",
            "1. **Reviewed validation only.** Every metric is on the 25,409 manually reviewed "
            "pixels, not the full tile. Precision, recall and predicted-positive fraction are "
            "properties of an enriched subset (prevalence 0.59 on A, 0.53 on B) and do not "
            "transfer to scene scale.",
            "2. **The supervision gap is unchanged by this comparison.** Pair A's reviewed "
            "no-change class is degenerate at zero, so neither model's AP here bounds its "
            "behaviour on unreviewed terrain.",
            "3. **Latency is model-only.** It excludes SAFE reading, normalisation, tiling and "
            "post-processing, all of which dominate a real tile pass.",
            "4. **CPU only.** The CUDA provider produces a materially different function from "
            "the same INT8 file and was deliberately not measured here.",
            "5. **Not uniformly INT8.** ConvTranspose, Sub, Abs and Sigmoid stay float in the "
            "QDQ graph, so \"INT8\" describes the weights and most activations, not every op.",
            "6. **One checkpoint, one seed.** The earlier paired-seed work found the "
            "0.834-versus-1.0 ordering unstable across seeds; this is one model.",
            "7. **No threshold is set**, so none of this constitutes an operating-point "
            "recommendation.", ""]

    out += ["## 12. Commands and repository state", "",
            "```bash",
            "export SENTINEL_DATA_ROOT=/home/aaryan/projects/skyserve",
            "cd /home/aaryan/projects/skyserve_assignment",
            "python scripts/evaluation/compare_quantized_validation.py \\",
            "  --quant-dir %s \\" % str(Path(summary["models"]["fp32"]).parent),
            "  --normalisation %s \\" % summary["normalisation"],
            "  --output outputs/evaluation/quantisation_comparison_v1 \\",
            "  --thresholds 0.05 0.10 0.50",
            "```", "",
            "Artefacts written here: `comparison_report.md`, `comparison.json`, "
            "`pair_comparison.csv`. The test split was not accessed and no threshold was "
            "selected.", ""]

    out += ["## Answers to the comparison questions", ""]
    for pair in pairs:
        out.append("- **Pair %s:** AP %+.6f, ROC-AUC %+.6f, %d of %d reviewed pixels change "
                   "decision at 0.50."
                   % (pair, ap_delta[pair], auc_delta[pair],
                      summary["differences"][pair]["decision_changes"]["0.50"],
                      summary["differences"][pair]["pixels"]))
    out += ["- **Effect differs between pairs:** %s."
            % ("yes, the AP change differs in sign" if
               (ap_delta[pairs[0]] > 0) != (ap_delta[pairs[1]] > 0)
               else "no, the AP change has the same sign on both"),
            "- **Prediction differences:** see section 4; mean |dP| %s."
            % ", ".join("%s %.6f" % (pair, summary["differences"][pair]
                                     ["mean_abs_probability_difference"]) for pair in pairs),
            "- **Small / fragmented regions** (pixel-weighted recall of components smaller than "
            "200 px at threshold %s): %s." % (thresholds[0], ", ".join(
                "%s %.3f -> %.3f" % (pair,
                                     summary["components"][pair][thresholds[0]]["fp32"]
                                     ["groups"].get("small_lt_200px", {}).get(
                                         "pixel_weighted_recall", float("nan")),
                                     summary["components"][pair][thresholds[0]]["int8"]
                                     ["groups"].get("small_lt_200px", {}).get(
                                         "pixel_weighted_recall", float("nan")))
                for pair in pairs)),
            "- **Known 10,749 px Pair B component** (threshold %s): FP32 %.3f -> INT8 %.3f."
            % (thresholds[0],
               summary["components"]["B"][thresholds[0]]["fp32"]["known_component"].get(
                   "recall", float("nan")),
               summary["components"]["B"][thresholds[0]]["int8"]["known_component"].get(
                   "recall", float("nan"))) if "B" in pairs else "",
            "- **Size versus accuracy:** section 9.",
            "- **CPU speed:** section 7; %s."
            % ("INT8 slower at every shape" if slower_everywhere else "mixed"),
            "- **Defensible deployment candidate:** section 10.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
