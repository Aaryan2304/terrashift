"""Export, verify, quantise and benchmark the onboard candidate model.

Five stages, each of which can be run on its own:

    export    PyTorch checkpoint -> ONNX FP32 with dynamic spatial axes
    parity    torch vs ONNX Runtime FP32 on the validation windows
    quantize  static QDQ INT8, calibrated on TRAIN split patches only
    compare   FP32 vs INT8 accuracy on validation, at fixed comparison thresholds
    bench     size and latency for CPU FP32, CPU INT8, and CUDA where available

Two rules this file enforces:

* Calibration reads the train split only. Using validation or test data to fit
  the quantisation ranges would be fitting a hyperparameter on the evaluation
  set.
* The test split is not touched. Everything here runs on validation, so the
  locked test regions are still only used by ``eval.py``.

Two data paths are supported. ``--dataset local_manual`` (the default) calibrates
on the local SAFE training patches and evaluates on the canonical validation
lattice with per-pair reflectance normalisation. ``--dataset oscd_original`` keeps
the original OSCD patch path for the historical comparisons already in
``outputs/quant/``.

The exported graph ends in a sigmoid, so a session returns a PROBABILITY. The
torch wrapper used here does the same, which is why nothing in this file applies
sigmoid to a model output.

Usage:
    python quantize.py --checkpoint <best.pt> \
        --normalisation outputs/dataset/v2/normalisation.json \
        --dataset local_manual --pairs A B \
        --output-dir outputs/quant/<name>
Nothing is overwritten unless --allow-overwrite is passed deliberately.

TensorRT is reported but not used: the runtime library is absent on this
machine, so the TensorRT provider cannot be created even though ONNX Runtime
lists it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent

from terrashift.data.local_patches import BANDS as LOCAL_BANDS
from terrashift.data.local_patches import LocalManualPatchDataset, pair_image_loader
from terrashift.data.manifest import load_manifest
from terrashift.preprocessing.normalisation import require_statistics
from terrashift.data.oscd_original import DEFAULT_BANDS, OSCDOriginalPatchDataset
from terrashift.data.validation_windows import (IGNORE_VALUE, ValidationLatticeDataset,
                                     read_validation_target)
from terrashift.models.siamese_unet import SiameseUNet
from onnxruntime.quantization import CalibrationDataReader
from terrashift.evaluation.lattice import LatticeAccumulator
from terrashift.metrics import average_precision, metrics_from_counts, roc_auc
from terrashift.inference.onnx_runtime import (CPU, CUDA, create_session, optimisation_label)

# Reusable implementations live in the package; these names stay importable from here
# because existing scripts do ``from quantize import load_model``.
from terrashift.inference.checkpoints import Wrapper, load_model
from terrashift.inference.export import export_dummy
from terrashift.inference.quantization import CalibrationReader

# Evaluation helpers now live in terrashift.evaluation; these names stay importable
# from here because existing scripts and reports import them from quantize.
from terrashift.evaluation.prediction import (predict_pair_onnx, predict_pair_torch,
                                             session_probabilities, valid_of)
from terrashift.evaluation.sessions import evaluate_ort, evaluate_sessions
from terrashift.evaluation.thresholds import metrics_at_thresholds



# QDQ models on CPU must not use the default optimiser: at ORT_ENABLE_EXTENDED
# and above ORT fuses Conv into QLinearConv, and the fused kernel reuses the
# int32 bias with x_scale*w_scale instead of the bias's own scale. Any
# convolution whose bias quantisation does not satisfy that invariant is then
# silently wrong. This model's second Siamese branch hits it. See
# docs/decisions/009-cpu-qdq-optimiser-defect.md.
CPU_QDQ_OPTIMISATION = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

#: The deployed patch size. The old 128 dummy exercised a shape inference never sees.
PATCH_SIZE = 256
#: Canonical inference stride shared by training, validation and predict_safe_pair.
STRIDE = 128
#: Comparison points carried over from earlier experiments. These are NOT validated
#: operating thresholds, and nothing here is allowed to optimise them.
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.50)
INPUT_NAMES = ["image1", "image2"]
#: The exported graph ends in sigmoid, so its output is already a probability.
ONNX_OUTPUT_IS_PROBABILITY = True
#: Conversion-integrity gate. The ONNX FP32 graph must reproduce the torch model to
#: this tolerance before it is legitimate to quantise it. This is NOT an operating
#: threshold or a detection-performance criterion.
PARITY_TOLERANCE = 1e-5
#: Stable identity of the Pair B change component that the spatial audit measured at
#: 10,749 px. Rank cannot be used as an identity: it only says "largest".
# Single source: the package owns this value; re-exported here so existing
# callers of ``quantize.KNOWN_COMPONENT`` keep the same object.
from terrashift.evaluation.components import KNOWN_COMPONENT
OUTPUT_NAME = "probability"




def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()




# --------------------------------------------------------------------------- data


def make_dataset(cfg, split, checkpoint, batch_size):
    """OSCD patch loader. Kept for the historical comparisons."""
    dataset = OSCDOriginalPatchDataset(
        Path(cfg["data"]["oscd_patch_index"]), split,
        tuple(checkpoint.get("bands", DEFAULT_BANDS)),
        checkpoint["mean"], checkpoint["std"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def calibration_plan(rows, batch_size, batches, seed=43):
    """Deterministically choose train-split rows, with both pairs represented.

    Proportional allocation by largest remainder, at least one row per pair, then a
    fixed-seed shuffle. Returns ``(indices, counts_by_pair)``.
    """
    if batch_size < 1 or batches < 1:
        raise ValueError("batch_size and batches must be positive")
    needed = int(batch_size) * int(batches)
    if needed > len(rows):
        raise ValueError("calibration asks for %d rows but the train split has %d"
                         % (needed, len(rows)))
    by_pair = {}
    for index, row in enumerate(rows):
        by_pair.setdefault(row["pair"], []).append(index)
    pairs = sorted(by_pair)
    if not pairs:
        raise ValueError("no train rows to calibrate on")
    if len(pairs) > needed:
        raise ValueError("cannot represent every pair in %d calibration rows" % needed)

    total = len(rows)
    exact = {pair: needed * len(by_pair[pair]) / total for pair in pairs}
    counts = {pair: max(1, int(np.floor(exact[pair]))) for pair in pairs}
    while sum(counts.values()) > needed:
        largest = max((pair for pair in pairs if counts[pair] > 1),
                      key=lambda pair: counts[pair], default=None)
        if largest is None:
            raise ValueError("cannot allocate %d rows across %d pairs"
                             % (needed, len(pairs)))
        counts[largest] -= 1
    while sum(counts.values()) < needed:
        pair = max(pairs, key=lambda name: exact[name] - counts[name])
        if counts[pair] >= len(by_pair[pair]):
            break
        counts[pair] += 1
    if sum(counts.values()) != needed:
        raise ValueError("calibration allocation produced %d of %d rows"
                         % (sum(counts.values()), needed))

    rng = random.Random(seed)
    indices = []
    for pair in pairs:
        indices.extend(rng.sample(sorted(by_pair[pair]), counts[pair]))
    rng.shuffle(indices)
    return indices, counts


def make_local_calibration_loader(cfg, checkpoint, args, normalisation):
    """Calibration loader over the local SAFE train patches.

    ``normalisation`` is the per-pair statistics payload. The dataset applies the
    pair's own statistics, so mean/std are deliberately not passed as well: doing
    both would normalise the inputs twice.
    """
    dataset = LocalManualPatchDataset(
        index_path=Path(cfg["data"]["local_patch_index"]),
        split="train",
        manifest_path=Path(args.manifest),
        registration_path=Path(args.registration),
        normalisation=normalisation,
        bands=tuple(checkpoint.get("bands", LOCAL_BANDS)))
    indices, counts = calibration_plan(dataset.rows, args.batch_size,
                                       args.calibration_batches,
                                       seed=args.calibration_seed)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    plan = {"rows_available": len(dataset.rows), "rows_selected": len(indices),
            "per_pair": counts, "seed": args.calibration_seed, "split": "train",
            "batch_size": args.batch_size, "batches": args.calibration_batches,
            "dataset": "local_manual"}
    return loader, plan


def build_validation_entries(cfg, args, checkpoint, normalisation):
    """Canonical validation lattice, one entry per requested pair."""
    bands = tuple(checkpoint.get("bands", LOCAL_BANDS))
    loader = pair_image_loader(args.manifest, args.registration, bands=bands)
    entries = {}
    for pair in args.pairs:
        labels = Path(args.labels_dir) / ("pair_%s_v2" % pair.lower()) / \
            ("manual_labels_%s.tif" % pair)
        partition = Path(args.partition_dir) / ("pair_%s_partition_10m.tif" % pair)
        target, col_offset, tile_width = read_validation_target(labels, partition)
        entries[pair] = {
            "target": target, "col_offset": col_offset, "labels_path": str(labels),
            "dataset": ValidationLatticeDataset(
                [pair], target, col_offset, tile_width, PATCH_SIZE, STRIDE,
                image_loader=lambda r, c, s, _p=pair: loader(_p, r, c, s),
                normalisation=normalisation),
        }
    return entries


# ------------------------------------------------------------------ evaluation






















# ---------------------------------------------------------------------- outputs


def ensure_new_outputs(paths, directory=None, allow_overwrite=False):
    """Refuse to clobber anything this run did not create.

    A partial overwrite is worse than a failure: it leaves a directory that looks
    like a finished run. The directory must therefore be absent or empty, and no
    target file may exist.
    """
    if allow_overwrite:
        return
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing:
        raise SystemExit("refusing to overwrite existing outputs: %s"
                         % ", ".join(existing))
    if directory is not None:
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise SystemExit("refusing to reuse non-empty output directory: %s"
                             % directory)


def write_json_atomic(path, payload):
    """Write the report only once every stage has finished, via a rename."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)




def dynamic_axes_for(input_names, output_name=OUTPUT_NAME):
    """Declare batch/height/width dynamic on BOTH inputs and the output.

    Leaving the output fixed pins the graph to whatever the export dummy was, which is
    how a 128 dummy once produced a graph whose output extent did not match the
    deployed 256 patch.
    """
    axes = {name: {0: "batch", 2: "height", 3: "width"} for name in input_names}
    axes[output_name] = {0: "batch", 2: "height", 3: "width"}
    return axes


def check_parity_gate(report, tolerance=PARITY_TOLERANCE):
    """Refuse to quantise a graph that has not proved it matches the torch model.

    Four conditions, all of them conversion-integrity checks: the parity stage ran, the
    measured difference is finite and within tolerance, no prediction is NaN or Inf, and
    the two score vectors have the same shape and ordering.
    """
    stages = (report or {}).get("stages") or {}
    parity = stages.get("parity")
    if not parity:
        raise SystemExit("parity stage has not run; refusing to quantise without it")
    value = parity.get("max_abs_difference")
    if value is None:
        raise SystemExit("parity stage recorded no max_abs_difference; refusing to quantise")
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit("parity produced a non-finite difference (%r); refusing to quantise"
                         % value)
    if value > tolerance:
        raise SystemExit("parity difference %.3e exceeds the %.1e tolerance; refusing to "
                         "quantise" % (value, tolerance))
    non_finite = int(parity.get("non_finite_torch", 0)) + int(parity.get("non_finite_onnx", 0))
    if non_finite:
        raise SystemExit("parity produced %d NaN/Inf predictions; refusing to quantise"
                         % non_finite)
    if parity.get("scores_match") is False:
        raise SystemExit("torch and ONNX scores differ in shape or ordering; refusing to "
                         "quantise")
    return {"status": "passed", "max_abs_difference": value, "tolerance": tolerance,
            "non_finite": non_finite, "scores_match": True}


def verify_export_contract(session, input_names, bands, output_name=OUTPUT_NAME,
                           batch_sizes=(1, 8), size=PATCH_SIZE):
    """Runtime contract check for a freshly exported graph. Not run in this task."""
    inputs = {value.name: value for value in session.get_inputs()}
    outputs = [value.name for value in session.get_outputs()]
    checks = {"graph_inputs": sorted(inputs), "graph_outputs": outputs,
              "batches": {}, "problems": []}
    missing = [name for name in input_names if name not in inputs]
    if missing:
        checks["problems"].append("missing graph inputs: %s" % missing)
    if output_name not in outputs:
        checks["problems"].append("missing graph output: %s" % output_name)
    for batch in batch_sizes:
        feed = {name: np.random.rand(batch, len(bands), size, size).astype(np.float32)
                for name in input_names}
        output = np.asarray(session.run(None, feed)[0])
        expected = (batch, 1, size, size)
        finite = bool(np.isfinite(output).all())
        within = bool(output.min() >= 0.0 and output.max() <= 1.0)
        entry = {"shape": list(output.shape), "expected_shape": list(expected),
                 "shape_ok": tuple(output.shape) == expected,
                 "finite": finite, "in_probability_range": within,
                 "min": float(output.min()), "max": float(output.max())}
        checks["batches"]["batch_%d" % batch] = entry
        if not (entry["shape_ok"] and finite and within):
            checks["problems"].append("batch %d failed the output contract" % batch)
    checks["passed"] = not checks["problems"]
    return checks


def benchmark(session, input_shapes, runs, warmup=5):
    feed = {name: np.random.rand(*shape).astype(np.float32)
            for name, shape in input_shapes.items()}
    for _ in range(warmup):
        session.run(None, feed)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, feed)
        timings.append((time.perf_counter() - start) * 1000.0)
    return {"median_ms": round(statistics.median(timings), 2),
            "min_ms": round(min(timings), 2),
            "max_ms": round(max(timings), 2),
            "runs": runs}


def bench_shapes(bands):
    """The deployed shape at two batch sizes, plus a labelled diagnostic shape."""
    return {"256x1": {"image1": (1, len(bands), 256, 256),
                      "image2": (1, len(bands), 256, 256)},
            "256x8": {"image1": (8, len(bands), 256, 256),
                      "image2": (8, len(bands), 256, 256)},
            "128x1_diagnostic": {"image1": (1, len(bands), 128, 128),
                                 "image2": (1, len(bands), 128, 128)}}


# ------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--dataset", choices=["local_manual", "oscd_original"],
                        default="local_manual")
    parser.add_argument("--pairs", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--normalisation", type=Path, default=None,
                        help="per-pair statistics JSON; required for local_manual")
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--partition-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--registration", type=Path, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=list(DEFAULT_THRESHOLDS),
                        help="fixed comparison thresholds; never optimised")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--calibration-batches", type=int, default=20)
    parser.add_argument("--calibration-seed", type=int, default=43)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--bench-runs", type=int, default=20)
    parser.add_argument("--allow-overwrite", action="store_true", default=False,
                        help="deliberate escape hatch for replacing this run's own "
                             "outputs; disabled by default")
    parser.add_argument("--stages", nargs="*",
                        default=["export", "verify", "parity", "quantize",
                                 "compare", "bench"])
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg["data"]
    args.labels_dir = Path(args.labels_dir or data_cfg["manual_label_dir"])
    args.partition_dir = Path(args.partition_dir or data_cfg["local_partition_dir"])
    args.manifest = Path(args.manifest or data_cfg["manifest"])
    args.registration = Path(args.registration or data_cfg["registration_summary"])

    thresholds = sorted({float(value) for value in args.thresholds})
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise SystemExit("thresholds must lie in (0, 1]")
    if args.dataset == "oscd_original":
        thresholds = [float(cfg["evaluation"]["threshold"])]

    fp32_path = args.output_dir / "onboard_fp32.onnx"
    int8_path = args.output_dir / "onboard_int8.onnx"
    report_path = args.output_dir / "quantization_report.json"
    ensure_new_outputs([fp32_path, int8_path, report_path], args.output_dir,
                       args.allow_overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    model, bands, checkpoint = load_model(args.checkpoint, torch.device("cpu"))

    if args.dataset == "local_manual":
        if args.normalisation is None:
            recorded = (checkpoint.get("run_config") or {}).get("normalisation_file")
            if not recorded:
                raise SystemExit("--normalisation is required for local_manual data")
            args.normalisation = Path(recorded)
        normalisation = require_statistics(args.normalisation, bands, tuple(args.pairs))
        calib_loader, calib_plan = make_local_calibration_loader(
            cfg, checkpoint, args, normalisation)
        entries = build_validation_entries(cfg, args, checkpoint, normalisation)
        val_loader = None
    else:
        normalisation = None
        calib_plan = {"split": "train", "dataset": "oscd_original"}
        calib_loader = make_dataset(cfg, "train", checkpoint, args.batch_size)
        val_loader = make_dataset(cfg, "validation", checkpoint, args.batch_size)
        entries = {}

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "bands": list(bands),
        "base_filters": int(checkpoint["base"]),
        "opset": args.opset,
        "dataset": args.dataset,
        "pairs": list(args.pairs),
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "thresholds": thresholds,
        "threshold_status": "fixed comparison points from earlier experiments; "
                            "not validated operating thresholds",
        "calibration": calib_plan,
        "calibration_split": "train",
        "evaluation_split": "validation",
        "normalisation": (str(Path(args.normalisation).resolve())
                          if normalisation is not None else None),
        "normalisation_sha256": (sha256(args.normalisation)
                                 if normalisation is not None else None),
        "onnx_output_is_probability": ONNX_OUTPUT_IS_PROBABILITY,
        "cpu_optimisation": optimisation_label(CPU_QDQ_OPTIMISATION),
        "test_split_used": False,
        "stages": {},
    }

    dummy = export_dummy(bands)
    input_names = INPUT_NAMES

    if "export" in args.stages:
        # torch 2.11 defaults to the dynamo exporter, which needs ``onnxscript``.
        # If it is absent, fall back to the TorchScript exporter, which only needs
        # ``onnx``. Either way the graph is checked before it is used.
        try:
            import onnxscript  # noqa: F401
            exporter = "dynamo"
        except ImportError:
            exporter = "torchscript_legacy"
        torch.onnx.export(
            model, dummy, str(fp32_path), input_names=input_names,
            output_names=[OUTPUT_NAME], opset_version=args.opset,
            dynamic_axes=dynamic_axes_for(input_names),
            do_constant_folding=True,
            dynamo=(exporter == "dynamo"))
        graph = onnx.load(str(fp32_path))
        onnx.checker.check_model(graph)
        # The legacy exporter restores the module to its previous training flag,
        # which for the wrapper propagates train mode into the inner model and
        # switches BatchNorm to batch statistics. Any reference computed afterwards
        # would then differ from the frozen graph.
        model.eval()
        report["stages"]["export"] = {
            "path": str(fp32_path), "size_bytes": fp32_path.stat().st_size,
            "sha256": sha256(fp32_path), "exporter": exporter,
            "dummy_shape": [1, len(bands), PATCH_SIZE, PATCH_SIZE],
            "dynamic_axes": dynamic_axes_for(input_names),
            "graph_inputs": [[dim.dim_value or dim.dim_param
                              for dim in value.type.tensor_type.shape.dim]
                             for value in graph.graph.input],
            "graph_outputs": [[dim.dim_value or dim.dim_param
                               for dim in value.type.tensor_type.shape.dim]
                              for value in graph.graph.output]}
        del graph
        print("exported", fp32_path, fp32_path.stat().st_size, "bytes,", exporter)

    fp32_session, fp32_providers = create_session(
        fp32_path, [CPU], require=CPU, optimization_level=CPU_QDQ_OPTIMISATION)

    if "verify" in args.stages:
        report["stages"]["verify"] = verify_export_contract(fp32_session, input_names, bands)
        if not report["stages"]["verify"]["passed"]:
            raise SystemExit("exported graph failed its runtime contract: %s"
                             % report["stages"]["verify"]["problems"])
        print("export contract verified at batch 1 and 8")

    if "parity" in args.stages:
        if args.dataset == "local_manual":
            worst, agreement, total = 0.0, 0, 0
            non_finite_torch = non_finite_onnx = 0
            scores_match = True
            per_pair = {}
            for pair, entry in entries.items():
                torch_scores, _ = predict_pair_torch(model, entry, args.eval_batch_size)
                onnx_scores, _ = predict_pair_onnx(fp32_session, input_names, entry,
                                                   args.eval_batch_size)
                delta = np.abs(torch_scores - onnx_scores)
                difference = float(delta.max())
                worst = max(worst, difference)
                agreement += int(((torch_scores >= 0.5) == (onnx_scores >= 0.5)).sum())
                total += int(torch_scores.size)
                per_pair[pair] = {
                    "max_abs_difference": difference,
                    "mean_abs_difference": float(delta.mean()),
                    "p95_abs_difference": float(np.percentile(delta, 95)),
                    "p99_abs_difference": float(np.percentile(delta, 99)),
                    "pearson_r": float(np.corrcoef(torch_scores, onnx_scores)[0, 1]),
                    "pixels": int(torch_scores.size),
                    "shape_match": tuple(torch_scores.shape) == tuple(onnx_scores.shape),
                    "threshold_agreement": float(
                        ((torch_scores >= 0.5) == (onnx_scores >= 0.5)).mean()),
                    "torch_mean": float(torch_scores.mean()),
                    "onnx_mean": float(onnx_scores.mean()),
                }
                non_finite_torch += int((~np.isfinite(torch_scores)).sum())
                non_finite_onnx += int((~np.isfinite(onnx_scores)).sum())
                scores_match = scores_match and (torch_scores.shape == onnx_scores.shape)
            report["stages"]["parity"] = {
                "max_abs_difference": worst,
                "threshold_agreement": agreement / max(total, 1),
                "per_pair": per_pair,
                "non_finite_torch": non_finite_torch,
                "non_finite_onnx": non_finite_onnx,
                "scores_match": scores_match,
                "tolerance": PARITY_TOLERANCE,
                "note": "both sources are probabilities; no sigmoid is applied here"}
            print("FP32 parity: max |torch - onnx| = %.3e, agreement %.6f"
                  % (worst, agreement / max(total, 1)))
        else:
            model.eval()
            max_abs, agreeing, total = 0.0, 0, 0
            for batch in val_loader:
                ort_out = session_probabilities(fp32_session, input_names,
                                                batch["image1"].numpy(),
                                                batch["image2"].numpy())[:, None]
                with torch.no_grad():
                    torch_out = model(batch["image1"], batch["image2"]).numpy()
                max_abs = max(max_abs, float(np.max(np.abs(torch_out - ort_out))))
                agreeing += int(((torch_out >= thresholds[0]) ==
                                 (ort_out >= thresholds[0])).sum())
                total += int(torch_out.size)
            ort_metrics, _ = evaluate_ort(fp32_session, val_loader, input_names,
                                          thresholds[0])
            report["stages"]["parity"] = {
                "max_abs_difference": max_abs,
                "threshold_agreement": agreeing / max(total, 1),
                "val_ap": ort_metrics["ap"], "val_auc": ort_metrics["auc"],
                "non_finite_torch": 0, "non_finite_onnx": 0, "scores_match": True,
                "tolerance": PARITY_TOLERANCE}

    if "quantize" in args.stages:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

        # Conversion-integrity gate: a quantised graph is only meaningful if the FP32
        # graph already reproduces the torch model on the validation windows.
        report["parity_gate"] = check_parity_gate(report)

        reader = CalibrationReader(calib_loader, input_names, args.calibration_batches)
        quantize_static(str(fp32_path), str(int8_path), reader,
                        quant_format=QuantFormat.QDQ,
                        per_channel=True, reduce_range=False,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8)
        onnx.checker.check_model(onnx.load(str(int8_path)))
        report["stages"]["quantize"] = {
            "path": str(int8_path), "size_bytes": int8_path.stat().st_size,
            "sha256": sha256(int8_path),
            "format": "QDQ", "activation": "QInt8", "weight": "QInt8",
            "per_channel": True, "calibration_batches": args.calibration_batches,
            "calibration_samples": (sum(calib_plan["per_pair"].values())
                                    if "per_pair" in calib_plan else None),
            "per_pair_calibration": calib_plan.get("per_pair"),
            "size_ratio_vs_fp32": round(int8_path.stat().st_size
                                        / fp32_path.stat().st_size, 3)}
        print("quantised", int8_path, int8_path.stat().st_size, "bytes")

    if "compare" in args.stages and int8_path.exists():
        int8_session, _ = create_session(int8_path, [CPU], require=CPU,
                                         optimization_level=CPU_QDQ_OPTIMISATION)
        if args.dataset == "local_manual":
            evaluated = evaluate_sessions(
                {"fp32": (fp32_session, input_names),
                 "int8": (int8_session, input_names)},
                entries, thresholds, args.eval_batch_size)
            difference = {}
            for pair in entries:
                fp32_scores = evaluated["fp32"]["scores"][pair]
                int8_scores = evaluated["int8"]["scores"][pair]
                delta = np.abs(fp32_scores - int8_scores)
                difference[pair] = {
                    "mean_abs_probability_difference": round(float(delta.mean()), 6),
                    "max_abs_probability_difference": round(float(delta.max()), 6),
                    "p95_abs_difference": round(float(np.percentile(delta, 95)), 6),
                    "p99_abs_difference": round(float(np.percentile(delta, 99)), 6),
                    "pearson_r": round(float(np.corrcoef(fp32_scores, int8_scores)[0, 1]), 6),
                    "decision_change_fraction": {
                        "%.2f" % threshold: round(float(
                            ((fp32_scores >= threshold) !=
                             (int8_scores >= threshold)).mean()), 6)
                        for threshold in thresholds}}
            report["stages"]["compare"] = {
                "fp32": {key: evaluated["fp32"][key] for key in ("pairs", "pooled", "macro")},
                "int8": {key: evaluated["int8"][key] for key in ("pairs", "pooled", "macro")},
                "difference": difference,
                "ap_delta_pooled": round(evaluated["int8"]["pooled"]["ap"]
                                         - evaluated["fp32"]["pooled"]["ap"], 6)}
            print("pooled AP %.4f -> %.4f (delta %+.4f)"
                  % (evaluated["fp32"]["pooled"]["ap"],
                     evaluated["int8"]["pooled"]["ap"],
                     report["stages"]["compare"]["ap_delta_pooled"]))
        else:
            fp32_metrics, fp32_scores = evaluate_ort(fp32_session, val_loader,
                                                     input_names, thresholds[0])
            int8_metrics, int8_scores = evaluate_ort(int8_session, val_loader,
                                                     input_names, thresholds[0])
            report["stages"]["compare"] = {
                "fp32": fp32_metrics, "int8": int8_metrics,
                "ap_delta": int8_metrics["ap"] - fp32_metrics["ap"],
                "max_probability_shift": float(np.max(np.abs(fp32_scores - int8_scores))),
                "mean_probability_shift": float(np.mean(np.abs(fp32_scores - int8_scores)))}
            print("INT8 vs FP32: AP %.4f -> %.4f (delta %+.4f)"
                  % (fp32_metrics["ap"], int8_metrics["ap"],
                     int8_metrics["ap"] - fp32_metrics["ap"]))

    if "bench" in args.stages:
        bench = {"shapes": {}, "available_providers": ort.get_available_providers()}
        for label, shapes in bench_shapes(bands).items():
            entry = {"cpu_fp32": dict(benchmark(fp32_session, shapes, args.bench_runs),
                                      providers=fp32_providers,
                                      optimisation=optimisation_label(CPU_QDQ_OPTIMISATION))}
            if int8_path.exists():
                int8_session, int8_providers = create_session(
                    int8_path, [CPU], require=CPU,
                    optimization_level=CPU_QDQ_OPTIMISATION)
                entry["cpu_int8"] = dict(benchmark(int8_session, shapes,
                                                   args.bench_runs),
                                         providers=int8_providers,
                                         optimisation=optimisation_label(
                                             CPU_QDQ_OPTIMISATION))
            bench["shapes"][label] = entry
        if CUDA in bench["available_providers"]:
            # A CUDA session that quietly fell back to CPU would produce a CPU
            # timing labelled as a GPU one, so the provider is verified before
            # anything is measured.
            try:
                cuda_session, cuda_providers = create_session(
                    fp32_path, [CUDA, CPU], require=CUDA, preload_cuda=True)
            except RuntimeError as error:
                bench["cuda_fp32"] = {"status": "not_measured",
                                      "reason": str(error)[:400]}
                print("CUDA not used:", str(error)[:200])
            else:
                bench["cuda_fp32"] = dict(
                    benchmark(cuda_session, bench_shapes(bands)["256x1"], args.bench_runs),
                    providers=cuda_providers, optimisation="ORT_ENABLE_ALL",
                    shape="256x1",
                    note="different provider; its INT8 output also differs from the "
                         "CPU one, so do not mix them into one comparison")
        bench["tensorrt"] = ("provider listed but the TensorRT runtime library is "
                             "not installed; not attempted by decision")
        report["stages"]["bench"] = bench

    import resource as _resource
    report["elapsed_seconds"] = round(time.perf_counter() - started_at, 2)
    report["peak_process_rss_mib"] = round(
        _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    write_json_atomic(report_path, report)
    print("wrote", report_path)


if __name__ == "__main__":
    main()
