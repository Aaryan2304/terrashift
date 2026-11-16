"""Export, verify, quantise and benchmark the onboard candidate model.

Five stages, each of which can be run on its own:

    export    PyTorch checkpoint -> ONNX FP32 with dynamic spatial axes
    parity    torch vs ONNX Runtime FP32 on held-out validation patches
    quantize  static QDQ INT8, calibrated on TRAIN split patches only
    compare   FP32 vs INT8 accuracy on validation, and parity of the outputs
    bench     size and latency for CPU FP32, CPU INT8, and CUDA where available

Two rules this file enforces:

* Calibration reads the train split only. Using validation or test data to fit
  the quantisation ranges would be fitting a hyperparameter on the evaluation
  set.
* The test split is not touched. Everything here runs on validation, so the
  locked test regions are still only used by ``eval.py``.

Usage:
    python quantize.py --checkpoint <best.pt> --output-dir outputs/quant/<name>
Nothing is overwritten.

TensorRT is reported but not used: the runtime library is absent on this
machine, so the TensorRT provider cannot be created even though ONNX Runtime
lists it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.oscd_original import DEFAULT_BANDS, OSCDOriginalPatchDataset
from models.siamese_unet import SiameseUNet
from onnxruntime.quantization import CalibrationDataReader
from utils.metrics import average_precision, metrics_from_counts, roc_auc
from utils.ort_session import (CPU, CUDA, create_session, optimisation_label)

# QDQ models on CPU must not use the default optimiser: at ORT_ENABLE_EXTENDED
# and above ORT fuses Conv into QLinearConv, and the fused kernel reuses the
# int32 bias with x_scale*w_scale instead of the bias's own scale. Any
# convolution whose bias quantisation does not satisfy that invariant is then
# silently wrong. This model's second Siamese branch hits it. See
# docs/decisions/009-cpu-qdq-optimiser-defect.md.
CPU_QDQ_OPTIMISATION = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC


class Wrapper(torch.nn.Module):
    """Two named inputs, so the ONNX graph matches the training call."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image1, image2):
        return torch.sigmoid(self.model(image1, image2))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    bands = tuple(checkpoint.get("bands", DEFAULT_BANDS))
    model = SiameseUNet(in_channels=len(bands), base=int(checkpoint["base"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return Wrapper(model).to(device), bands, checkpoint


def make_dataset(cfg, split, checkpoint, batch_size):
    dataset = OSCDOriginalPatchDataset(
        Path(cfg["data"]["oscd_patch_index"]), split,
        tuple(checkpoint.get("bands", DEFAULT_BANDS)),
        checkpoint["mean"], checkpoint["std"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def valid_of(batch):
    """Validity mask for a batch, defaulting to all-valid.

    The original-OSCD patches carry no validity mask: the rectified rasters
    have no NoData and the labels cover every pixel. The local SAFE dataset does
    carry one, so this helper keeps one evaluation path for both.
    """
    valid = batch.get("valid_mask")
    if valid is None:
        return np.ones(batch["mask"].shape, dtype=bool)
    return valid.numpy().astype(bool)


def evaluate_ort(session, loader, input_names, threshold):
    scores, targets = [], []
    counts = np.zeros(4, dtype=np.int64)
    for batch in loader:
        feed = {input_names[0]: batch["image1"].numpy().astype(np.float32),
                input_names[1]: batch["image2"].numpy().astype(np.float32)}
        probabilities = session.run(None, feed)[0][:, 0]
        masks = batch["mask"].numpy()
        valid = valid_of(batch)
        for i in range(probabilities.shape[0]):
            prob = probabilities[i]
            truth = (masks[i] == 1)
            v = valid[i]
            scores.append(prob[v])
            targets.append(truth[v])
            pred = prob[v] >= threshold
            t = truth[v]
            counts += np.array([int((pred & t).sum()), int((pred & ~t).sum()),
                                int((~pred & t).sum()), int((~pred & ~t).sum())])
    scores = np.concatenate(scores)
    targets = np.concatenate(targets)
    result = metrics_from_counts(*counts.tolist())
    result["ap"] = average_precision(targets, scores)
    result["auc"] = roc_auc(targets, scores)
    result["mean_probability"] = float(scores.mean())
    return result, scores


class CalibrationReader(CalibrationDataReader):
    """Feed train-split patches to the static quantiser."""

    def __init__(self, loader, input_names, limit):
        self.input_names = input_names
        self.batches = []
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            self.batches.append({
                input_names[0]: batch["image1"].numpy().astype(np.float32),
                input_names[1]: batch["image2"].numpy().astype(np.float32)})
        if not self.batches:
            raise ValueError("no calibration batches were produced")
        self._position = 0

    def get_next(self):
        if self._position >= len(self.batches):
            return None
        batch = self.batches[self._position]
        self._position += 1
        return batch

    def rewind(self):
        self._position = 0


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--calibration-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bench-runs", type=int, default=20)
    parser.add_argument("--stages", nargs="*",
                        default=["export", "parity", "quantize", "compare", "bench"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = args.output_dir / "onboard_fp32.onnx"
    int8_path = args.output_dir / "onboard_int8.onnx"
    report_path = args.output_dir / "quantization_report.json"
    if report_path.exists():
        raise SystemExit("refusing to overwrite existing report: %s" % report_path)

    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    threshold = float(cfg["evaluation"]["threshold"])

    model, bands, checkpoint = load_model(args.checkpoint, torch.device("cpu"))
    val_loader = make_dataset(cfg, "validation", checkpoint, args.batch_size)
    train_loader = make_dataset(cfg, "train", checkpoint, args.batch_size)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "bands": list(bands),
        "base_filters": int(checkpoint["base"]),
        "opset": args.opset,
        "threshold": threshold,
        "calibration_split": "train",
        "evaluation_split": "validation",
        "test_split_used": False,
        "stages": {},
    }

    dummy = (torch.zeros(1, len(bands), 128, 128), torch.zeros(1, len(bands), 128, 128))
    input_names = ["image1", "image2"]

    if "export" in args.stages:
        # torch 2.11 defaults to the dynamo exporter, which needs the
        # ``onnxscript`` package. If it is absent, fall back to the TorchScript
        # exporter, which only needs ``onnx``. Either way the graph is checked
        # before it is used.
        try:
            import onnxscript  # noqa: F401
            exporter = "dynamo"
        except ImportError:
            exporter = "torchscript_legacy"
        torch.onnx.export(
            model, dummy, str(fp32_path), input_names=input_names,
            output_names=["probability"], opset_version=args.opset,
            dynamic_axes={name: {0: "batch", 2: "height", 3: "width"}
                          for name in input_names},
            do_constant_folding=True,
            dynamo=(exporter == "dynamo"))
        onnx.checker.check_model(onnx.load(str(fp32_path)))
        # The legacy exporter restores the module to its previous training
        # flag, which for the wrapper propagates train mode into the inner
        # model and switches BatchNorm to batch statistics. Any reference
        # computed afterwards would then differ from the frozen graph.
        model.eval()
        report["stages"]["export"] = {
            "path": str(fp32_path), "size_bytes": fp32_path.stat().st_size,
            "sha256": sha256(fp32_path), "exporter": exporter,
            "dynamic_axes": ["batch", "height", "width"]}
        print("exported", fp32_path, fp32_path.stat().st_size, "bytes,", exporter)

    fp32_session, fp32_providers = create_session(
        fp32_path, [CPU], require=CPU, optimization_level=CPU_QDQ_OPTIMISATION)

    if "parity" in args.stages:
        model.eval()
        max_abs = 0.0
        agreeing = 0
        total = 0
        for batch in val_loader:
            feed = {input_names[0]: batch["image1"].numpy().astype(np.float32),
                    input_names[1]: batch["image2"].numpy().astype(np.float32)}
            ort_out = fp32_session.run(None, feed)[0]
            with torch.no_grad():
                torch_out = model(batch["image1"], batch["image2"]).numpy()
            max_abs = max(max_abs, float(np.max(np.abs(torch_out - ort_out))))
            agreeing += int(((torch_out >= threshold) == (ort_out >= threshold)).sum())
            total += int(torch_out.size)
        ort_metrics, _ = evaluate_ort(fp32_session, val_loader, input_names, threshold)
        report["stages"]["parity"] = {
            "max_abs_difference": max_abs,
            "threshold_agreement": agreeing / max(total, 1),
            "val_ap": ort_metrics["ap"], "val_auc": ort_metrics["auc"]}
        print("FP32 parity: max |torch - onnx| = %.3e, agreement %.6f, val AP %.4f"
              % (max_abs, agreeing / max(total, 1), ort_metrics["ap"]))

    if "quantize" in args.stages:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

        reader = CalibrationReader(train_loader, input_names, args.calibration_batches)
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
            "size_ratio_vs_fp32": round(int8_path.stat().st_size
                                        / fp32_path.stat().st_size, 3)}
        print("quantised", int8_path, int8_path.stat().st_size, "bytes")

    if "compare" in args.stages and int8_path.exists():
        int8_session, _ = create_session(int8_path, [CPU], require=CPU,
                                         optimization_level=CPU_QDQ_OPTIMISATION)
        fp32_metrics, fp32_scores = evaluate_ort(fp32_session, val_loader, input_names, threshold)
        int8_metrics, int8_scores = evaluate_ort(int8_session, val_loader, input_names, threshold)
        report["stages"]["compare"] = {
            "fp32": fp32_metrics, "int8": int8_metrics,
            "ap_delta": int8_metrics["ap"] - fp32_metrics["ap"],
            "max_probability_shift": float(np.max(np.abs(fp32_scores - int8_scores))),
            "mean_probability_shift": float(np.mean(np.abs(fp32_scores - int8_scores))),
        }
        print("INT8 vs FP32: AP %.4f -> %.4f (delta %+.4f), AUC %.4f -> %.4f"
              % (fp32_metrics["ap"], int8_metrics["ap"],
                 int8_metrics["ap"] - fp32_metrics["ap"],
                 fp32_metrics["auc"], int8_metrics["auc"]))

    if "bench" in args.stages:
        shapes = {"image1": (1, len(bands), 128, 128),
                  "image2": (1, len(bands), 128, 128)}
        bench = {}
        bench["cpu_fp32"] = dict(benchmark(fp32_session, shapes, args.bench_runs),
                                 providers=fp32_providers,
                                 optimisation=optimisation_label(CPU_QDQ_OPTIMISATION))
        if int8_path.exists():
            int8_session, int8_providers = create_session(
                int8_path, [CPU], require=CPU,
                optimization_level=CPU_QDQ_OPTIMISATION)
            bench["cpu_int8"] = dict(benchmark(int8_session, shapes, args.bench_runs),
                                     providers=int8_providers,
                                     optimisation=optimisation_label(CPU_QDQ_OPTIMISATION))
        available = ort.get_available_providers()
        bench["available_providers"] = available
        if CUDA in available:
            # A CUDA session that quietly fell back to CPU would produce a CPU
            # timing labelled as a GPU one, so the provider is verified before
            # anything is measured. preload_cuda_libraries() covers the case
            # where torch has not already loaded the CUDA libraries.
            try:
                cuda_session, cuda_providers = create_session(
                    fp32_path, [CUDA, CPU], require=CUDA, preload_cuda=True)
            except RuntimeError as error:
                bench["cuda_fp32"] = {"status": "not_measured",
                                      "reason": str(error)[:400]}
                print("CUDA not used:", str(error)[:200])
            else:
                bench["cuda_fp32"] = dict(benchmark(cuda_session, shapes,
                                                    args.bench_runs),
                                          providers=cuda_providers,
                                          optimisation="ORT_ENABLE_ALL")
        bench["tensorrt"] = ("provider listed but the TensorRT runtime library is "
                             "not installed; not attempted by decision")
        report["stages"]["bench"] = bench
        print(json.dumps(bench, indent=2))

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", report_path)


if __name__ == "__main__":
    main()
