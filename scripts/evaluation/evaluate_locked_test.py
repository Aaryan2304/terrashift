"""Final locked test evaluation on the untouched test stripe.

Mirrors the validation protocol exactly: the same inference lattice, core-offset stitching and
``LatticeAccumulator``, the same metric implementation, normalisation, band order and validity
handling, and CPU FP32 so it stays comparable with the validation reference.

The target selects partition code 3 and opens the label raster only inside the test stripe, so no
train or validation label column is read. Thresholds 0.05 / 0.10 / 0.50 are fixed diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

import quantize
from terrashift.data.validation_windows import (IGNORE_VALUE, ValidationLatticeDataset,
                                     validation_stripe, validation_target)
from terrashift.evaluation.testlock import TEST_CODE, read_test_target

THRESHOLDS = (0.05, 0.10, 0.50)
from terrashift.inference.torch_session import TorchSession
from terrashift.evaluation.components import component_recall, identify_known_component
CHECKPOINT = ROOT / "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"
NORMALISATION = ROOT / "outputs/dataset/v2/normalisation.json"
OUT = ROOT / "outputs/evaluation/final_locked_test_v1"


def build_test_entries(args, checkpoint, normalisation):
    bands = tuple(checkpoint.get("bands", quantize.LOCAL_BANDS))
    loader = quantize.pair_image_loader(args.manifest, args.registration, bands=bands)
    entries = {}
    for pair in args.pairs:
        labels = Path(args.labels_dir) / ("pair_%s_v2" % pair.lower()) / \
            ("manual_labels_%s.tif" % pair)
        partition = Path(args.partition_dir) / ("pair_%s_partition_10m.tif" % pair)
        target, col_offset, tile_width = read_test_target(labels, partition)
        entries[pair] = {
            "target": target, "col_offset": col_offset, "labels_path": str(labels),
            "dataset": ValidationLatticeDataset(
                [pair], target, col_offset, tile_width, quantize.PATCH_SIZE,
                quantize.STRIDE,
                image_loader=lambda r, c, s, _p=pair: loader(_p, r, c, s),
                normalisation=normalisation),
        }
    return entries


def coverage_counts(entry, batch_size=8):
    """Windows, valid evaluated pixels and ignored pixels for one pair."""
    target = entry["target"]
    dataset = entry["dataset"]
    valid_total = 0
    invalid = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        valid = batch["valid_mask"].numpy().astype(bool)
        present = batch["mask"].numpy() != IGNORE_VALUE
        valid_total += int((valid & present).sum())
        invalid += int((present & ~valid).sum())
    reviewed = int((target == 0).sum() + (target == 1).sum())
    # ``valid_total`` below counts per-window pixel hits, so overlapping windows count a
    # pixel more than once. The metric itself aggregates to unique pixels via the
    # accumulator, and evaluated_positive/negative are filled from the payload labels.
    return {
        "windows": len(dataset),
        "reviewed_pixels_in_stripe": reviewed,
        "reviewed_positive_in_stripe": int((target == 1).sum()),
        "reviewed_negative_in_stripe": int((target == 0).sum()),
        "window_pixel_hits_valid": valid_total,
        "window_pixel_hits_invalid": invalid,
        "ignored_uncertain_255_pixels": int((target == IGNORE_VALUE).sum()),
        "evaluated_unique_positive": None, "evaluated_unique_negative": None,
        "reviewed_pixels_dropped_as_invalid": None,
    }


def build_manifest(args, payload, results) -> dict:
    """Exact provenance for this evaluation."""
    import datetime
    import platform

    import onnxruntime

    partition_hashes = {}
    for pair in args.pairs:
        path = Path(args.partition_dir) / ("pair_%s_partition_10m.tif" % pair)
        labels = Path(args.labels_dir) / ("pair_%s_v2" % pair.lower()) / \
            ("manual_labels_%s.tif" % pair)
        partition_hashes[pair] = {
            "partition": {"path": str(path), "sha256": file_sha256(path)},
            "labels": {"path": str(labels), "sha256": file_sha256(labels)},
        }
    artifacts = {}
    for label, relative in (
            ("checkpoint", "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/"
                           "seed_43/best.pt"),
            ("normalisation", "outputs/dataset/v2/normalisation.json"),
            ("onnx_fp32", "outputs/quant/local_manual_v2_seed43_posweight_0p834/"
                          "onboard_fp32.onnx"),
            ("onnx_int8", "outputs/quant/local_manual_v2_seed43_posweight_0p834_int8_v1/"
                          "onboard_int8.onnx")):
        path = ROOT / relative
        if path.exists():
            artifacts[label] = {"path": relative, "sha256": file_sha256(path),
                                "bytes": path.stat().st_size}
    return {
        "evaluation": "final locked test evaluation",
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": "SENTINEL_DATA_ROOT",
        "splits": partition_hashes,
        "artifacts": artifacts,
        "protocol": payload["protocol"],
        "counts": payload["counts"],
        "models_evaluated": list(results),
        "software": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__, "rasterio": rasterio.__version__,
        },
        "commands": [
            "python scripts/evaluation/split_integrity_check.py",
            "python scripts/evaluation/evaluate_locked_test.py "
            "--registration outputs/audit/registration_v2_observation_valid/registration.json "
            "--partition-dir outputs/partitions_v2_observation_valid --batch-size 8",
        ],
        "test_labels_used_for_training_or_selection": False,
        "threshold_selected_from_test": False,
    }


def make_figure(payload, references, output):
    """Score distributions by class and precision-recall curves per pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = "torch_fp32" if "torch_fp32" in references else next(iter(references))
    scores, labels = references[label]["scores"], references[label]["labels"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, pair in enumerate(sorted(scores)):
        score, truth = scores[pair], labels[pair]
        positive, negative = score[truth == 1], score[truth == 0]
        bins = np.linspace(0, 1, 51)
        axes[0, column].hist(negative, bins=bins, alpha=0.6, density=True,
                             label="no change (n=%d)" % negative.size)
        axes[0, column].hist(positive, bins=bins, alpha=0.6, density=True,
                             label="change (n=%d)" % positive.size)
        axes[0, column].set_title("Pair %s score distribution" % pair)
        axes[0, column].set_xlabel("probability")
        axes[0, column].legend(fontsize=8)

        order = np.argsort(-score, kind="stable")
        ordered = truth[order]
        tp = np.cumsum(ordered == 1)
        total_positive = max(int((truth == 1).sum()), 1)
        precision = tp / np.arange(1, ordered.size + 1)
        recall = tp / total_positive
        axes[1, column].plot(recall, precision, lw=1.5)
        axes[1, column].set_title("Pair %s PR (AP %.4f)" % (pair, payload["models"][label][
            "pairs"][pair]["ap"]))
        axes[1, column].set_xlabel("recall")
        axes[1, column].set_ylabel("precision")
        axes[1, column].set_xlim(0, 1)
        axes[1, column].set_ylim(0, 1.02)
    figure.suptitle("Locked test split, CPU FP32, reviewed test pixels only")
    figure.tight_layout()
    path = output / "test_score_diagnostics.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def report_path(path):
    """Repo-relative when the path is inside the project, absolute otherwise.

    ``Path.relative_to`` raises for paths outside ROOT and when one side is relative
    and the other absolute, which is why the output directory used to crash the run.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--normalisation", type=Path, default=NORMALISATION)
    parser.add_argument("--manifest", default=str(ROOT / "configs/data_manifest.yaml"))
    parser.add_argument("--registration", default=None)
    parser.add_argument("--labels-dir", default=str(ROOT / "outputs/labels"))
    parser.add_argument("--partition-dir", default=str(ROOT / "outputs/partitions"))
    parser.add_argument("--pairs", nargs="+", default=["A", "B"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["torch_fp32", "onnx_fp32", "onnx_int8"])
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        print("output directory already populated, refusing to overwrite:", out)
        return 2

    _, _, checkpoint = quantize.load_model(str(args.checkpoint), "cpu")
    normalisation = json.load(open(args.normalisation, encoding="utf-8"))
    entries = build_test_entries(args, checkpoint, normalisation)
    print("test entries:", {p: len(entries[p]["dataset"]) for p in entries})

    sessions = {}
    if "torch_fp32" in args.models:
        model, _, _ = quantize.load_model(str(args.checkpoint), "cpu")
        sessions["torch_fp32"] = (TorchSession(model, torch.device("cpu"), half=False),
                                  quantize.INPUT_NAMES)
    for label, relative in (("onnx_fp32", "outputs/quant/local_manual_v2_seed43_posweight_0p834/onboard_fp32.onnx"),
                            ("onnx_int8", "outputs/quant/local_manual_v2_seed43_posweight_0p834_int8_v1/onboard_int8.onnx")):
        path = ROOT / relative
        if label in args.models and path.exists():
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            sessions[label] = (session, [i.name for i in session.get_inputs()])

    results = quantize.evaluate_sessions(sessions, entries, THRESHOLDS, args.batch_size)
    counts = {pair: coverage_counts(entry) for pair, entry in entries.items()}
    for label in results:
        for pair in results[label]["labels"]:
            labels = results[label]["labels"][pair]
            positive = int((labels == 1).sum())
            negative = int((labels == 0).sum())
            counts[pair]["evaluated_unique_positive"] = positive
            counts[pair]["evaluated_unique_negative"] = negative
            counts[pair]["reviewed_pixels_dropped_as_invalid"] = (
                counts[pair]["reviewed_pixels_in_stripe"] - positive - negative)

    components = {}
    for label, payload in results.items():
        per_pair = {}
        for pair in payload["labels"]:
            per_threshold = {}
            for threshold in THRESHOLDS:
                data = component_recall(payload["scores"][pair], payload["labels"][pair],
                                        entries[pair]["target"], threshold)
                data["known_component"] = identify_known_component(data["rows"], pair)
                per_threshold["%.2f" % threshold] = data
            per_pair[pair] = per_threshold
        components[label] = per_pair

    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "split": "test", "partition_code": TEST_CODE,
            "patch_size": quantize.PATCH_SIZE, "stride": quantize.STRIDE,
            "device": "cpu", "precision": "fp32 (primary)",
            "thresholds": list(THRESHOLDS),
            "threshold_status": "fixed diagnostic operating points, not optimised",
            "eligible_labels": {"change": 1, "no_change": 0, "ignored": IGNORE_VALUE},
            "invalid_observation_pixels_excluded": True,
            "label_reads_restricted_to_test_stripe": True,
        },
        "counts": counts,
        "models": {label: {k: v for k, v in p.items() if k not in ("scores", "labels")}
                   for label, p in results.items()},
        "components": components,
    }
    predictions = out / "test_predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    for label, p in results.items():
        for pair in p["scores"]:
            # One float32 vector per pair per model over the evaluated unique test pixels,
            # plus the shared label vector. This is the compact reproducible form.
            np.save(predictions / ("pair_%s_%s_scores.npy" % (pair, label)),
                    p["scores"][pair].astype(np.float32))
    reference = next(iter(results.values()))
    for pair in reference["labels"]:
        np.save(predictions / ("pair_%s_labels.npy" % pair),
                reference["labels"][pair].astype(np.uint8))
    payload["predictions_directory"] = report_path(predictions)
    payload["predictions_note"] = ("one row per evaluated unique test pixel; ordering is "
                                   "identical across models and to the label vector")

    (out / "test_metrics.json").write_text(json.dumps(payload, indent=1) + "\n")

    manifest = build_manifest(args, payload, results)
    (out / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    try:
        figure_path = make_figure(payload, references=results, output=predictions)
        print("figure:", figure_path)
    except Exception as error:  # a figure must never invalidate the metrics
        print("figure skipped:", type(error).__name__, error)


    for label, p in results.items():
        print("%-11s pooled AP %.6f AUC %.6f | macro AP %.6f | A %.6f B %.6f"
              % (label, p["pooled"]["ap"], p["pooled"]["auc"], p["macro"]["ap"],
                 p["pairs"]["A"]["ap"], p["pairs"]["B"]["ap"]))
    print("wrote", out / "test_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
