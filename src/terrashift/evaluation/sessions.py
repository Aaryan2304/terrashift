"""Evaluation sessions: per-model scoring, aggregation and reporting."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from terrashift.data.local_patches import BANDS, pair_image_loader
from terrashift.data.manifest import load_manifest, pair_config
from terrashift.data.validation_windows import (
    IGNORE_VALUE,
    ValidationLatticeDataset,
    read_validation_target,
)
from terrashift.evaluation.lattice import LatticeAccumulator, evaluate_lattice
from terrashift.evaluation.prediction import (predict_pair_onnx, predict_pair_torch,
                                             session_probabilities, valid_of)
from terrashift.evaluation.thresholds import metrics_at_thresholds
from terrashift.metrics import average_precision, metrics_from_counts, roc_auc
from terrashift.models.siamese_unet import SiameseUNet
from terrashift.preprocessing.normalisation import require_statistics

PAIRS = ("A", "B")
PATCH, STRIDE, BATCH = 256, 128, 4

def evaluate_sessions(sessions, entries, thresholds, batch_size=4):
    """Evaluate sessions on the same lattice, per pair, pooled and macro.

    ``sessions`` maps a label ("fp32", "int8") to ``(session, input_names)``.
    Pooled figures are micro over the concatenated pairs; macro figures are the
    unweighted mean of the two pairs. Both are reported because they disagree on
    this model.
    """
    per_session = {}
    for label, (session, input_names) in sessions.items():
        scores, labels = {}, {}
        for pair, entry in entries.items():
            pair_scores, pair_labels = predict_pair_onnx(session, input_names, entry,
                                                         batch_size)
            scores[pair], labels[pair] = pair_scores, pair_labels
        order = sorted(scores)
        pair_metrics = {pair: metrics_at_thresholds(scores[pair], labels[pair], thresholds)
                        for pair in order}
        pooled_scores = np.concatenate([scores[pair] for pair in order])
        pooled_labels = np.concatenate([labels[pair] for pair in order])
        result = {
            "pairs": pair_metrics,
            "pooled": metrics_at_thresholds(pooled_scores, pooled_labels, thresholds),
            # Macro is the unweighted mean of the two pairs, so Pair A counts as much as
            # Pair B. Pooled micro stays as it was: it answers a different question, and
            # on this model the two disagree by roughly 0.09 AP.
            "macro": {
                "ap": round(float(np.mean([pair_metrics[pair]["ap"] for pair in order])), 6),
                "auc": round(float(np.mean([pair_metrics[pair]["auc"] for pair in order])), 6),
                "pairs_averaged": len(order),
                "thresholds": {},
            },
            "scores": scores, "labels": labels,
        }
        for threshold in thresholds:
            key = "%.2f" % threshold
            fields = ("precision", "recall", "f1", "iou", "predicted_positive_fraction")
            result["macro"]["thresholds"][key] = {
                field: round(float(np.nanmean(
                    [pair_metrics[pair]["thresholds"][key][field] for pair in order])), 6)
                for field in fields}
        per_session[label] = result
    return per_session

def evaluate_ort(session, loader, input_names, threshold):
    """Patch-level evaluation for the OSCD path. Behaviour unchanged."""
    scores, targets = [], []
    counts = np.zeros(4, dtype=np.int64)
    for batch in loader:
        probabilities = session_probabilities(session, input_names,
                                              batch["image1"].numpy(),
                                              batch["image2"].numpy())
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

PAIRS = ("A", "B")
PATCH, STRIDE, BATCH = 256, 128, 4
def build_datasets(args):
    manifest = load_manifest(args.manifest)
    loader = pair_image_loader(args.manifest, args.registration, bands=BANDS)
    normalisation = require_statistics(args.normalisation, BANDS, PAIRS)
    datasets = {}
    for pair in PAIRS:
        labels = Path(args.labels_dir) / ("pair_%s_v2" % pair.lower()) / \
            ("manual_labels_%s.tif" % pair)
        partition = Path(args.partition_dir) / ("pair_%s_partition_10m.tif" % pair)
        target, col_offset, tile_width = read_validation_target(labels, partition)
        datasets[pair] = {
            "target": target, "col_offset": col_offset,
            "labels_path": str(labels),
            "dataset": ValidationLatticeDataset(
                [pair], target, col_offset, tile_width, PATCH, STRIDE,
                image_loader=lambda r, c, s, _p=pair: loader(_p, r, c, s),
                normalisation=normalisation),
        }
    return datasets, manifest, normalisation
def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(state["bands"]) != tuple(BANDS):
        raise SystemExit("checkpoint bands do not match: %s" % path)
    model = SiameseUNet(in_channels=len(BANDS), base=int(state["base"]))
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval(), state
def predict_pair(model, pair, entry, device):
    """One pass per pair: canonical accumulator plus the reviewed-pixel vectors."""
    loader = DataLoader(entry["dataset"], batch_size=BATCH, shuffle=False, num_workers=0)
    target = entry["target"]
    acc = LatticeAccumulator(target.shape, entry["col_offset"])
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x1 = batch["image1"].to(device)
            x2 = batch["image2"].to(device)
            probs = torch.sigmoid(model(x1, x2).float()).squeeze(1).cpu().numpy()
            valid = batch["valid_mask"].numpy()
            for item in range(probs.shape[0]):
                acc.add(int(batch["row_off"][item]), int(batch["col_off"][item]),
                        probs[item], valid[item],
                        core_offset=int(batch["core_offset"][item]),
                        core_count=int(batch["core_count"][item]))
    covered = (acc.count > 0) & (target != IGNORE_VALUE)
    scores = (acc.total[covered] / acc.count[covered]).astype(np.float32)
    labels = target[covered].astype(np.int64)
    return acc, scores, labels
