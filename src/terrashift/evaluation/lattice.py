"""Validation metrics over the inference lattice, scored once per pixel.

Full-tile inference averages overlapping window predictions with equal weight and writes one value
per pixel, so validation has to do the same or it measures a different model. This accumulates that
average over the validation stripe and scores each reviewed pixel once.

Two rules are enforced rather than trusted: a window contributes only to its strip's core rows,
matching what ``predict_safe_pair`` writes, and only pixels whose target is 0 or 1 take part, so
train, guard and unreviewed pixels are ignored by construction.
"""

from __future__ import annotations

import numpy as np

from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.unique_regions import unique_validation_metrics

__all__ = ["LatticeAccumulator", "evaluate_lattice"]


class LatticeAccumulator:
    """Equal-weight mean of overlapping predictions, over the validation stripe."""

    def __init__(self, shape, col_offset):
        self.shape = (int(shape[0]), int(shape[1]))
        if self.shape[0] < 1 or self.shape[1] < 1:
            raise ValueError("accumulator needs a non-empty grid")
        self.col_offset = int(col_offset)
        self.total = np.zeros(self.shape, dtype=np.float64)
        self.count = np.zeros(self.shape, dtype=np.uint16)

    def add(self, row_off, col_off, probabilities, valid, core_offset=0, core_count=None):
        """Add one window, clipped to its core rows and to the stripe."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        if probabilities.ndim != 2:
            raise ValueError("probabilities must be 2-D")
        if valid.shape != probabilities.shape:
            raise ValueError("valid mask shape does not match the prediction")
        count = probabilities.shape[0] if core_count is None else int(core_count)
        if core_offset < 0 or count < 1 or core_offset + count > probabilities.shape[0]:
            raise ValueError("core range %r..%r falls outside the prediction"
                             % (core_offset, core_offset + count))
        # A window at the stripe edge extends past it; inference still runs it, so
        # clip the contribution to the grid instead of rejecting the window. The
        # clipped-away pixels are outside the validation split and are never scored.
        row_start = int(row_off) + int(core_offset)
        row_end = row_start + count
        col_start = int(col_off) - self.col_offset
        col_end = col_start + probabilities.shape[1]
        r0, r1 = max(row_start, 0), min(row_end, self.shape[0])
        c0, c1 = max(col_start, 0), min(col_end, self.shape[1])
        if r1 <= r0 or c1 <= c0:
            raise ValueError("window %r..%r x %r..%r lies entirely outside the grid"
                             % (row_start, row_end, col_start, col_end))
        patch_rows = slice(int(core_offset) + (r0 - row_start),
                           int(core_offset) + (r1 - row_start))
        patch_cols = slice(c0 - col_start, c1 - col_start)
        sub_valid = valid[patch_rows, patch_cols]
        self.total[r0:r1, c0:c1] += np.where(sub_valid,
                                             probabilities[patch_rows, patch_cols], 0.0)
        self.count[r0:r1, c0:c1] += sub_valid.astype(np.uint16)

    @property
    def covered_pixels(self):
        return int((self.count > 0).sum())

    def finalize(self, target):
        """Scores and labels for each reviewed validation pixel, exactly once.

        ``target`` is the sanitised validation-only raster. No label file is read
        here, so no train or test value can enter a metric.
        """
        target = np.asarray(target)
        if target.shape != self.shape:
            raise ValueError("target shape does not match the accumulator")
        covered = (self.count > 0) & (target != IGNORE_VALUE)
        n = int(covered.sum())
        if n == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64), 0
        scores = (self.total[covered] / self.count[covered]).astype(np.float32)
        truth = target[covered].astype(np.int64)
        if not np.all(np.isin(truth, (0, 1))):
            raise ValueError("reviewed pixels must carry a 0 or 1 label")
        return scores, truth, int((self.count[covered] > 1).sum())


def evaluate_lattice(model, loader, device, threshold=0.5, criterion=None,
                     target=None, col_offset=0, max_batches=0):
    """Run a lattice validation loader; return metrics over unique pixels.

    Loss is the mean patch loss restricted to validation pixels, which keeps it
    comparable with training; every reported metric is computed once per reviewed
    validation pixel after equal-weight averaging.
    """
    import torch

    model.eval()
    if target is None:
        raise ValueError("the sanitised validation target is required")
    accumulator = LatticeAccumulator(np.asarray(target).shape, col_offset)
    total_loss = 0.0
    n_batches = 0
    n_windows = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            x1 = batch["image1"].to(device, non_blocking=True)
            x2 = batch["image2"].to(device, non_blocking=True)
            valid = batch["valid_mask"]
            if valid.ndim == 3:
                valid = valid.unsqueeze(1)
            valid = valid.to(device, non_blocking=True).bool()
            target_t = batch["mask"].to(device, non_blocking=True).float().unsqueeze(1)
            # Zero the invalid targets before the loss sees them, so 255 never
            # enters the arithmetic even as a multiplied-by-zero term.
            target_t = torch.where(valid, target_t, torch.zeros_like(target_t))
            logits = model(x1, x2)
            if criterion is not None:
                total_loss += float(criterion(logits, target_t, valid))
                n_batches += 1
            probabilities = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            valid_np = valid.squeeze(1).cpu().numpy()
            for item in range(probabilities.shape[0]):
                accumulator.add(
                    int(batch["row_off"][item]), int(batch["col_off"][item]),
                    probabilities[item], valid_np[item],
                    core_offset=int(batch["core_offset"][item]),
                    core_count=int(batch["core_count"][item]))
                n_windows += 1
    scored = (accumulator.count > 0) & (np.asarray(target) != IGNORE_VALUE)
    votes = accumulator.count[scored]
    scores, truth, repeated = accumulator.finalize(target)
    non_finite = int((~np.isfinite(scores)).sum())
    result = unique_validation_metrics(scores, truth, threshold=threshold)
    result["unique_repeated_pixels"] = repeated
    result["accumulated_pixels"] = accumulator.covered_pixels
    result["loss"] = (total_loss / n_batches) if n_batches else float("nan")
    result["loss_basis"] = "mean patch loss over validation pixels only"
    result["validation_windows"] = n_windows
    result["votes"] = {
        "min": int(votes.min()) if votes.size else 0,
        "median": float(np.median(votes)) if votes.size else 0.0,
        "mean": float(votes.mean()) if votes.size else 0.0,
        "max": int(votes.max()) if votes.size else 0,
    }
    result["finite_score_fraction"] = (
        float(1.0 - non_finite / scores.size) if scores.size else float("nan"))
    # Kept so per-pair vectors can be concatenated exactly; dropped before reporting.
    result["_scores"] = scores
    result["_labels"] = truth
    return result


def evaluate_lattice_all(model, loaders, device, threshold=0.5, criterion=None,
                         max_batches=0):
    """Validate every pair on the inference lattice and combine the metrics.

    ``loaders`` maps pair name to ``(loader, dataset)``. Per-pair score vectors are
    concatenated and scored once, so the combined AP is exact rather than an average
    of averages.
    """
    if not loaders:
        raise ValueError("no validation datasets supplied")
    scores, labels, per_pair, losses = [], [], {}, []
    for pair, (loader, dataset) in loaders.items():
        result = evaluate_lattice(
            model, loader, device, threshold=threshold, criterion=criterion,
            target=dataset.target, col_offset=dataset.col_offset,
            max_batches=max_batches)
        per_pair[pair] = {
            "windows": result["validation_windows"],
            "reviewed_pixels": result["unique_valid_pixels"],
            "accumulated_pixels": result["accumulated_pixels"],
            "repeat_pixels": result["unique_repeated_pixels"],
            "votes": result["votes"],
            "finite_score_fraction": result["finite_score_fraction"],
            "ap": result["ap"],
        }
        losses.append(result["loss"])
        scores.append(result["_scores"])
        labels.append(result["_labels"])

    combined = unique_validation_metrics(np.concatenate(scores),
                                         np.concatenate(labels), threshold=threshold)
    combined["per_pair"] = per_pair
    combined["loss"] = float(np.nanmean(losses))
    combined["loss_basis"] = "mean of per-pair patch losses over validation pixels only"
    combined["validation_windows"] = sum(info["windows"] for info in per_pair.values())
    combined["accumulator"] = "equal-weight mean of overlapping window probabilities"
    return combined
