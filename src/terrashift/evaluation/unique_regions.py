"""Unique-pixel statistics for the local patch layout.

Patches are 256 px at stride 128, so a labelled pixel falls in up to four of them. That is fine for
training and wrong for validation, where one reviewed pixel must not become four observations.
``coverage_1d`` gives the per-coordinate multiplicity that makes a weighted sum equal the sum over
unique pixels; ``UniqueValidationAccumulator`` stitches and averages overlaps so each pixel scores
once. Weighting is exact only while a pixel keeps the same validity in every patch covering it,
which fails where halos differ at the tile border.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window

from terrashift.metrics import average_precision, metrics_from_counts, roc_auc


def coverage_1d(origins, patch_size, length):
    """How many patch windows cover each coordinate of one axis.

    Parameters
    ----------
    origins : sequence of int
        Window start positions, as returned by ``patch_origins``.
    patch_size : int
    length : int

    Returns
    -------
    numpy.ndarray, int64, shape ``(length,)``
        Zero means no patch covers that coordinate.
    """
    counts = np.zeros(int(length), dtype=np.int64)
    for origin in origins:
        start = int(origin)
        stop = min(start + int(patch_size), int(length))
        if start < 0 or start >= stop:
            raise ValueError("patch origin %d is outside the axis" % start)
        counts[start:stop] += 1
    return counts


def _axis_slices(origin, size):
    return slice(int(origin), int(origin) + int(size))


def weighted_training_statistics(dataset):
    """Band statistics and class counts over unique training pixels.

    Parameters
    ----------
    dataset : terrashift.data.local_patches.LocalManualPatchDataset
        A dataset for the ``train`` split with ``mean=None`` so the values are
        raw reflectance, not standardised.

    Returns
    -------
    dict with ``mean``, ``std`` (float32, one value per band, pooled over both
    dates), ``positive`` and ``negative`` (weighted pixel counts),
    ``unique_pixels``, ``patch_pixels`` and ``weighting_rounding_error``.

    The mean and standard deviation pool date 1 and date 2, which is what the
    trainer applies to both dates.
    """
    if getattr(dataset, "split", None) != "train":
        raise ValueError("training statistics require the train split")

    sizes = {int(r["height"]) for r in dataset.rows}
    if len(sizes) != 1:
        raise ValueError("all patches must share one height for coverage weighting")
    size = sizes.pop()

    # Multiplicity at (row, col) is the number of distinct row origins that
    # cover the row times the number of distinct column origins that cover the
    # column. Counting patch occurrences instead would multiply by the number
    # of patches on the other axis.
    height = max(int(r["row_off"]) + size for r in dataset.rows)
    width = max(int(r["col_off"]) + size for r in dataset.rows)
    row_counts = coverage_1d(sorted({int(r["row_off"]) for r in dataset.rows}),
                             size, height)
    col_counts = coverage_1d(sorted({int(r["col_off"]) for r in dataset.rows}),
                             size, width)
    if np.any(row_counts <= 0) or np.any(col_counts <= 0):
        raise ValueError("patch origins leave a gap; coverage weighting is undefined")

    n_bands = len(dataset.bands)
    sums = np.zeros(n_bands, dtype=np.float64)
    squares = np.zeros(n_bands, dtype=np.float64)
    total = 0.0
    patch_pixels = 0
    positive = 0.0
    negative = 0.0

    for index in range(len(dataset)):
        sample = dataset[index]
        row = dataset.rows[index]
        r0 = int(row["row_off"])
        c0 = int(row["col_off"])
        weight = 1.0 / np.outer(row_counts[r0:r0 + size], col_counts[c0:c0 + size])
        valid = sample["valid_mask"].numpy().astype(bool)
        weights = weight[valid]
        patch_pixels += int(valid.sum())
        total += float(weights.sum())

        for key in ("image1", "image2"):
            image = sample[key].numpy()
            for band in range(n_bands):
                values = image[band][valid].astype(np.float64)
                sums[band] += float((values * weights).sum())
                squares[band] += float(((values * values) * weights).sum())

        labels = sample["mask"].numpy()[valid]
        positive += float(weights[labels == 1].sum())
        negative += float(weights[labels == 0].sum())

    if total <= 0:
        raise ValueError("training annotations contain no valid pixels")

    # The trainer applies one mean and one std to both dates, so the band
    # statistic pools date 1 and date 2. Each date contributes one value per
    # unique pixel, hence the factor of two.
    pooled = total * 2.0
    mean = sums / pooled
    variance = squares / pooled - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-12))
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "positive": positive,
        "negative": negative,
        "unique_pixels": total,
        "pooled_values": pooled,
        "patch_pixels": patch_pixels,
        "weighting_rounding_error": abs(total - round(total)),
        "bands": tuple(dataset.bands),
    }


class UniqueValidationAccumulator:
    """Stitch patch probabilities onto one pair's grid and score each pixel once.

    Overlapping predictions are averaged. A pixel counts as valid when at least
    one contributing patch marked it valid; the mean is taken over the patches
    that did, so a pixel is never dropped because one patch had a border halo.
    """

    def __init__(self, pair, shape, label_path, max_pixels=130_000_000):
        self.pair = str(pair)
        self.shape = (int(shape[0]), int(shape[1]))
        if self.shape[0] * self.shape[1] > int(max_pixels):
            raise ValueError(
                "validation grid %dx%d exceeds the %d pixel budget"
                % (self.shape[0], self.shape[1], max_pixels))
        self.label_path = label_path
        self.total = np.zeros(self.shape, dtype=np.float32)
        self.count = np.zeros(self.shape, dtype=np.uint8)

    def add(self, row_off, col_off, probabilities, valid):
        rows = _axis_slices(row_off, probabilities.shape[0])
        cols = _axis_slices(col_off, probabilities.shape[1])
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != probabilities.shape:
            raise ValueError("valid mask shape does not match the patch")
        if rows.stop > self.shape[0] or cols.stop > self.shape[1]:
            raise ValueError("patch %r falls outside the validation grid" % ((row_off, col_off),))
        window = (rows, cols)
        self.total[window] += np.where(valid, probabilities.astype(np.float32), 0.0)
        self.count[window] += valid.astype(np.uint8)

    @property
    def covered_pixels(self):
        return int((self.count > 0).sum())

    def finalize(self):
        """Return (scores, labels) for every reviewed pixel, exactly once."""
        covered = self.count > 0
        n = int(covered.sum())
        if n == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        rows = np.flatnonzero(covered.any(axis=1))
        cols = np.flatnonzero(covered.any(axis=0))
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        with rasterio.open(self.label_path) as ds:
            labels = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
        sub_covered = covered[r0:r1, c0:c1]
        sub_total = self.total[r0:r1, c0:c1]
        sub_count = self.count[r0:r1, c0:c1]
        scores = (sub_total[sub_covered] / sub_count[sub_covered]).astype(np.float32)
        truth = labels[sub_covered].astype(np.int64)
        if not np.all(np.isin(truth, (0, 1))):
            raise ValueError("reviewed pixels must carry a 0 or 1 label")
        return scores, truth


def unique_validation_metrics(scores, labels, threshold=0.5):
    """Ranking and threshold metrics over unique reviewed pixels."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same length")
    result: dict = {
        "unique_valid_pixels": int(scores.size),
        "unique_positive_pixels": int((labels == 1).sum()),
    }
    if scores.size == 0:
        result.update({"ap": float("nan"), "auc": float("nan"),
                       "precision": float("nan"), "recall": float("nan"),
                       "f1": float("nan"), "iou": float("nan"),
                       "tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0})
        return result
    pred = scores >= threshold
    truth = labels == 1
    counts = metrics_from_counts(int((pred & truth).sum()), int((pred & ~truth).sum()),
                                 int((~pred & truth).sum()), int((~pred & ~truth).sum()))
    result.update(counts)
    result["ap"] = average_precision(labels, scores)
    result["auc"] = roc_auc(labels, scores)
    return result


def evaluate_unique(model, loader, device, threshold=0.5, criterion=None,
                    max_batches=0):
    """Run a validation loader and return metrics over unique pixels.

    The loader must iterate samples that carry ``pair``, ``row_off``,
    ``col_off``, ``label_path`` and the patch tensors. Patches of one pair must
    arrive consecutively, which holds for ``shuffle=False`` over an index that
    is grouped by pair.

    When ``criterion`` is given, the mean patch loss is returned as ``loss``.
    That keeps the loss patch-weighted, which is what training optimises, while
    every reported metric is computed once per reviewed pixel.
    """
    import torch

    model.eval()
    accumulators = {}
    collected = {}
    n_patches = 0
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            x1 = batch["image1"].to(device)
            x2 = batch["image2"].to(device)
            logits = model(x1, x2)
            if criterion is not None:
                target = batch["mask"].to(device, non_blocking=True).float().unsqueeze(1)
                valid_t = batch.get("valid_mask")
                if valid_t is None:
                    valid_t = torch.ones_like(target, dtype=torch.bool)
                else:
                    valid_t = valid_t.to(device, non_blocking=True).bool()
                    if valid_t.ndim == 3:
                        valid_t = valid_t.unsqueeze(1)
                total_loss += float(criterion(logits, target, valid_t).detach()) * x1.shape[0]
            n_samples += int(x1.shape[0])
            probs = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
            valid = batch["valid_mask"].cpu().numpy().astype(bool)
            pairs = list(batch["pair"])
            roffs = batch["row_off"].tolist()
            coffs = batch["col_off"].tolist()
            label_paths = list(batch["label_path"])
            for i, pair in enumerate(pairs):
                if pair not in accumulators:
                    # The stitching grid is the pair's label raster, not the
                    # patch shape.
                    with rasterio.open(label_paths[i]) as label_ds:
                        grid_shape = (label_ds.height, label_ds.width)
                    accumulators[pair] = UniqueValidationAccumulator(
                        pair, grid_shape, label_paths[i])
                accumulators[pair].add(roffs[i], coffs[i], probs[i], valid[i])
                n_patches += 1

    all_scores = []
    all_labels = []
    for pair, accumulator in accumulators.items():
        scores, labels = accumulator.finalize()
        all_scores.append(scores)
        all_labels.append(labels)
        collected[pair] = {"covered_pixels": accumulator.covered_pixels,
                           "reviewed_pixels": int(scores.size)}

    if all_scores:
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
    else:
        scores = np.empty(0, dtype=np.float32)
        labels = np.empty(0, dtype=np.int64)

    result = unique_validation_metrics(scores, labels, threshold)
    result["validation_patches"] = n_patches
    result["per_pair"] = collected
    result["loss"] = (total_loss / n_samples) if n_samples else float("nan")
    return result
