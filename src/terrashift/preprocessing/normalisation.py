"""Per-pair, per-band normalisation statistics for the local data.

One mean and standard deviation per band and pair, pooled over both dates, from valid pixels in the
TRAIN partition, counting each geographic pixel once. DN 0 and mask-rejected pixels are excluded.

The accumulator uses Welford's update with Chan's merge, so block summaries combine exactly without
holding the region in memory or losing precision to cancellation.

These are local statistics (Strategy B). The OSCD checkpoint's stored values are raw DN and must not
be applied to reflectance. Matching the first two moments does not equalise the distributions:
higher moments, spatial correlation and the per-date offset all remain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ["BandMoments", "PairStatistics", "NormalisationError",
           "compute_pair_statistics", "compute_window_statistics", "load_statistics",
           "require_statistics", "save_statistics", "normalise_reflectance",
           "pair_statistics"]


class BandMoments:
    """Streaming mean and variance for one band, by Welford / Chan."""

    __slots__ = ("count", "mean", "m2")

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values):
        """Add a batch of finite values."""
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return self
        batch_count = int(values.size)
        batch_mean = float(values.mean())
        centred = values - batch_mean
        batch_m2 = float(np.dot(centred, centred))
        self.merge_raw(batch_count, batch_mean, batch_m2)
        return self

    def merge_raw(self, count, mean, m2):
        if count <= 0:
            return self
        total = self.count + count
        delta = mean - self.mean
        self.mean = self.mean + delta * (count / total)
        self.m2 = self.m2 + m2 + delta * delta * self.count * count / total
        self.count = total
        return self

    def merge(self, other):
        return self.merge_raw(other.count, other.mean, other.m2)

    def summary(self):
        if self.count < 2:
            raise ValueError("need at least two pixels for a standard deviation")
        variance = self.m2 / self.count
        return {"count": int(self.count), "mean": float(self.mean),
                "std": float(np.sqrt(max(variance, 0.0))),
                "variance": float(max(variance, 0.0))}


class PairStatistics:
    """One :class:`BandMoments` per band for a single pair."""

    def __init__(self, bands):
        self.bands = tuple(bands)
        self.moments = {band: BandMoments() for band in self.bands}

    def update(self, band, values):
        self.moments[band].update(values)
        return self

    def summary(self):
        return {band: self.moments[band].summary() for band in self.bands}

    def as_arrays(self):
        """``(mean, std)`` arrays in band order, float32, shape ``(C, 1, 1)``."""
        summary = self.summary()
        mean = np.array([summary[b]["mean"] for b in self.bands], dtype=np.float32)
        std = np.array([summary[b]["std"] for b in self.bands], dtype=np.float32)
        return mean.reshape(-1, 1, 1), std.reshape(-1, 1, 1)


def compute_pair_statistics(date1, date2, partition, validity, bands,
                            block=512, stride=1024, max_blocks=60):
    """Statistics from valid TRAIN-partition pixels of both dates, each counted once.

    ``partition`` is the split raster, ``validity`` the observation-validity mask
    (``None`` means the partition is the only filter). Blocks are non-overlapping
    slices on a fixed stride, so every pixel is visited at most once and the block
    summaries merge exactly.

    Validation and test columns are never read: candidate blocks are selected from
    the TRAIN partition before any band window is opened.
    """
    partition = np.asarray(partition)
    if partition.ndim != 2:
        raise ValueError("partition must be 2-D")
    train_code = 1
    columns = np.flatnonzero((partition == train_code).any(axis=0))
    if columns.size == 0:
        raise ValueError("partition has no TRAIN columns")
    col_lo, col_hi = int(columns[0]), int(columns[-1]) + 1
    height = partition.shape[0]

    blocks = []
    for row0 in range(0, height - block + 1, stride):
        for col0 in range(col_lo, col_hi - block + 1, stride):
            window = partition[row0:row0 + block, col0:col0 + block]
            if np.any(window == train_code):
                blocks.append((row0, col0))
    if not blocks:
        raise ValueError("no TRAIN block found at block=%d stride=%d" % (block, stride))
    if len(blocks) > max_blocks:
        step = max(1, len(blocks) // max_blocks)
        blocks = blocks[::step][:max_blocks]

    statistics = PairStatistics(bands)
    # Counts are per band-read, so they are comparable across bands. "excluded"
    # means finite but rejected by the partition or the validity mask; "nodata"
    # means NaN, which is DN = 0.
    rejected = {"nodata": 0, "excluded": 0, "total": 0, "kept": 0}
    for row0, col0 in blocks:
        window = partition[row0:row0 + block, col0:col0 + block]
        allowed = (window == train_code)
        if validity is not None:
            allowed = allowed & np.asarray(validity)[row0:row0 + block, col0:col0 + block]
        for product in (date1, date2):
            for band in bands:
                values = product.read_band(
                    band, window=((row0, row0 + block), (col0, col0 + block)))[0]
                finite = np.isfinite(values)
                selected = finite & allowed
                rejected["total"] += int(values.size)
                rejected["nodata"] += int((~finite).sum())
                rejected["excluded"] += int((finite & ~allowed).sum())
                rejected["kept"] += int(selected.sum())
                statistics.update(band, values[selected])
    for band in bands:
        if statistics.moments[band].count < 2:
            raise ValueError("band %s has too few valid TRAIN pixels" % band)
    return statistics, {"blocks": len(blocks), "block_size": block, "stride": stride,
                        "train_columns": [col_lo, col_hi], "rejected": rejected}


def normalise_reflectance(image, mean, std, valid=None):
    """Z-score with the pair statistics, then set invalid pixels to neutral zero.

    Invalid pixels are NaN at read time. Z-scoring the filled value would place them
    at roughly -3 sigma, a systematic outlier the model never saw in training, so
    they are zeroed *after* normalisation instead.
    """
    image = np.asarray(image, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    if image.ndim != 3 or image.shape[0] != mean.shape[0]:
        raise ValueError("image must be (bands, rows, cols) matching the statistics")
    finite = np.isfinite(image)
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool)[None, :, :]
    filled = np.where(finite, image, 0.0)
    normalised = (filled - mean) / std
    normalised = np.where(finite, normalised, 0.0).astype(np.float32)
    return normalised


def save_statistics(path, pairs, extra=None):
    payload = {"scheme": "per-pair per-band z-score, pooled over both dates",
               "source": "valid TRAIN-partition pixels including unlabelled",
               "units": "TOA reflectance, (DN + offset) / quantification",
               "pairs": pairs}
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


def load_statistics(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "pairs" not in payload:
        raise ValueError("normalisation file has no pairs section")
    return payload


def pair_statistics(payload, pair):
    """``(mean, std)`` arrays for one pair, from a statistics payload."""
    entry = payload["pairs"][pair]
    bands = list(entry)
    mean = np.array([entry[b]["mean"] for b in bands], dtype=np.float32)
    std = np.array([entry[b]["std"] for b in bands], dtype=np.float32)
    if np.any(std <= 0) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("pair %s has unusable statistics" % pair)
    return mean.reshape(-1, 1, 1), std.reshape(-1, 1, 1)


def compute_window_statistics(rows, image_loader, validity_by_pair, bands,
                              shapes=None):
    """Statistics over each **unique** valid pixel inside the training windows.

    This is the population the model is actually shown, but each geographic pixel is
    counted once per date, not once per window. With 256 px windows at stride 128 the
    windows overlap by half, so a naive loop counts 48-57 % of pixels more than once
    and weights the interior of the annotation clusters more heavily than the scene.

    ``image_loader(pair, row_off, col_off, size)`` is the same loader the patches are
    materialised with, so date 2 carries the registration shift here exactly as it
    does in the dataset.

    Returns ``(statistics, counts)`` where ``counts`` separates instances from unique
    coordinates.
    """
    seen = {}
    statistics = PairStatistics(bands)
    counts = {"instances": 0, "unique": 0, "repeated": 0, "nodata": 0,
              "invalid": 0, "per_pair": {}}
    for row in rows:
        pair = row["pair"]
        row_off, col_off = int(row["row_off"]), int(row["col_off"])
        size = int(row["height"])
        if pair not in seen:
            if shapes is None or pair not in shapes:
                raise ValueError("shape for pair %s is required" % pair)
            seen[pair] = np.zeros(shapes[pair], dtype=bool)
            counts["per_pair"][pair] = {"instances": 0, "unique": 0, "repeated": 0}
        image1, image2 = image_loader(pair, row_off, col_off, size)
        footprint = seen[pair][row_off:row_off + size, col_off:col_off + size]
        allowed = np.ones(footprint.shape, dtype=bool)
        validity = validity_by_pair.get(pair)
        rejected_by_validity = None
        if validity is not None:
            crop = np.asarray(validity)[row_off:row_off + size,
                                        col_off:col_off + size]
            allowed &= crop
            rejected_by_validity = ~crop
        allowed &= ~footprint                 # this window's new geographic pixels only
        counts["per_pair"][pair]["instances"] += int(size * size)
        counts["instances"] += int(size * size)
        taken = int(allowed.sum())
        counts["per_pair"][pair]["unique"] += taken
        counts["per_pair"][pair]["repeated"] += int(size * size) - taken
        counts["unique"] += taken
        counts["repeated"] += int(size * size) - taken
        for image in (image1, image2):
            for index, band in enumerate(bands):
                values = np.asarray(image[index], dtype=np.float32)
                finite = np.isfinite(values)
                selected = finite & allowed
                counts["nodata"] += int((~finite).sum())
                if rejected_by_validity is not None:
                    counts["invalid"] += int((finite & rejected_by_validity).sum())
                statistics.update(band, values[selected])
        seen[pair][row_off:row_off + size, col_off:col_off + size] |= allowed
    for band in bands:
        if statistics.moments[band].count < 2:
            raise ValueError("band %s has too few pixel values in the windows" % band)
    return statistics, counts


class NormalisationError(ValueError):
    """Raised when a required normalisation file is missing or unusable."""


def require_statistics(path, bands, pairs, expected_scheme="per-pair"):
    """Load a statistics file and fail loudly if anything is wrong.

    Called by training and inference whenever the dataset declares pair-specific
    normalisation. There is deliberately no fallback: a run that cannot prove which
    statistics it used is a run whose inputs cannot be reproduced.
    """
    path = Path(path)
    if not path.is_file():
        raise NormalisationError(
            "pair-specific normalisation requires a statistics file; %s does not exist"
            % path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NormalisationError("normalisation file is not readable JSON: %s (%s)"
                                 % (path, error)) from error
    if not isinstance(payload, dict) or "pairs" not in payload:
        raise NormalisationError("normalisation file has no 'pairs' section: %s" % path)
    scheme = str(payload.get("scheme", ""))
    if expected_scheme and expected_scheme not in scheme:
        raise NormalisationError(
            "normalisation scheme %r is not the expected %r scheme" % (scheme, expected_scheme))
    declared = payload.get("band_order")
    if declared is not None and tuple(declared) != tuple(bands):
        raise NormalisationError(
            "normalisation bands %s do not match the model bands %s"
            % (list(declared), list(bands)))
    missing = [pair for pair in pairs if pair not in payload["pairs"]]
    if missing:
        raise NormalisationError(
            "normalisation file has no statistics for pair(s) %s" % ", ".join(missing))
    for pair in pairs:
        entry = payload["pairs"][pair]
        if not isinstance(entry, dict) or set(entry) != set(bands):
            raise NormalisationError(
                "pair %s statistics cover %s, expected %s"
                % (pair, sorted(entry) if isinstance(entry, dict) else entry,
                   list(bands)))
        for band in bands:
            stats = entry[band]
            if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
                raise NormalisationError(
                    "pair %s band %s has no mean/std" % (pair, band))
            mean, std = float(stats["mean"]), float(stats["std"])
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                raise NormalisationError(
                    "pair %s band %s statistics are unusable: mean=%r std=%r"
                    % (pair, band, mean, std))
            if int(stats.get("count", 0)) < 2:
                raise NormalisationError(
                    "pair %s band %s has fewer than two pixels" % (pair, band))
    payload["_file"] = str(path.resolve())
    payload["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


def pooled_statistics(payload, bands):
    """Count-weighted mean and std across pairs, for checkpoint compatibility.

    The checkpoint carries one mean/std pair for legacy consumers. When training uses
    per-pair statistics this pooled value is only a reference: inference selects the
    pair's own statistics from ``normalisation_file``.
    """
    total = 0
    sums = np.zeros(len(bands), dtype=np.float64)
    squares = np.zeros(len(bands), dtype=np.float64)
    for pair, entry in payload["pairs"].items():
        for index, band in enumerate(bands):
            stats = entry[band]
            n = int(stats["count"])
            mean = float(stats["mean"])
            std = float(stats["std"])
            total += n
            sums[index] += mean * n
            squares[index] += (std ** 2 + mean ** 2) * n
    if total == 0:
        raise ValueError("no statistics to pool")
    mean = sums / total
    variance = np.maximum(squares / total - mean ** 2, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
