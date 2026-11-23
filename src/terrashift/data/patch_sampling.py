"""Patch-grid sampling helpers shared by the audit and comparison scripts.

Extracted from ``scripts/compare_patch_sampling_v2.py``; the bodies are unchanged.
The functions operate on arrays and raster handles passed in by the caller, so
importing this module reads no data and creates no files.
"""

from __future__ import annotations

import numpy as np
import rasterio

__all__ = ["SPLITS", "PairData", "patch_origins", "build_pool", "select_plain"]

SPLITS = {1: "train", 2: "validation", 3: "test"}


# One canonical grid implementation: the validated version with type hints and
# argument checks. The unvalidated copy here produced negative origins for a
# scene smaller than the patch (e.g. (3, 8, 8) -> [-5]) and accepted stride >
# patch size; on every valid input the sequences are provably identical.
from terrashift.data.patch_grid import patch_origins


class PairData:
    """One pair's label and partition rasters, with summed-area tables."""

    def __init__(self, pair, label_path, partition_path):
        self.pair = pair
        with rasterio.open(label_path) as dataset:
            self.label = dataset.read(1)
            self.shape = dataset.shape
        with rasterio.open(partition_path) as dataset:
            partition = dataset.read(1)
        if partition.shape != self.shape:
            raise SystemExit("label and partition grids differ for pair %s" % pair)
        # The partitions are x-stripes, so containment is a column property. Verify
        # that assumption instead of trusting it.
        first_row = partition[0:1, :]
        if not np.array_equal(partition, np.broadcast_to(first_row, partition.shape)):
            raise SystemExit("partition raster for pair %s is not a column stripe" % pair)
        self.column_code = first_row[0].astype(np.uint8)
        self.change = self.label == 1
        self.reviewed = self.label != 255
        self.no_change = self.reviewed & ~self.change
        self.table_reviewed = _integral(self.reviewed)
        self.table_change = _integral(self.change)

    def split_of(self, col0, size):
        """Split name if the column range sits in one non-guard stripe, else None."""
        codes = np.unique(self.column_code[col0:col0 + size])
        if codes.size != 1 or codes[0] == 0:
            return None
        return SPLITS.get(int(codes[0]))

    def window(self, row0, col0, size):
        reviewed = _window_sum(self.table_reviewed, row0, col0, size)
        change = _window_sum(self.table_change, row0, col0, size)
        return {
            "pair": self.pair, "row0": int(row0), "col0": int(col0), "size": int(size),
            "reviewed": reviewed, "change": change, "no_change": reviewed - change,
            "reviewed_fraction": reviewed / float(size * size),
        }


def build_pool(data, size, stride, include_splits=("train", "validation")):
    pool = []
    for row0 in patch_origins(data.shape[0], size, stride):
        for col0 in patch_origins(data.shape[1], size, stride):
            split = data.split_of(col0, size)
            if split is None or split not in include_splits:
                continue
            window = data.window(row0, col0, size)
            window["split"] = split
            window["category"] = category(window)
            pool.append(window)
    return pool


def select_plain(pool, threshold):
    return [w for w in pool if w["reviewed"] > 0
            and w["reviewed_fraction"] >= threshold]
