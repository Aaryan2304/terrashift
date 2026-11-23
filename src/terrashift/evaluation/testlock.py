"""Reading the locked test target from the label and partition rasters.

Extracted from ``scripts/evaluation/evaluate_locked_test.py``. The function body is unchanged and
performs no dataset access at import time: the rasters are opened only when the function
is called, so this module imports without ``SENTINEL_DATA_ROOT``.
"""

from __future__ import annotations

import rasterio
from rasterio.windows import Window

from terrashift.data.validation_windows import (
    IGNORE_VALUE,
    validation_stripe,
    validation_target,
)

__all__ = ["TEST_CODE", "read_test_target"]

TEST_CODE = 3


def read_test_target(labels_path, partition_path, ignore=IGNORE_VALUE):
    """0/1 where a pixel is reviewed *and* in the test split, else ignore.

    Reads the label raster through a window covering only the test stripe, so train and
    validation label columns are never opened.
    """
    with rasterio.open(partition_path) as handle:
        partition = handle.read(1)
        height, width = partition.shape
    col0, col1 = validation_stripe(partition, validation_code=TEST_CODE)
    with rasterio.open(labels_path) as handle:
        labels = handle.read(1, window=Window(col0, 0, col1 - col0, height))
    partition_stripe = partition[:, col0:col1]
    target = validation_target(labels, partition_stripe, validation_code=TEST_CODE,
                              ignore=ignore)
    return target, col0, width
