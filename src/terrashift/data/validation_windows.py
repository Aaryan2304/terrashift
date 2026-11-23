"""Validation windows taken from the inference lattice, not a partition lattice.

Inference writes only the core rows of each strip, so validation has to index the same windows or it
measures a different model from the deployed one. This reuses the inference tiler and keeps three
departures from the training index explicit: windows may cross partition boundaries; a window
contributes only to its strip's core rows; and labels outside the validation split are set to ignore,
so train and test pixels never enter a metric. Training selection stays in
``terrashift.data.local_patches`` with its containment rule.
"""

from __future__ import annotations

import numpy as np

from terrashift.inference.sliding import _window_origins

DEFAULT_BLOCK_ROWS = 512
DEFAULT_HALO_ROWS = 256
IGNORE_VALUE = 255
VALIDATION_CODE = 2

def inference_strips(length, block=DEFAULT_BLOCK_ROWS, halo=DEFAULT_HALO_ROWS):
    """Core and read ranges, matching ``predict_safe_pair``'s strip loop exactly.

    Yields ``(core_start, core_stop, read_start, read_stop)``.
    """
    if block < 1 or halo < 0:
        raise ValueError("block must be positive and halo non-negative")
    for core_start in range(0, length, block):
        core_stop = min(core_start + block, length)
        yield (core_start, core_stop,
               max(0, core_start - halo), min(length, core_stop + halo))

def strip_origins(read_start, read_stop, patch, stride):
    """Row origins inside one strip's read range, using the inference tiler."""
    if read_stop - read_start < patch:
        # Shorter than a patch: the tiler returns a single origin at read_start and
        # the caller pads, exactly as inference does at an image edge.
        return [read_start]
    return [read_start + offset for offset in _window_origins(read_stop - read_start, patch, stride)]

def column_origins(width, patch, stride):
    return list(_window_origins(width, patch, stride))

def window_grid(read_start, read_stop, width, patch, stride):
    """Every ``(row0, col0)`` that inference runs for one strip."""
    rows = strip_origins(read_start, read_stop, patch, stride)
    cols = column_origins(width, patch, stride)
    return [(row0, col0) for row0 in rows for col0 in cols]

def core_slice(window, patch):
    """Rows of a window's prediction that inference actually keeps.

    A window read from a strip's halo covers rows outside the core; inference crops
    them. Returns ``(offset_into_patch, count)``.
    """
    first = max(window["row0"], window["core_start"])
    last = min(window["row0"] + patch, window["core_stop"])
    if last <= first:
        raise ValueError("window %r does not overlap its core rows" % (window,))
    return first - window["row0"], last - first

def validation_target(labels, partition, validation_code=VALIDATION_CODE,
                      ignore=IGNORE_VALUE):
    """0/1 exactly where the pixel is reviewed *and* in the validation split.

    Everything else, train, test, guard, and any 255 label, becomes the ignore
    value. This is what a validation metric must see: no train pixel and no test
    pixel can contribute, even if a window covers them.
    """
    labels = np.asarray(labels)
    partition = np.asarray(partition)
    if labels.shape != partition.shape:
        raise ValueError("labels and partition must share a shape")
    target = np.full(labels.shape, ignore, dtype=np.uint8)
    selected = (partition == validation_code) & (labels != IGNORE_VALUE)
    if not np.all(np.isin(labels[selected], (0, 1))):
        raise ValueError("reviewed validation labels must be 0 or 1")
    target[selected] = labels[selected]
    return target

def select_windows(target, patch, stride, block=DEFAULT_BLOCK_ROWS,
                   halo=DEFAULT_HALO_ROWS, min_reviewed=1,
                   tile_width=None, col_offset=0):
    """Windows inference would run that contain at least ``min_reviewed`` targets.

    ``target`` may be a stripe slice rather than the whole tile, but the tiling must
    still be computed over the **full tile width**, because that is what inference
    runs. Pass ``tile_width`` and ``col_offset`` whenever ``target`` is narrower than
    the tile; the returned ``col0`` is always a tile coordinate.

    ``target`` is the output of :func:`validation_target`, so "reviewed validation
    pixel" means exactly that and nothing else.
    """
    target = np.asarray(target)
    if target.ndim != 2:
        raise ValueError("target must be 2-D")
    if min_reviewed < 1:
        raise ValueError("min_reviewed must be at least 1; empty windows are excluded")
    height, stripe_width = target.shape
    col_offset = int(col_offset)
    width = int(stripe_width + col_offset) if tile_width is None else int(tile_width)
    if width < col_offset + stripe_width:
        raise ValueError("tile_width is smaller than the stripe it must contain")
    windows = []
    for core_start, core_stop, read_start, read_stop in inference_strips(height, block, halo):
        for row0, col0 in window_grid(read_start, read_stop, width, patch, stride):
            # Only the rows inference keeps can contribute, so count there.
            first = max(row0, core_start)
            last = min(row0 + patch, core_stop)
            if last <= first:
                continue
            left = max(col0, col_offset)
            right = min(col0 + patch, col_offset + stripe_width)
            if right - left < 1:
                continue
            count = int((target[first:last, left - col_offset:right - col_offset]
                         != IGNORE_VALUE).sum())
            if count < min_reviewed:
                continue
            windows.append({"row0": int(row0), "col0": int(col0),
                            "core_start": int(core_start), "core_stop": int(core_stop),
                            "reviewed_validation_px": count})
    return windows

def origins_by_pixel(windows, patch, shape, ignore=IGNORE_VALUE):
    """Map each pixel of a small grid to the set of window origins covering it.

    Only for tests and small synthetic grids: the full tile would need a set per
    pixel. Rows outside a window's core are excluded, matching inference.
    """
    grid = {}
    for window in windows:
        offset, count = core_slice(window, patch)
        for row in range(window["row0"] + offset, window["row0"] + offset + count):
            for col in range(window["col0"], min(window["col0"] + patch, shape[1])):
                if row >= shape[0] or col >= shape[1]:
                    continue
                grid.setdefault((row, col), set()).add((window["row0"], window["col0"]))
    return grid

# Sanitised targets and the validation dataset the trainer actually uses

def validation_stripe(partition, validation_code=VALIDATION_CODE):
    """First and last+1 column holding validation pixels.

    The partition is spatial: every validation pixel of a pair shares one column
    range, so a windowed read of that range never touches a test column.
    """
    partition = np.asarray(partition)
    columns = np.flatnonzero((partition == validation_code).any(axis=0))
    if columns.size == 0:
        raise ValueError("partition has no validation columns")
    return int(columns[0]), int(columns[-1]) + 1

def read_validation_target(labels_path, partition_path, ignore=IGNORE_VALUE):
    """Read a validation-only target, never reading a test-label column.

    The label raster is opened with a window covering only the validation stripe.
    Test columns are outside that window, so their pixel values are never read from
    disk, isolation does not depend on masking something already loaded.

    Returns ``(target, column_offset, tile_width)``. ``target`` is 0/1 on reviewed
    validation pixels and ``ignore`` everywhere else inside the stripe. The tile
    width is needed because the tiling is computed over the whole tile even though
    only the stripe is loaded.
    """
    import rasterio
    from rasterio.windows import Window as RasterWindow

    with rasterio.open(partition_path) as handle:
        partition = handle.read(1)
    col0, col1 = validation_stripe(partition)
    with rasterio.open(labels_path) as handle:
        stripe = handle.read(1, window=RasterWindow(col0, 0, col1 - col0, handle.height))
        tile_width = int(handle.width)
        partition_stripe = partition[:, col0:col1]
    target = np.full(stripe.shape, ignore, dtype=np.uint8)
    selected = (partition_stripe == VALIDATION_CODE) & (stripe != ignore)
    if selected.any() and not np.all(np.isin(stripe[selected], (0, 1))):
        raise ValueError("reviewed validation labels must be 0 or 1")
    target[selected] = stripe[selected]
    return target, col0, tile_width

class ValidationLatticeDataset:
    """Validation windows on the inference lattice, targets validation-only.

    Every window is one that ``predict_safe_pair`` would run; ``row_off``/``col_off``
    are tile coordinates so the accumulator can place the prediction, and
    ``core_offset``/``core_count`` say which rows inference would keep.

    ``image_loader(pair, row_off, col_off, size)`` returns ``(image1, image2)`` as
    float arrays of shape ``(C, size, size)``. It exists so tests can supply
    synthetic imagery through the same class the trainer uses.
    """

    def __init__(self, pairs, target, col_offset, tile_width, patch, stride,
                 image_loader, mean=None, std=None, normalisation=None,
                 block=DEFAULT_BLOCK_ROWS, halo=DEFAULT_HALO_ROWS):
        self.pairs = tuple(pairs)
        self.target = np.asarray(target)
        self.col_offset = int(col_offset)
        self.patch = int(patch)
        self.stride = int(stride)
        if self.patch % 8:
            raise ValueError("patch size must be divisible by 8")
        if not 0 < self.stride <= self.patch:
            raise ValueError("stride must be positive and no larger than the patch")
        self.image_loader = image_loader
        self.mean = mean
        self.std = std
        self.normalisation = normalisation
        self.tile_width = int(tile_width)
        self.windows = select_windows(self.target, self.patch, self.stride, block, halo,
                                      tile_width=self.tile_width,
                                      col_offset=self.col_offset)
        if not self.windows:
            raise ValueError("no inference window contains a reviewed validation pixel")

    def __len__(self):
        return len(self.windows)

    def pixel_window(self, row_off):
        """Strip core range for a window, in tile coordinates."""
        for window in self.windows:
            if window["row0"] == row_off:
                return window["core_start"], window["core_stop"]
        raise KeyError(row_off)

    def __getitem__(self, index):
        window = self.windows[index]
        row0, col0 = window["row0"], window["col0"]
        offset, count = core_slice(window, self.patch)
        # Imagery is read at TILE coordinates. Only the target is stripe-relative;
        # passing a stripe-relative column here would read the wrong part of the
        # scene, and near the left edge it would read nothing at all.
        image1, image2 = self.image_loader(row0, col0, self.patch)
        # A window may extend beyond the stripe, so build the patch target on a
        # field of ignore values and copy in only the stripe overlap.
        target = np.full((self.patch, self.patch), IGNORE_VALUE, dtype=np.uint8)
        local_col = col0 - self.col_offset
        src_c0 = max(local_col, 0)
        dst_c0 = max(-local_col, 0)
        take_cols = min(self.patch - dst_c0, self.target.shape[1] - src_c0)
        src_r1 = min(self.patch, self.target.shape[0] - row0)
        if take_cols > 0 and src_r1 > 0:
            target[:src_r1, dst_c0:dst_c0 + take_cols] = self.target[
                row0:row0 + src_r1, src_c0:src_c0 + take_cols]
        valid = target != IGNORE_VALUE
        import torch
        mean, std = self.mean, self.std
        if self.normalisation is not None:
            from terrashift.preprocessing.normalisation import pair_statistics
            mean, std = pair_statistics(self.normalisation, self.pairs[0])
        return {
            "image1": _as_float_tensor(image1, mean, std),
            "image2": _as_float_tensor(image2, mean, std),
            "mask": torch.from_numpy(np.where(valid, target, 0).astype(np.int64)),
            "valid_mask": torch.from_numpy(valid),
            "row_off": int(row0), "col_off": int(col0),
            "core_offset": int(offset), "core_count": int(count),
            "name": "lattice_r%d_c%d" % (row0, col0),
        }

def _as_float_tensor(array, mean, std):
    """Z-score with the given statistics; invalid pixels become neutral zero.

    ``normalise_reflectance`` zeroes NaN and infinite input *after* standardising,
    so a nodata pixel cannot arrive at the model as a large negative z-score.
    """
    import torch
    from terrashift.preprocessing.normalisation import normalise_reflectance

    values = np.asarray(array, dtype=np.float32)
    if mean is None:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        values = normalise_reflectance(values, mean, std)
    return torch.from_numpy(values)
