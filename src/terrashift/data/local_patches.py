"""Windowed local Sentinel-2 patches gated by manual labels and partitions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window
from torch.utils.data import Dataset

from terrashift.data.manifest import load_manifest, pair_config
from terrashift.registration import align_to_reference
from terrashift.preprocessing.normalisation import normalise_reflectance, pair_statistics
from terrashift.data.safeloader import SafeProduct, BAND_ORDER
from terrashift.data.patch_grid import patch_origins

BANDS = ("B02", "B03", "B04", "B08")
SPLIT_CODES = {"train": 1, "validation": 2, "test": 3}


def select_pairs(available, requested=None):
    """Which pairs to index.

    ``requested`` is None or a list. None means every pair in the manifest, which
    keeps a production build strict: a missing label raster fails loudly instead
    of silently indexing half the dataset. A list restricts the build to pairs
    whose labels exist, which is what a single-pair build needs while the other
    pair is still being annotated.
    """
    available = list(available)
    if requested is None:
        return sorted(available)
    unknown = [pair for pair in requested if pair not in available]
    if unknown:
        raise ValueError("pairs not in the manifest: %s" % ", ".join(unknown))
    return list(requested)


def resolve_split_thresholds(min_labeled_fraction, splits):
    """Resolve a scalar or a split-to-threshold mapping into one value per split.

    A scalar keeps the historical behaviour and its historical constraint: it must
    be strictly greater than zero, because a zero threshold on a scalar call was
    never a supported input. A mapping may contain zero for any split, which is how
    "at least one reviewed pixel" is expressed for validation; the empty-window
    guard in ``build_local_patch_index`` still rejects windows with no reviewed
    pixels, so a zero threshold cannot admit an empty window.

    Unknown split names, missing splits and out-of-range values are errors rather
    than silent defaults. ``test`` may not be configured here at all: test is
    excluded from the sampling workflow and is unlocked only by ``include_test``.
    """
    if isinstance(min_labeled_fraction, dict):
        unknown = set(min_labeled_fraction) - set(SPLIT_CODES)
        if unknown:
            raise ValueError(
                "min_labeled_fraction names unknown splits: %s (allowed: %s)"
                % (sorted(unknown), sorted(SPLIT_CODES)))
        if "test" in min_labeled_fraction:
            raise ValueError(
                "the test split must stay excluded from sampling; "
                "do not configure a threshold for it")
        missing = set(splits) - set(min_labeled_fraction) - {"test"}
        if missing:
            raise ValueError(
                "min_labeled_fraction has no threshold for split(s): %s"
                % sorted(missing))
        resolved = {}
        for split in set(splits) - {"test"}:
            value = min_labeled_fraction[split]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "threshold for split %r must be a number, got %r"
                    % (split, value))
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "threshold for split %r must be in [0, 1], got %r"
                    % (split, value))
            resolved[split] = value
        if "test" in splits:
            # Test is unlocked only by include_test, never by the mapping; when it is
            # unlocked it follows the training threshold rather than inventing one.
            resolved["test"] = resolved["train"]
        return resolved

    if isinstance(min_labeled_fraction, bool) or not isinstance(
            min_labeled_fraction, (int, float)):
        raise ValueError(
            "min_labeled_fraction must be a number or a split mapping, got %r"
            % (min_labeled_fraction,))
    if not 0.0 < float(min_labeled_fraction) <= 1.0:
        raise ValueError("min_labeled_fraction must be in (0, 1]")
    return {split: float(min_labeled_fraction) for split in splits}


def build_local_patch_index(manifest_path: Path, label_dir: Path,
                            partition_dir: Path, patch_size: int = 256,
                            stride: int = 128,
                            min_labeled_fraction: float | dict[str, float] = 0.05,
                            include_test: bool = False,
                            pairs: list[str] | None = None) -> list[dict]:
    """Index split-contained windows; test label pixels are excluded by default.

    ``min_labeled_fraction`` is either one number applied to every split, or a
    mapping such as ``{"train": 0.002, "validation": 0.0}``. Training is selected
    on label density; validation is selected for coverage, so its threshold is
    normally zero and the only requirement is at least one reviewed binary pixel.

    ``pairs`` restricts the build to pairs whose label rasters exist. The default
    is every pair in the manifest, which fails loudly on a missing raster rather
    than silently indexing half the dataset.
    """
    if patch_size <= 0 or stride <= 0 or stride > patch_size:
        raise ValueError("patch_size and stride must be positive, with stride <= patch_size")
    splits = {name: code for name, code in SPLIT_CODES.items()
              if name != "test" or include_test}
    thresholds = resolve_split_thresholds(min_labeled_fraction, splits)
    manifest = load_manifest(manifest_path)
    rows = []
    # ``pairs`` restricts the build to the pairs whose labels exist. The default
    # stays strict: every pair must have a label raster, so a production run
    # fails loudly rather than silently indexing half the dataset.
    selected = select_pairs(manifest["pairs"], pairs)
    for pair in selected:
        label_path = label_dir / ("manual_labels_%s.tif" % pair)
        partition_path = partition_dir / ("pair_%s_partition_10m.tif" % pair)
        if not label_path.is_file():
            raise FileNotFoundError("manual label raster missing: %s" % label_path)
        with rasterio.open(label_path) as labels, rasterio.open(partition_path) as parts:
            if labels.crs != parts.crs or labels.transform != parts.transform:
                raise ValueError("manual labels and partitions are not on the same grid: %s" % pair)
            if labels.width != parts.width or labels.height != parts.height or labels.res != (10.0, 10.0):
                raise ValueError("expected matching 10 m label and partition grids: %s" % pair)
            pair_cfg = pair_config(manifest, pair)
            for split, code in splits.items():
                for row_off in patch_origins(labels.height, patch_size, stride):
                    for col_off in patch_origins(labels.width, patch_size, stride):
                        window = Window(col_off, row_off, patch_size, patch_size)
                        part_window = parts.read(1, window=window)
                        if part_window.shape != (patch_size, patch_size) or not np.all(part_window == code):
                            continue
                        # Only read label pixels for windows in train/validation,
                        # unless the caller explicitly unlocks test indexing.
                        label_window = labels.read(1, window=window)
                        values = set(int(x) for x in np.unique(label_window))
                        if not values <= {0, 1, 255}:
                            raise ValueError("unsupported manual label values for %s: %s" % (pair, values))
                        known = label_window != 255
                        # The empty-window guard is independent of the threshold.
                        # No window lacking a reviewed binary pixel is ever indexed,
                        # even when this split's threshold is zero, and only 0/1
                        # count as reviewed: 255 is uncertain or unreviewed.
                        if not known.any():
                            continue
                        if not np.any(label_window[known] == 0) and not np.any(label_window[known] == 1):
                            continue
                        if known.mean() < thresholds[split]:
                            continue
                        rows.append({
                            "pair": pair, "split": split,
                            "row_off": row_off, "col_off": col_off,
                            "height": patch_size, "width": patch_size,
                            "label_path": str(label_path.resolve()),
                            "partition_path": str(partition_path.resolve()),
                            "partition_code": code,
                            "labeled_pixels": int(known.sum()),
                            "positive_pixels": int((label_window == 1).sum()),
                            "negative_pixels": int((label_window == 0).sum()),
                            "manual_label_fraction": float(known.mean()),
                            "crs": str(parts.crs),
                            "date1": pair_cfg["date1"], "date2": pair_cfg["date2"],
                        })
    return rows


def write_local_patch_index(rows: list[dict], output_path: Path,
                            patch_size: int, stride: int,
                            min_labeled_fraction,
                            include_test: bool = False) -> None:
    row_splits = {row.get("split") for row in rows}
    unknown_splits = row_splits - set(SPLIT_CODES)
    if unknown_splits:
        raise ValueError("local patch index has unsupported splits: %s" % sorted(unknown_splits))
    if not include_test and "test" in row_splits:
        raise ValueError("cannot write test rows when include_test is false")
    for row in rows:
        if row.get("partition_code") != SPLIT_CODES[row["split"]]:
            raise ValueError("local patch row has a partition code inconsistent with its split")
    # Record the threshold actually applied to each split, so the index cannot be
    # read without the policy that produced it.
    # Resolve against every split the index may represent, not only the ones that
    # happened to produce rows, so a scalar policy is recorded for both splits even
    # when one of them yields no windows.
    configured = {"train", "validation"} | {row["split"] for row in rows}
    resolved = resolve_split_thresholds(
        min_labeled_fraction,
        {name: code for name, code in SPLIT_CODES.items() if name in configured})
    meta_path = output_path.with_suffix(".json")
    existing = [path for path in (output_path, meta_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite local patch index outputs: %s"
            % ", ".join(str(path) for path in existing))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair", "split", "row_off", "col_off", "height", "width",
              "label_path", "partition_path", "partition_code", "labeled_pixels", "positive_pixels",
              "negative_pixels", "manual_label_fraction", "crs", "date1", "date2"]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "index_csv": str(output_path.resolve()),
        "patch_size": patch_size, "stride": stride,
        "min_labeled_fraction": (min_labeled_fraction
                                 if isinstance(min_labeled_fraction, dict)
                                 else float(min_labeled_fraction)),
        "min_labeled_fraction_by_split": resolved,
        "empty_window_policy": ("rejected: a window must contain at least one "
                                "reviewed binary pixel (0 or 1); 255 never counts"),
        "patch_count_by_split": {
            split: sum(row["split"] == split for row in rows)
            for split in SPLIT_CODES
        },
        "pair_count": len({row["pair"] for row in rows}),
        "uses_human_labels_only": True,
        "uses_osm_labels": False,
        "buffer_or_cross_split_windows_included": False,
        "test_index_included": bool(include_test),
        "test_label_pixels_read": bool(include_test),
        "test_labels_used_for_training": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _read_padded_band(product: SafeProduct, band: str, row: int, col: int,
                      size: int, halo: int) -> np.ndarray:
    path = product.band_path(band)
    with rasterio.open(path) as src:
        y0, x0 = row - halo, col - halo
        y1, x1 = row + size + halo, col + size + halo
        r0, c0 = max(0, y0), max(0, x0)
        r1, c1 = min(src.height, y1), min(src.width, x1)
        values, _, _ = product.read_band(
            band, window=((r0, r1), (c0, c1)))
    pads = ((r0 - y0, y1 - r1), (c0 - x0, x1 - c1))
    if any(a or b for a, b in pads):
        mode = "reflect" if values.shape[0] > 1 and values.shape[1] > 1 else "edge"
        values = np.pad(values, pads, mode=mode)
    if values.shape != (size + 2 * halo, size + 2 * halo):
        raise RuntimeError("padded band window has the wrong shape")
    return values


def read_pair_window(date1, date2, row_off: int, col_off: int, size: int,
                     bands, shift=(0.0, 0.0)):
    """Both dates of one window, date 2 aligned to date 1.

    Shared by the training dataset and the inference-lattice validation dataset so
    the two cannot drift apart in registration or band order. Returns two
    ``(len(bands), size, size)`` float32 arrays.
    """
    dx, dy = shift
    halo = max(4, int(np.ceil(max(abs(dx), abs(dy)))) + 2)
    bands1, bands2 = [], []
    for band in bands:
        first, _, _ = date1.read_band(
            band, window=((row_off, row_off + size), (col_off, col_off + size)))
        moving = _read_padded_band(date2, band, row_off, col_off, size, halo)
        moving = align_to_reference(moving, dx, dy)
        moving = moving[halo:halo + size, halo:halo + size]
        bands1.append(first)
        bands2.append(moving)
    return np.stack(bands1).astype(np.float32), np.stack(bands2).astype(np.float32)


def pair_image_loader(manifest, registration_path, bands=BANDS, products=None):
    """Return ``loader(pair, row_off, col_off, size)`` for the lattice dataset."""
    cache = {} if products is None else products
    manifest_obj = load_manifest(manifest)
    registration = json.loads(Path(registration_path).read_text(encoding="utf-8"))
    shifts = {}
    for pair, value in registration.items():
        applied = value["initial"].get("applied_shift_px")
        shifts[pair] = (0.0, 0.0) if applied is None else tuple(map(float, applied))

    def loader(pair, row_off, col_off, size):
        if pair not in cache:
            cfg = pair_config(manifest_obj, pair)
            cache[pair] = (SafeProduct(cfg["path_date1"]), SafeProduct(cfg["path_date2"]))
        date1, date2 = cache[pair]
        return read_pair_window(date1, date2, row_off, col_off, size, tuple(bands),
                                shifts.get(pair, (0.0, 0.0)))

    return loader


class LocalManualPatchDataset(Dataset):
    """Read labelled local SAFE patches on demand; unknown pixels are ignored."""

    def __init__(self, index_path: Path, split: str, manifest_path: Path,
                 registration_path: Path, mean=None, std=None, transform=None,
                 bands=BANDS, normalisation=None):
        if split not in SPLIT_CODES:
            raise ValueError("split must be train, validation, or test")
        self.index_path = Path(index_path)
        self.split = split
        with self.index_path.open(newline="", encoding="utf-8") as fh:
            self.rows = [row for row in csv.DictReader(fh) if row["split"] == split]
        if not self.rows:
            raise ValueError("no %s local patches in %s" % (split, index_path))
        self.manifest = load_manifest(manifest_path)
        self.bands = tuple(bands)
        if not self.bands or any(band not in BAND_ORDER for band in self.bands):
            raise ValueError("unknown or empty local band list")
        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = None if std is None else torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must both be supplied or both omitted")
        # Per-pair statistics take precedence over a single pair of arrays: the two
        # pairs are separate scenes with different reflectance distributions.
        self.normalisation = normalisation
        if self.mean is not None:
            if self.mean.shape != (len(self.bands), 1, 1) or self.std.shape != self.mean.shape:
                raise ValueError("normalization statistics do not match bands")
            if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or torch.any(self.std <= 0):
                raise ValueError("normalization statistics must be finite with positive std")
        registration = json.loads(Path(registration_path).read_text(encoding="utf-8"))
        self.shifts = {}
        for pair, value in registration.items():
            applied = value["initial"].get("applied_shift_px")
            self.shifts[pair] = (0.0, 0.0) if applied is None else tuple(map(float, applied))
        self.transform = transform
        self._products = {}

    def __len__(self):
        return len(self.rows)

    def _pair_products(self, pair):
        if pair not in self._products:
            cfg = pair_config(self.manifest, pair)
            self._products[pair] = (SafeProduct(cfg["path_date1"]),
                                    SafeProduct(cfg["path_date2"]))
        return self._products[pair]

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        pair = row["pair"]
        row_off, col_off = int(row["row_off"]), int(row["col_off"])
        size = int(row["height"])
        date1, date2 = self._pair_products(pair)
        dx, dy = self.shifts.get(pair, (0.0, 0.0))
        image1_np, image2_np = read_pair_window(
            date1, date2, row_off, col_off, size, self.bands, (dx, dy))
        raw1 = np.nan_to_num(image1_np, nan=0.0, posinf=0.0, neginf=0.0)
        raw2 = np.nan_to_num(image2_np, nan=0.0, posinf=0.0, neginf=0.0)
        image1 = torch.from_numpy(raw1)
        image2 = torch.from_numpy(raw2)
        label_path = Path(row["label_path"])
        with rasterio.open(label_path) as src:
            raw_label = src.read(1, window=Window(col_off, row_off, size, size))
        valid_mask = (raw_label != 255) & np.isfinite(image1_np).all(axis=0) & np.isfinite(image2_np).all(axis=0)
        if np.any(~np.isin(raw_label[valid_mask], (0, 1))):
            raise ValueError("manual mask has an unsupported class")
        mask = np.where(valid_mask, raw_label, 0).astype(np.int64)
        if self.normalisation is not None:
            pair_mean, pair_std = pair_statistics(self.normalisation, pair)
            image1_np = normalise_reflectance(image1_np, pair_mean, pair_std)
            image2_np = normalise_reflectance(image2_np, pair_mean, pair_std)
            image1 = torch.from_numpy(image1_np)
            image2 = torch.from_numpy(image2_np)
        elif self.mean is not None:
            image1_np = normalise_reflectance(
                image1_np, self.mean.numpy(), self.std.numpy())
            image2_np = normalise_reflectance(
                image2_np, self.mean.numpy(), self.std.numpy())
            image1 = torch.from_numpy(image1_np)
            image2 = torch.from_numpy(image2_np)
        sample = {
            "image1": image1, "image2": image2,
            "mask": torch.from_numpy(mask),
            "valid_mask": torch.from_numpy(valid_mask),
            "name": "%s_%s_r%d_c%d" % (pair, self.split, row_off, col_off),
            "pair": pair, "split": self.split,
            # Kept so unique-pixel stitching can place the patch on the pair
            # grid. Only exact while no geometric transform is applied, which
            # is the case for validation and test.
            "row_off": int(row_off), "col_off": int(col_off),
            "label_path": str(label_path),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def local_training_statistics(dataset: LocalManualPatchDataset) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated: patch-weighted channel statistics.

    Retained for comparison only. Overlapping patches give an interior pixel
    several votes, so this function is biased towards the interior of the
    annotated area. The trainer uses
    :func:`terrashift.evaluation.unique_regions.weighted_training_statistics` instead.
    """
    if dataset.split != "train":
        raise ValueError("training statistics require the train split")
    sums = np.zeros(len(dataset.bands), dtype=np.float64)
    squares = np.zeros(len(dataset.bands), dtype=np.float64)
    counts = np.zeros(len(dataset.bands), dtype=np.int64)
    for index in range(len(dataset)):
        sample = dataset[index]
        valid = sample["valid_mask"].numpy()
        for i, key in enumerate(("image1", "image2")):
            image = sample[key].numpy()
            for band in range(len(dataset.bands)):
                values = image[band][valid].astype(np.float64)
                sums[band] += values.sum(dtype=np.float64)
                squares[band] += np.square(values).sum(dtype=np.float64)
                counts[band] += values.size
    if np.any(counts == 0):
        raise ValueError("training annotations contain no valid pixels")
    mean = sums / counts
    variance = squares / counts - mean * mean
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)


def local_change_fraction(index_path: Path, split: str = "train") -> float:
    """Deprecated: patch-weighted positive fraction.

    Summing the index columns counts an overlapped pixel once per patch that
    covers it. The trainer derives the fraction from
    :func:`terrashift.evaluation.unique_regions.weighted_training_statistics` instead.
    """
    with Path(index_path).open(newline="", encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh) if row["split"] == split]
    positive = sum(int(row["positive_pixels"]) for row in rows)
    negative = sum(int(row["negative_pixels"]) for row in rows)
    total = positive + negative
    return positive / total if total else float("nan")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build the local manual-label patch index.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--partition-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--min-labeled-fraction", type=float, default=None)
    parser.add_argument("--validation-min-labeled-fraction", type=float, default=None,
                        help="validation coverage threshold; 0 selects every window "
                             "holding at least one reviewed pixel. Omit to apply the "
                             "single --min-labeled-fraction to every split.")
    parser.add_argument("--include-test", action="store_true",
                        help="read test labels only after model selection")
    parser.add_argument("--pairs", default=None,
                        help="comma-separated pairs to index, e.g. A. "
                             "Default: every pair in the manifest, which fails if "
                             "any label raster is missing.")
    args = parser.parse_args()
    pairs = (None if args.pairs is None
             else [item.strip() for item in args.pairs.split(",") if item.strip()])
    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg["data"]
    manifest = args.manifest or Path(data_cfg["manifest"])
    label_dir = args.label_dir or Path(data_cfg["manual_label_dir"])
    partition_dir = args.partition_dir or Path(data_cfg["local_partition_dir"])
    output = args.output or Path(data_cfg["local_patch_index"])
    patch_size = args.patch_size if args.patch_size is not None else int(data_cfg["local_patch_size"])
    stride = args.stride if args.stride is not None else int(data_cfg["local_patch_stride"])
    min_fraction = (args.min_labeled_fraction if args.min_labeled_fraction is not None
                    else float(data_cfg["local_min_labeled_fraction"]))
    if args.validation_min_labeled_fraction is not None:
        min_fraction = {"train": min_fraction,
                        "validation": args.validation_min_labeled_fraction}
    rows = build_local_patch_index(manifest, label_dir, partition_dir,
                                   patch_size, stride, min_fraction,
                                   include_test=args.include_test, pairs=pairs)
    write_local_patch_index(rows, output, patch_size, stride, min_fraction,
                            include_test=args.include_test)
    print("wrote", output, "rows", len(rows))
    for split in SPLIT_CODES:
        subset = [row for row in rows if row["split"] == split]
        print(split, "patches", len(subset), "pairs", sorted({row["pair"] for row in subset}))
    print("No automatic labels were created.")


if __name__ == "__main__":
    main()
