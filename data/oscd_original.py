"""Dataset loader for the original, region-level OSCD release.

Unlike OSCD100, this loader reads the authors' 14/10 geographic split and the
registered ``imgs_N_rect`` 10 m grids. The rectified rasters have no CRS; this
module uses them only as aligned image grids, never as geographic coordinates.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window
from torch.utils.data import Dataset

BAND_ORDER = ("B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B8A", "B09", "B10", "B11", "B12")
DEFAULT_BANDS = ("B02", "B03", "B04", "B08")
DEFAULT_SPLIT_CONFIG = Path("configs/oscd_region_splits.yaml")


def _city_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    result = [value.strip() for value in text.replace("\n", "").split(",")]
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError("invalid region list: %s" % path)
    return result


def _label_path(data_root: Path, split: str, city: str) -> Path:
    group = "Train" if split in ("train", "validation") else "Test"
    # The excluded Mumbai labels remain in the official train label directory.
    if split == "excluded":
        group = "Train"
    return (data_root / ("Onera Satellite Change Detection dataset - %s Labels" % group)
            / city / "cm" / (city + "-cm.tif"))


def validate_region_splits(data_root: Path, split_config_path: Path) -> dict:
    """Validate custom train/validation and official test city lists."""
    images_root = data_root / "Onera Satellite Change Detection dataset - Images"
    with split_config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    actual_train = _city_list(images_root / "train.txt")
    actual_test = _city_list(images_root / "test.txt")
    actual_all = _city_list(images_root / "all.txt")
    configured_official_train = list(cfg["official_train"])
    if len(configured_official_train) != len(set(configured_official_train)):
        raise ValueError("configured official_train contains duplicate regions")
    if set(configured_official_train) != set(actual_train):
        raise ValueError("configured official_train does not match OSCD train.txt")
    if cfg["official_test"] != actual_test:
        raise ValueError("configured official_test order/content differs from OSCD test.txt")
    if set(actual_train) & set(actual_test) or set(actual_train) | set(actual_test) != set(actual_all):
        raise ValueError("official OSCD train/test lists are not a disjoint cover of all.txt")

    train = list(cfg["train"])
    validation = list(cfg["validation"])
    test = list(cfg["test"])
    excluded = set(cfg.get("excluded", {}))
    for name, cities in (("train", train), ("validation", validation), ("test", test)):
        if len(cities) != len(set(cities)):
            raise ValueError("configured %s split contains duplicate regions" % name)
    groups = (set(train), set(validation), set(test), excluded)
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            if left & right:
                raise ValueError("OSCD train/validation/test/excluded groups overlap")
    if set(train) | set(validation) | excluded != set(actual_train):
        raise ValueError("configured train/validation/excluded do not cover official training regions")
    if test != actual_test:
        raise ValueError("test split must equal official test.txt exactly and in order")
    if not set(train) <= set(actual_train) or not set(validation) <= set(actual_train):
        raise ValueError("train and validation regions must come from official train.txt")
    if excluded - set(actual_train):
        raise ValueError("excluded regions must come from official train.txt")
    expected_validation = deterministic_validation_regions(
        actual_train, excluded, count=len(validation))
    if validation != expected_validation:
        raise ValueError("validation regions do not match the recorded stable-hash rule")
    for split, cities in (("train", train), ("validation", validation), ("test", test),
                          ("excluded", sorted(excluded))):
        for city in cities:
            if not (images_root / city / "imgs_1_rect").is_dir():
                raise FileNotFoundError("missing rectified scene for %s" % city)
            if not _label_path(data_root, split, city).is_file():
                raise FileNotFoundError("missing %s mask for %s" % (split, city))
    return {"config": cfg, "images_root": images_root, "train": train,
            "validation": validation, "test": test, "excluded": sorted(excluded)}


def patch_origins(length: int, patch_size: int, stride: int) -> list[int]:
    """Return unique windows that cover the full dimension, including its end."""
    if patch_size <= 0 or stride <= 0 or stride > patch_size:
        raise ValueError("patch size and stride must be positive, with stride <= patch size")
    if length < patch_size:
        return []
    origins = list(range(0, length - patch_size + 1, stride))
    edge = length - patch_size
    if not origins or origins[-1] != edge:
        origins.append(edge)
    return sorted(set(origins))


def build_patch_index(data_root: Path, split_config_path: Path,
                      patch_size: int = 128, stride: int = 128) -> list[dict]:
    """Build one row per window without reading any label pixels."""
    split_data = validate_region_splits(data_root, split_config_path)
    images_root = split_data["images_root"]
    rows = []
    for split in ("train", "validation", "test"):
        for city in split_data[split]:
            scene = images_root / city
            with rasterio.open(scene / "imgs_1_rect" / "B02.tif") as src:
                height, width = src.height, src.width
                if src.crs is not None:
                    raise ValueError("expected rectified OSCD grid without CRS: %s" % city)
                if src.res != (1.0, 1.0):
                    raise ValueError("unexpected rectified pixel grid for %s: %s" % (city, src.res))
            image1 = scene / "imgs_1_rect"
            image2 = scene / "imgs_2_rect"
            for row_off in patch_origins(height, patch_size, stride):
                for col_off in patch_origins(width, patch_size, stride):
                    rows.append({
                        "city": city, "split": split,
                        "row_off": row_off, "col_off": col_off,
                        "height": patch_size, "width": patch_size,
                        "image1_dir": str(image1.resolve()),
                        "image2_dir": str(image2.resolve()),
                        "label_path": str(_label_path(data_root, split, city).resolve()),
                        "label_encoding": "binary encoding auto-detected from each mask at load time",
                    })
    return rows


def write_patch_index(rows: list[dict], path: Path, patch_size: int, stride: int) -> str:
    """Write a reproducible CSV patch index and sidecar metadata.

    Returns ``"unchanged"`` when the file already holds exactly these rows. The
    caller can then keep the existing file untouched: rebuilding the index on
    every training run must not rewrite a committed artefact, and a rebuild
    that would change the content is refused instead of silently replacing it.
    """
    path = Path(path)
    fields = ["city", "split", "row_off", "col_off", "height", "width",
              "image1_dir", "image2_dir", "label_path", "label_encoding"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    rendered = buffer.getvalue()
    meta = {
        "index_csv": str(path.resolve()),
        "patch_size": patch_size,
        "stride": stride,
        "patch_count_by_split": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "region_count_by_split": {
            split: len({row["city"] for row in rows if row["split"] == split})
            for split in ("train", "validation", "test")
        },
        "official_test_regions": sorted({row["city"] for row in rows if row["split"] == "test"}),
        "test_labels_read_during_index_build": False,
    }
    meta_text = json.dumps(meta, indent=2) + "\n"

    if path.exists():
        existing = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if existing != rendered.replace("\r\n", "\n"):
            raise FileExistsError(
                "refusing to overwrite an existing patch index with different content: %s\n"
                "Delete it deliberately, or pass a different --output path." % path)
        existing_meta = (path.with_suffix(".json").read_text(encoding="utf-8")
                         if path.with_suffix(".json").exists() else None)
        if existing_meta == meta_text:
            return "unchanged"
        # The rows match; only the sidecar is missing or stale, so refresh it.
        path.with_suffix(".json").write_text(meta_text, encoding="utf-8")
        return "metadata_rewritten"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(rendered)
    path.with_suffix(".json").write_text(meta_text, encoding="utf-8")
    return "written"


def read_label_encoding(path: Path) -> tuple[int, int, str]:
    """Inspect the complete small mask and return no-change/change values."""
    with rasterio.open(path) as src:
        values = set(int(value) for value in np.unique(src.read(1)))
    if values == {1, 2}:
        return 1, 2, "observed_1_no_change_2_change"
    if values == {0, 1}:
        return 0, 1, "documented_0_no_change_1_change"
    raise ValueError("unsupported label values %s in %s" % (sorted(values), path))


def read_band_window(directory: Path, band: str, window: Window) -> np.ndarray:
    path = directory / (band + ".tif")
    with rasterio.open(path) as src:
        return src.read(1, window=window).astype(np.float32)


def read_mask_window(path: Path, window: Window, encoding: tuple[int, int, str]) -> np.ndarray:
    no_change, change, _ = encoding
    with rasterio.open(path) as src:
        raw = src.read(1, window=window)
    valid_values = set(int(value) for value in np.unique(raw))
    allowed = {no_change, change}
    if not valid_values <= allowed:
        raise ValueError("unexpected label values %s in %s" % (sorted(valid_values), path))
    return (raw == change).astype(np.uint8)


def compute_training_statistics(rows: Iterable[dict], bands=DEFAULT_BANDS) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-band mean/std from the training scenes only."""
    train_rows = [row for row in rows if row["split"] == "train"]
    if not train_rows:
        raise ValueError("patch index contains no train rows")
    cities = sorted({row["city"] for row in train_rows})
    sums = np.zeros(len(bands), dtype=np.float64)
    squares = np.zeros(len(bands), dtype=np.float64)
    counts = np.zeros(len(bands), dtype=np.int64)
    for city in cities:
        row = next(row for row in train_rows if row["city"] == city)
        for directory in (Path(row["image1_dir"]), Path(row["image2_dir"])):
            for i, band in enumerate(bands):
                with rasterio.open(directory / (band + ".tif")) as src:
                    for _, window in src.block_windows(1):
                        image = src.read(1, window=window).astype(np.float64)
                        valid = image > 0
                        values = image[valid]
                        if values.size:
                            sums[i] += values.sum(dtype=np.float64)
                            squares[i] += np.square(values).sum(dtype=np.float64)
                            counts[i] += values.size
    if np.any(counts == 0):
        raise ValueError("a training band has no valid positive-valued pixels")
    means = sums / counts
    variances = squares / counts - means * means
    stds = np.sqrt(np.maximum(variances, 1e-12))
    return means.astype(np.float32), stds.astype(np.float32)


class OSCDOriginalPatchDataset(Dataset):
    """Map-style dataset of 128 px patches from original OSCD region rasters."""

    def __init__(self, index_path: Path, split: str, bands=DEFAULT_BANDS,
                 mean=None, std=None, transform=None, regions=None):
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        self.index_path = Path(index_path)
        self.split = split
        self.bands = tuple(bands)
        if not self.bands or any(band not in BAND_ORDER for band in self.bands):
            raise ValueError("unknown or empty band list: %s" % (self.bands,))
        with self.index_path.open(newline="", encoding="utf-8") as fh:
            self.rows = [row for row in csv.DictReader(fh) if row["split"] == split]
        if regions is not None:
            # Restrict to named regions, so a caller can report metrics per
            # city on a split that spans several cities.
            wanted = set(regions)
            self.rows = [row for row in self.rows if row["city"] in wanted]
            missing = wanted - {row["city"] for row in self.rows}
            if missing:
                raise ValueError("requested regions are absent from the %s split: %s"
                                 % (split, sorted(missing)))
        if not self.rows:
            raise ValueError("no %s patches in %s" % (split, self.index_path))
        if mean is None or std is None:
            raise ValueError("mean and std from the training split are required")
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape != (len(self.bands),) or std.shape != mean.shape:
            raise ValueError("normalization statistics do not match the selected bands")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("normalization statistics must be finite with positive std")
        self.mean = torch.from_numpy(mean).view(-1, 1, 1)
        self.std = torch.from_numpy(std).view(-1, 1, 1)
        self.transform = transform
        self._encodings: dict[str, tuple[int, int, str]] = {}

    def __len__(self):
        return len(self.rows)

    def change_fraction(self) -> float:
        """Changed-pixel fraction across unique regions in this split."""
        changed = 0
        total = 0
        for city, row in self._one_row_per_city().items():
            path = Path(row["label_path"])
            encoding = self._encoding(path)
            with rasterio.open(path) as src:
                raw = src.read(1)
            changed += int((raw == encoding[1]).sum())
            total += int(raw.size)
        return changed / total if total else float("nan")

    def _one_row_per_city(self) -> dict[str, dict]:
        return {row["city"]: row for row in self.rows}

    def _encoding(self, path: Path) -> tuple[int, int, str]:
        key = str(path)
        if key not in self._encodings:
            self._encodings[key] = read_label_encoding(path)
        return self._encodings[key]

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        window = Window(int(row["col_off"]), int(row["row_off"]),
                        int(row["width"]), int(row["height"]))
        arrays1 = [read_band_window(Path(row["image1_dir"]), band, window) for band in self.bands]
        arrays2 = [read_band_window(Path(row["image2_dir"]), band, window) for band in self.bands]
        image1 = torch.from_numpy(np.stack(arrays1, axis=0))
        image2 = torch.from_numpy(np.stack(arrays2, axis=0))
        image1 = (image1 - self.mean) / self.std
        image2 = (image2 - self.mean) / self.std
        mask = read_mask_window(Path(row["label_path"]), window,
                                self._encoding(Path(row["label_path"])))
        sample = {
            "image1": image1,
            "image2": image2,
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "name": "%s_r%s_c%s" % (row["city"], row["row_off"], row["col_off"]),
            "city": row["city"],
            "split": self.split,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def deterministic_validation_regions(official_train: list[str], excluded: set[str], count: int = 2) -> list[str]:
    """Return deterministic holdout regions by a documented stable hash rule."""
    candidates = [city for city in official_train if city not in excluded]
    ranked = sorted(candidates,
                    key=lambda city: hashlib.sha256(("oscd-val-v1:" + city).encode()).hexdigest())
    if count <= 0 or count >= len(candidates):
        raise ValueError("validation count must leave at least one training region")
    return ranked[:count]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create the original-OSCD patch index.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("outputs/patches/oscd_original_index.csv"))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    args = parser.parse_args()
    if args.patch_size % 8:
        parser.error("patch-size must be a multiple of 8 for the Siamese U-Net")
    rows = build_patch_index(args.data_root, args.split_config, args.patch_size, args.stride)
    write_patch_index(rows, args.output, args.patch_size, args.stride)
    print("wrote", args.output, "rows", len(rows))
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        print(split, "patches", len(split_rows), "regions", len({row["city"] for row in split_rows}))
    print("Test labels were not read while building the index.")


if __name__ == "__main__":
    main()
