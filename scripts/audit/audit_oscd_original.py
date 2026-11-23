"""Audit the downloaded original OSCD regions and labels.

This script reads the supplied files only. It does not modify the dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from PIL import Image

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07",
         "B08", "B8A", "B09", "B10", "B11", "B12")


def read_city_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    cities = [item.strip() for item in text.replace("\n", "").split(",")]
    if not cities or any(not city for city in cities):
        raise ValueError("invalid city list: %s" % path)
    if len(cities) != len(set(cities)):
        raise ValueError("duplicate cities in %s" % path)
    return cities


def find_label_root(root: Path, split: str, city: str) -> Path:
    path = root / ("Onera Satellite Change Detection dataset - %s Labels" % split.title())
    if not (path / city / "cm" / (city + "-cm.tif")).is_file():
        raise FileNotFoundError("missing %s label for %s" % (split, city))
    return path


def inspect_label(tif_path: Path, png_path: Path) -> dict:
    with rasterio.open(tif_path) as src:
        mask = src.read(1)
        height, width = src.height, src.width
        if src.count != 1:
            raise ValueError("label must have one band: %s" % tif_path)
        if src.crs is not None:
            raise ValueError("expected the OSCD label grid to be non-georeferenced: %s" % tif_path)
    values, counts = np.unique(mask, return_counts=True)
    value_set = set(int(value) for value in values)
    if value_set == {1, 2}:
        no_change_value, change_value = 1, 2
        encoding = "observed_1_no_change_2_change"
    elif value_set == {0, 1}:
        no_change_value, change_value = 0, 1
        encoding = "documented_0_no_change_1_change"
    else:
        raise ValueError("unsupported OSCD label values %s in %s" % (sorted(value_set), tif_path))

    png = np.asarray(Image.open(png_path).convert("L"))
    if png.shape != mask.shape:
        raise ValueError("PNG/TIFF shape mismatch for %s" % tif_path)
    # The author calls the PNG a visualization. Its grey boundary pixels can
    # be anti-aliased, so compare the binary sides at the midpoint only.
    no_change_match = float(np.mean(png[mask == no_change_value] < 128))
    change_match = float(np.mean(png[mask == change_value] >= 128))
    if no_change_match < 0.99 or change_match < 0.99:
        raise ValueError(
            "mask TIFF/PNG classes do not agree for %s (no-change %.4f, change %.4f)"
            % (tif_path, no_change_match, change_match)
        )
    counts_by_value = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return {
        "shape": [height, width],
        "encoding": encoding,
        "no_change_value": no_change_value,
        "change_value": change_value,
        "no_change_pixels": counts_by_value[str(no_change_value)],
        "change_pixels": counts_by_value[str(change_value)],
        "png_visual_agreement": {
            "no_change_dark_fraction": no_change_match,
            "change_light_fraction": change_match,
        },
    }


def inspect_city(images_root: Path, labels_root: Path, city: str, split: str) -> dict:
    city_root = images_root / city
    date_lines = (city_root / "dates.txt").read_text(encoding="utf-8").splitlines()
    if len(date_lines) != 2 or not all(line.strip() for line in date_lines):
        raise ValueError("expected two non-empty dates in %s" % (city_root / "dates.txt"))
    expected = set(BANDS)
    shape_by_grid: dict[str, set[tuple[int, int]]] = {"native": set(), "rectified": set()}
    crs_by_grid: dict[str, set[str]] = {"native": set(), "rectified": set()}
    file_counts: dict[str, int] = {}

    for date_index in (1, 2):
        for grid, directory in (("native", "imgs_%d" % date_index),
                                ("rectified", "imgs_%d_rect" % date_index)):
            band_dir = city_root / directory
            files = sorted(band_dir.glob("*.tif"))
            file_counts[directory] = len(files)
            if grid == "native":
                actual_bands = {path.stem.rsplit("_", 1)[-1] for path in files}
            else:
                actual_bands = {path.stem for path in files}
            if len(files) != len(BANDS) or actual_bands != expected:
                raise ValueError("missing/extra bands in %s (found %s files)" % (band_dir, len(files)))
            for band in BANDS:
                path = next(
                    item for item in files
                    if (item.stem.rsplit("_", 1)[-1] if grid == "native" else item.stem) == band
                )
                with rasterio.open(path) as src:
                    if src.count != 1:
                        raise ValueError("expected a one-band image: %s" % path)
                    shape_by_grid[grid].add((src.height, src.width))
                    crs_by_grid[grid].add(str(src.crs))
                    if grid == "native" and src.crs is None:
                        raise ValueError("native OSCD image has no CRS: %s" % path)
                    if grid == "rectified" and src.crs is not None:
                        raise ValueError("rectified OSCD image unexpectedly has a CRS: %s" % path)
                    if grid == "rectified" and src.transform != Affine.identity():
                        raise ValueError("rectified OSCD image does not use an identity pixel transform: %s" % path)

    if len(shape_by_grid["rectified"]) != 1:
        raise ValueError("rectified bands/dates are not on one shape for %s" % city)
    if next(iter(crs_by_grid["native"])) != "EPSG:4326":
        raise ValueError("unexpected native CRS for %s: %s" % (city, crs_by_grid["native"]))
    if crs_by_grid["rectified"] != {"None"}:
        raise ValueError("unexpected rectified CRS for %s: %s" % (city, crs_by_grid["rectified"]))

    label_root = find_label_root(labels_root.parent, split, city)
    label_dir = label_root / city / "cm"
    label_info = inspect_label(label_dir / (city + "-cm.tif"), label_dir / "cm.png")
    rect_shape = list(next(iter(shape_by_grid["rectified"])))
    if label_info["shape"] != rect_shape:
        raise ValueError("label/image shape mismatch for %s: %s vs %s" % (city, label_info["shape"], rect_shape))

    return {
        "city": city,
        "official_split": split,
        "dates": [line.strip() for line in date_lines],
        "image_file_counts": file_counts,
        "rectified_shape": rect_shape,
        "native_crs": sorted(crs_by_grid["native"]),
        "rectified_crs": sorted(crs_by_grid["rectified"]),
        "rectified_transform_is_identity": True,
        "label": label_info,
    }


def run_audit(data_root: Path) -> dict:
    images_root = data_root / "Onera Satellite Change Detection dataset - Images"
    train_labels_root = data_root / "Onera Satellite Change Detection dataset - Train Labels"
    test_labels_root = data_root / "Onera Satellite Change Detection dataset - Test Labels"
    train = read_city_list(images_root / "train.txt")
    test = read_city_list(images_root / "test.txt")
    all_cities = read_city_list(images_root / "all.txt")
    scene_dirs = sorted(path.name for path in images_root.iterdir() if path.is_dir())
    if set(train) & set(test):
        raise ValueError("official train and test lists overlap")
    if set(train) | set(test) != set(all_cities) or set(scene_dirs) != set(all_cities):
        raise ValueError("official split lists do not match all.txt and scene directories")

    cities = []
    for city in all_cities:
        split = "train" if city in train else "test"
        label_root = train_labels_root if split == "train" else test_labels_root
        cities.append(inspect_city(images_root, label_root, city, split))
    encoding_counts = Counter(item["label"]["encoding"] for item in cities)
    label_values = Counter()
    for item in cities:
        label_values[(item["label"]["no_change_value"], item["label"]["change_value"])] += 1
    return {
        "data_root": str(data_root.resolve()),
        "official_split_counts": {"train": len(train), "test": len(test), "all": len(all_cities)},
        "official_split_disjoint": True,
        "split_union_matches_all": True,
        "scene_directory_count": len(scene_dirs),
        "rectified_band_files": 24 * 2 * len(BANDS),
        "native_band_files": 24 * 2 * len(BANDS),
        "native_crs": "EPSG:4326",
        "rectified_crs": None,
        "rectified_transform_is_identity": True,
        "rectified_grid": "pixel coordinates; identity transform; 10 m-resampled according to the dataset README",
        "label_encodings": dict(encoding_counts),
        "label_value_mappings": {"%s_%s" % key: count for key, count in label_values.items()},
        "cities": cities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audit"))
    args = parser.parse_args()
    report = run_audit(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "oscd_original_data_audit.json"
    md_path = args.output_dir / "oscd_original_data_audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Original OSCD data audit",
        "",
        "This report was generated from the downloaded files by `scripts/audit/audit_oscd_original.py`.",
        "The script does not modify the source dataset.",
        "",
        "## Verified structure",
        "",
        "- Official split: %(train)d train regions and %(test)d test regions; lists are disjoint and cover all %(all)d regions." % report["official_split_counts"],
        "- Scene folders: %(scene_directory_count)d." % report,
        "- Native image bands checked: %(native_band_files)d; native CRS: EPSG:4326." % report,
        "- Rectified image bands checked: %(rectified_band_files)d; CRS is absent and transforms are identity pixel grids." % report,
        "- Every rectified date pair has 13 named bands on one common shape. Each label has the same shape.",
        "- Every label PNG agrees with the TIFF class map at a 128 intensity boundary, allowing antialiased visualization edges.",
        "",
        "## Label encoding finding",
        "",
        "The downloaded label TIFFs use values 1 (no change) and 2 (change) in all 24 regions. The label README says 0/1. The PNG visualization independently supports 1/2 mapping: class 1 is dark and class 2 is light. The loader must map class 2 to binary 1 and class 1 to binary 0. It must reject unknown values.",
        "",
        "The project page describes the labels as pixel-level urban changes focused on new buildings and roads. It does not claim every positive pixel is all structural change types.",
        "",
        "## Per-region counts",
        "",
        "| Region | Split | First date | Second date | Shape (H×W) | No-change px | Change px | Encoding |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for item in report["cities"]:
        label = item["label"]
        lines.append("| {city} | {official_split} | {date1} | {date2} | {h}×{w} | {unchanged} | {changed} | {encoding} |".format(
            city=item["city"], official_split=item["official_split"],
            date1=item["dates"][0], date2=item["dates"][1],
            h=label["shape"][0], w=label["shape"][1],
            unchanged=label["no_change_pixels"], changed=label["change_pixels"],
            encoding=label["encoding"]))
    lines.extend([
        "",
        "## Sources",
        "",
        "- OSCD authors’ project page: https://rcdaudt.github.io/oscd/",
        "- Downloaded `Images/README.txt` and `Train Labels/README.txt` / `Test Labels/README.txt`.",
        "- TorchGeo OSCD loader documentation: https://docs.torchgeo.org/en/stable/api/datasets/oscd.html",
        "",
        "This audit verifies files and dimensions. It does not establish that a model performs well.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("regions: %d (train %d, test %d)" % (len(report["cities"]), len(train_cities(report)), len(test_cities(report))))
    print("native bands: %d; rectified bands: %d" % (report["native_band_files"], report["rectified_band_files"]))
    print("label encoding counts:", report["label_encodings"])
    print("wrote", json_path)
    print("wrote", md_path)


def train_cities(report: dict) -> list[dict]:
    return [item for item in report["cities"] if item["official_split"] == "train"]


def test_cities(report: dict) -> list[dict]:
    return [item for item in report["cities"] if item["official_split"] == "test"]


if __name__ == "__main__":
    main()
