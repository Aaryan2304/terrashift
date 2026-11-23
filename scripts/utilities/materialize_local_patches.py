"""Materialize on-demand local manual patches as compressed NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
from terrashift.data.local_patches import LocalManualPatchDataset


def materialize(index_path: Path, manifest_path: Path, registration_path: Path,
                output_dir: Path, include_test: bool = False) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "refusing to overwrite a non-empty materialization directory: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_names = ("train", "validation", "test") if include_test else ("train", "validation")
    summaries = {}
    for split in split_names:
        dataset = LocalManualPatchDataset(
            index_path, split, manifest_path, registration_path,
            mean=None, std=None, transform=None)
        rows = []
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for index in range(len(dataset)):
            sample = dataset[index]
            target = split_dir / (sample["name"] + ".npz")
            np.savez_compressed(
                target,
                image1=sample["image1"].numpy().astype(np.float32),
                image2=sample["image2"].numpy().astype(np.float32),
                mask=sample["mask"].numpy().astype(np.uint8),
                valid_mask=sample["valid_mask"].numpy().astype(np.uint8),
            )
            rows.append({
                "file": str(target.resolve()), "name": sample["name"],
                "pair": sample["pair"], "split": split,
                "valid_pixels": int(sample["valid_mask"].sum()),
                "change_pixels": int(sample["mask"].sum()),
                "reflectance_scale": "TOA_reflectance",
                "normalization_applied": False,
            })
        fields = list(rows[0]) if rows else ["file", "name", "pair", "split",
                                             "valid_pixels", "change_pixels",
                                             "reflectance_scale", "normalization_applied"]
        with (split_dir / "index.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summaries[split] = len(rows)
    metadata = {
        "patch_index": str(Path(index_path).resolve()),
        "materialized_split_counts": summaries,
        "includes_test": bool(include_test),
        "feature_scale": "TOA reflectance from SafeProduct; date 2 uses saved global registration shift",
        "labels": "manual only; mask 255 is excluded by valid_mask",
        "normalization_applied": False,
    }
    (output_dir / "materialization.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--include-test", action="store_true",
                        help="materialize test patches only after final model selection")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg["data"]
    output_dir = args.output_dir or Path("outputs/patches/local_manual")
    result = materialize(
        Path(data_cfg["local_patch_index"]), Path(data_cfg["manifest"]),
        Path(data_cfg["registration_summary"]), output_dir, args.include_test)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
