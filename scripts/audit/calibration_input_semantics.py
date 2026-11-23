"""Establish what the calibration inputs actually contain.

Three things are routinely conflated: the manually reviewed label mask, the
observation-validity mask, and the image tensors the model sees. This script separates them
on real patches and fails if unreviewed-but-valid imagery is being zeroed.

    python scripts/audit/calibration_input_semantics.py \
        --report outputs/evaluation/int8_quantisation_execution_v1/calibration_input_semantics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import quantize
from terrashift.data.local_patches import LocalManualPatchDataset, read_pair_window
from terrashift.preprocessing.normalisation import normalise_reflectance, pair_statistics


def inspect_sample(dataset, index, normalisation):
    sample = dataset[index]
    pair = sample["pair"]
    row_off, col_off = int(sample["row_off"]), int(sample["col_off"])
    date1, date2 = dataset._pair_products(pair)
    dx, dy = dataset.shifts.get(pair, (0.0, 0.0))
    raw1, raw2 = read_pair_window(date1, date2, row_off, col_off, 256, dataset.bands, (dx, dy))

    with rasterio.open(sample["label_path"]) as handle:
        label = handle.read(1, window=Window(col_off, row_off, 256, 256))

    reviewed = label != 255
    finite1 = np.isfinite(raw1).all(axis=0)
    finite2 = np.isfinite(raw2).all(axis=0)
    genuinely_invalid = ~(finite1 & finite2)

    mean, std = pair_statistics(normalisation, pair)
    expected1 = normalise_reflectance(raw1, mean, std).astype(np.float32)

    tensor1 = sample["image1"].numpy()
    tensor2 = sample["image2"].numpy()

    unreviewed_valid = (~reviewed) & ~genuinely_invalid
    # neutral means every band is exactly 0 for that pixel
    zero_in_tensor = (tensor1 == 0.0).all(axis=0)

    return {
        "index": int(index), "pair": pair, "row_off": row_off, "col_off": col_off,
        "pixels": int(reviewed.size),
        "reviewed_pixels": int(reviewed.sum()),
        "reviewed_fraction": round(float(reviewed.mean()), 6),
        "genuinely_invalid_pixels": int(genuinely_invalid.sum()),
        "genuinely_invalid_fraction": round(float(genuinely_invalid.mean()), 6),
        "unreviewed_but_valid_pixels": int(unreviewed_valid.sum()),
        "max_abs_tensor_vs_single_normalisation": float(np.abs(tensor1 - expected1).max()),
        "tensor_all_finite": bool(np.isfinite(tensor1).all() and np.isfinite(tensor2).all()),
        "invalid_pixels_are_neutral": bool(zero_in_tensor[genuinely_invalid].all())
        if genuinely_invalid.any() else None,
        "unreviewed_valid_kept_as_data": float((~zero_in_tensor[unreviewed_valid]).mean())
        if unreviewed_valid.any() else None,
        "unreviewed_valid_std": round(float(tensor1[:, unreviewed_valid].std()), 4)
        if unreviewed_valid.any() else None,
        "tensor_zero_fraction": round(float(zero_in_tensor.mean()), 6),
        "label_classes_in_patch": sorted(int(v) for v in np.unique(label)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint",
                        default="outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt")
    parser.add_argument("--normalisation", default="outputs/dataset/v2/normalisation.json")
    parser.add_argument("--samples-per-pair", type=int, default=1)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_cfg = cfg["data"]
    _, bands, checkpoint = quantize.load_model(args.checkpoint, "cpu")
    normalisation = quantize.require_statistics(args.normalisation, bands, ("A", "B"))

    dataset = LocalManualPatchDataset(
        index_path=Path(data_cfg["local_patch_index"]), split="train",
        manifest_path=Path(data_cfg["manifest"]),
        registration_path=Path(data_cfg["registration_summary"]),
        normalisation=normalisation, bands=tuple(bands))

    by_pair = {}
    for index, row in enumerate(dataset.rows):
        by_pair.setdefault(row["pair"], []).append(index)

    records = []
    for pair in ("A", "B"):
        for index in by_pair.get(pair, [])[:args.samples_per_pair]:
            records.append(inspect_sample(dataset, index, normalisation))

    checks = []
    for record in records:
        checks.append({
            "pair": record["pair"],
            "single_normalisation_exact":
                record["max_abs_tensor_vs_single_normalisation"] < 1e-5,
            "unreviewed_valid_pixels_retained":
                (record["unreviewed_valid_kept_as_data"] or 0.0) > 0.999,
            "invalid_observations_neutral":
                record["invalid_pixels_are_neutral"] is not False,
            "finite": record["tensor_all_finite"],
        })
    passed = all(all(value for key, value in check.items() if key != "pair")
                 for check in checks)

    payload = {
        "question": "Do the calibration tensors keep real imagery outside the reviewed mask?",
        "definition": {
            "reviewed_label_mask": "manual label raster != 255 (annotation coverage)",
            "observation_validity_mask": "both dates finite at every band (true NoData)",
            "input_tensor": "4-band patch, pair-specific z-score over the whole patch",
            "neutral_value": "0.0 after standardising, applied only where an observation "
                             "is genuinely invalid",
        },
        "samples": records, "checks": checks, "passed": passed,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload, indent=1))
    if not passed:
        raise SystemExit("calibration input semantics FAILED")


if __name__ == "__main__":
    main()
