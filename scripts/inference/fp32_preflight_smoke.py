"""Data-loader smoke test for the local calibration adapter, on real SAFE data.

Reads two real batches from the training split and checks the contract the quantiser
depends on. It does not calibrate, does not compute quantisation ranges and does not
touch validation labels beyond what the adapter already uses.

    python scripts/inference/fp32_preflight_smoke.py --report outputs/evaluation/fp32_onnx_preflight_v1/loader_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import quantize
from terrashift.data.local_patches import LocalManualPatchDataset, read_pair_window
from terrashift.preprocessing.normalisation import normalise_reflectance, pair_statistics

BATCHES_TO_READ = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint",
                        default="outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt")
    parser.add_argument("--normalisation", default="outputs/dataset/v2/normalisation.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-batches", type=int, default=20)
    parser.add_argument("--calibration-seed", type=int, default=43)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_cfg = cfg["data"]
    stub = argparse.Namespace(
        manifest=data_cfg["manifest"],
        registration=data_cfg["registration_summary"],
        batch_size=args.batch_size, calibration_batches=args.calibration_batches,
        calibration_seed=args.calibration_seed)

    _, bands, checkpoint = quantize.load_model(args.checkpoint, "cpu")
    normalisation = quantize.require_statistics(args.normalisation, bands, ("A", "B"))
    loader, plan = quantize.make_local_calibration_loader(cfg, checkpoint, stub, normalisation)

    report = {"bands": list(bands), "plan": plan, "batches": [], "checks": {}}
    seen_pairs = set()
    saw_full_batch = True

    for index, batch in enumerate(loader):
        if index >= BATCHES_TO_READ:
            break
        entry = {
            "batch_index": index,
            "shape_image1": list(batch["image1"].shape),
            "shape_image2": list(batch["image2"].shape),
            "dtype_image1": str(batch["image1"].dtype),
            "finite_image1": bool(np.isfinite(batch["image1"].numpy()).all()),
            "finite_image2": bool(np.isfinite(batch["image2"].numpy()).all()),
            "pairs": sorted(set(batch["pair"])),
            "rows": [int(v) for v in batch["row_off"]],
            "valid_mask_shape": list(batch["valid_mask"].shape),
            "valid_fraction": [round(float(v), 4)
                               for v in batch["valid_mask"].numpy().reshape(len(batch["pair"]), -1).mean(axis=1)],
            "mean": round(float(batch["image1"].mean()), 4),
            "std": round(float(batch["image1"].std()), 4),
        }
        seen_pairs.update(batch["pair"])
        if batch["image1"].shape != (len(batch["pair"]), 4, 256, 256):
            saw_full_batch = False
        report["batches"].append(entry)

    # ---- normalisation applied exactly once ------------------------------
    dataset = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else None
    single_application = None
    if isinstance(dataset, LocalManualPatchDataset):
        first = loader.dataset[0]
        pair = first["pair"]
        row_off, col_off = int(first["row_off"]), int(first["col_off"])
        date1, date2 = dataset._pair_products(pair)
        dx, dy = dataset.shifts.get(pair, (0.0, 0.0))
        raw1, _ = read_pair_window(date1, date2, row_off, col_off, 256, dataset.bands, (dx, dy))
        mean, std = pair_statistics(normalisation, pair)
        once = normalise_reflectance(raw1, mean, std).astype(np.float32)
        twice = normalise_reflectance(once, mean, std).astype(np.float32)
        loaded = first["image1"].numpy()
        single_application = {
            "pair": pair,
            "max_abs_vs_single_application": float(np.abs(loaded - once).max()),
            "max_abs_vs_double_application": float(np.abs(loaded - twice).max()),
            "applied_once": bool(np.abs(loaded - once).max() < 1e-5),
            "not_applied_twice": bool(np.abs(loaded - twice).max() > 1e-2),
        }

    # ---- validity mask behaviour ----------------------------------------
    first = loader.dataset[0]
    valid = first["valid_mask"].numpy()
    mask = first["mask"].numpy()
    validity = {
        "invalid_pixels": int((~valid).sum()),
        "mask_classes": sorted(int(v) for v in np.unique(mask)),
        "mask_zero_where_invalid": bool(np.all(mask[~valid] == 0)),
        "mask_dtype": str(first["mask"].dtype),
    }

    # ---- train split only -----------------------------------------------
    split_check = {
        "dataset_split": dataset.split if isinstance(dataset, LocalManualPatchDataset) else None,
        "rows_split_values": sorted({row["split"] for row in dataset.rows})
        if isinstance(dataset, LocalManualPatchDataset) else None,
    }

    report["checks"] = {
        "both_pairs_in_two_batches": sorted(seen_pairs) == ["A", "B"],
        "pairs_seen": sorted(seen_pairs),
        "full_batches_of_8": saw_full_batch,
        "single_normalisation": single_application,
        "validity": validity,
        "split": split_check,
    }
    report["passed"] = bool(
        report["checks"]["both_pairs_in_two_batches"]
        and report["checks"]["full_batches_of_8"]
        and single_application and single_application["applied_once"]
        and single_application["not_applied_twice"]
        and validity["mask_zero_where_invalid"]
        and split_check["rows_split_values"] == ["train"])

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    if not report["passed"]:
        raise SystemExit("calibration loader smoke test FAILED")


if __name__ == "__main__":
    main()
