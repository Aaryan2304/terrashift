"""Compare two normalisation strategies on real TRAIN imagery, without training.

Strategy A keeps the OSCD checkpoint's input contract: stored DN mean and std mapped into reflectance
by ``mean_R = (mean_DN - 1000) / 10000`` and ``std_R = std_DN / 10000``. Strategy B recomputes the
statistics from local TRAIN imagery, which is what the trainer does today.

Samples are deterministic 512 px blocks across the TRAIN partition, valid unlabelled pixels included.
Validation and test regions are never read: block origins are filtered by the partition raster before
any band is opened.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.local_patches import BANDS
from terrashift.data.manifest import load_manifest, pair_config
from terrashift.data.safeloader import SafeProduct

BLOCK = 512
MAX_BLOCKS = 60
RESERVOIR = 200_000
PERCENTILES = (1, 5, 50, 95, 99)
QUANTIFICATION = 10000.0
OFFSET = 1000.0


def deterministic_blocks(train_columns, height, stride):
    """Block origins whose columns lie entirely inside the train partition."""
    blocks = []
    for row0 in range(0, height - BLOCK + 1, stride):
        for col0 in range(0, train_columns[1] - BLOCK + 1, stride):
            if col0 >= train_columns[0] and col0 + BLOCK <= train_columns[1]:
                blocks.append((row0, col0))
    return blocks


def train_columns(partition, code=1):
    columns = np.flatnonzero((partition == code).any(axis=0))
    if columns.size == 0:
        raise SystemExit("partition has no train columns")
    return int(columns[0]), int(columns[-1]) + 1


def collect(product, band, blocks, valid_mask, review_mask=None):
    """Finite, valid, train-partition values from every block, plus a reservoir."""
    total = 0
    nodata = 0
    invalid = 0
    kept = []
    reservoir = []
    rng = np.random.default_rng(12345)
    for row0, col0 in blocks:
        window = ((row0, row0 + BLOCK), (col0, col0 + BLOCK))
        values = product.read_band(band, window=window)[0]
        total += values.size
        finite = np.isfinite(values)
        nodata += int((~finite).sum())
        block_valid = valid_mask[row0:row0 + BLOCK, col0:col0 + BLOCK]
        invalid += int((finite & ~block_valid).sum())
        selected = finite & block_valid
        if review_mask is not None:
            selected &= review_mask[row0:row0 + BLOCK, col0:col0 + BLOCK]
        values = values[selected].astype(np.float64)
        if not values.size:
            continue
        kept.append((values.sum(), (values ** 2).sum(), values.size))
        take = min(values.size, max(1, RESERVOIR // max(1, len(blocks))))
        if take >= values.size:
            reservoir.append(values)
        else:
            reservoir.append(rng.choice(values, size=take, replace=False))
    if not kept:
        raise ValueError("no valid pixels collected")
    sums = sum(item[0] for item in kept)
    squares = sum(item[1] for item in kept)
    count = sum(item[2] for item in kept)
    mean = sums / count
    variance = max(squares / count - mean ** 2, 0.0)
    return {
        "count": int(count), "mean": float(mean), "std": float(np.sqrt(variance)),
        "nodata_pixels": nodata, "invalid_pixels": invalid, "total_pixels": total,
        "sample": np.concatenate(reservoir),
    }


def summarise(sample, mean, std, label):
    z = (sample - mean) / std
    return {
        "strategy": label,
        "mean": float(mean), "std": float(std),
        "percentiles": {str(p): float(np.percentile(sample, p)) for p in PERCENTILES},
        "max": float(sample.max()), "min": float(sample.min()),
        "z_percentiles": {str(p): float(np.percentile(z, p)) for p in PERCENTILES},
        "z_mean": float(z.mean()), "z_std": float(z.std()),
        "z_max": float(z.max()), "z_min": float(z.min()),
        "pct_below_minus5": round(100 * float((z < -5).mean()), 3),
        "pct_above_plus5": round(100 * float((z > 5).mean()), 3),
        "pct_outside_pm5": round(100 * float((np.abs(z) > 5).mean()), 3),
        "non_finite": int((~np.isfinite(sample)).sum()),
        "non_finite_pct": round(100 * float((~np.isfinite(sample)).mean()), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-dir", default=os.path.join(ROOT, "outputs", "labels"))
    parser.add_argument("--partition-dir",
                        default=os.path.join(ROOT, "outputs",
                                             "partitions_v2_observation_valid"))
    parser.add_argument("--validity-dir", default=os.path.join(ROOT, "outputs", "baseline"))
    parser.add_argument("--manifest", default=os.path.join(ROOT, "configs",
                                                           "data_manifest.yaml"))
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    import torch
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    bands = list(state["bands"])
    ckpt_mean_dn = [float(v) for v in state["mean"]]
    ckpt_std_dn = [float(v) for v in state["std"]]
    strategy_a = {
        band: {"mean_R": (ckpt_mean_dn[i] - OFFSET) / QUANTIFICATION,
               "std_R": ckpt_std_dn[i] / QUANTIFICATION,
               "mean_DN": ckpt_mean_dn[i], "std_DN": ckpt_std_dn[i]}
        for i, band in enumerate(bands)}

    manifest = load_manifest(args.manifest)
    report = {
        "checkpoint_statistics_DN": {"mean": ckpt_mean_dn, "std": ckpt_std_dn},
        "strategy_A_affine_equivalent": strategy_a,
        "block_size": BLOCK, "max_blocks_per_pair": MAX_BLOCKS,
        "sampling": "512 px blocks on a fixed stride whose columns lie wholly inside "
                    "the TRAIN partition; both dates; invalid observations excluded",
        "pairs": {},
    }

    for pair in ("A", "B"):
        cfg = pair_config(manifest, pair)
        with rasterio.open(os.path.join(args.partition_dir,
                                        "pair_%s_partition_10m.tif" % pair)) as handle:
            partition = handle.read(1)
        validity_path = os.path.join(args.validity_dir, "observation_valid_%s.tif" % pair)
        if os.path.isfile(validity_path):
            with rasterio.open(validity_path) as handle:
                valid_mask = handle.read(1) > 0
            validity_source = validity_path
        else:
            valid_mask = np.ones(partition.shape, dtype=bool)
            validity_source = "partition only (no validity raster found)"
        labels_path = os.path.join(args.labels_dir, "pair_%s_v2" % pair.lower(),
                                   "manual_labels_%s.tif" % pair)
        review_mask = None
        if os.path.isfile(labels_path):
            with rasterio.open(labels_path) as handle:
                review_mask = handle.read(1) != 255

        columns = train_columns(partition)
        blocks = deterministic_blocks(columns, partition.shape[0], BLOCK * 2)
        if len(blocks) > MAX_BLOCKS:
            step = max(1, len(blocks) // MAX_BLOCKS)
            blocks = blocks[::step][:MAX_BLOCKS]
        entry = {"train_columns": list(columns), "blocks": len(blocks),
                 "validity_source": validity_source, "bands": {},
                 "reviewed_subset": {}}
        print("pair %s: %d blocks over train columns %s" % (pair, len(blocks), columns))

        for date_key in ("path_date1", "path_date2"):
            product = SafeProduct(cfg[date_key])
            for band in bands:
                values = collect(product, band, blocks, valid_mask)
                sample = values["sample"]
                local_mean_R = values["mean"]
                local_std_R = values["std"]
                a = strategy_a[band]
                entry["bands"].setdefault(band, {})[date_key] = {
                    "count": values["count"],
                    "mean_R": local_mean_R, "std_R": local_std_R,
                    "mean_DN": local_mean_R * QUANTIFICATION + OFFSET,
                    "std_DN": local_std_R * QUANTIFICATION,
                    "nodata_pct": round(100 * values["nodata_pixels"] /
                                        max(1, values["total_pixels"]), 3),
                    "invalid_pct": round(100 * values["invalid_pixels"] /
                                         max(1, values["total_pixels"]), 3),
                    "percentiles_R": {str(p): float(np.percentile(sample, p))
                                      for p in PERCENTILES},
                    "max_R": float(sample.max()), "min_R": float(sample.min()),
                    "strategy_A": summarise(sample, a["mean_R"], a["std_R"],
                                            "A: checkpoint statistics in reflectance"),
                    "strategy_B": summarise(sample, local_mean_R, local_std_R,
                                            "B: local TRAIN statistics"),
                }
                print("  %s %s: n=%d mean_R %.4f std_R %.4f | nodata %.2f%% invalid %.2f%%"
                      % (band, date_key[-1], values["count"], local_mean_R, local_std_R,
                         entry["bands"][band][date_key]["nodata_pct"],
                         entry["bands"][band][date_key]["invalid_pct"]))
                print("      A z mean %+.2f std %.2f | outside ±5σ %.2f%% || "
                      "B z mean %+.2f std %.2f | outside ±5σ %.2f%%"
                      % (entry["bands"][band][date_key]["strategy_A"]["z_mean"],
                         entry["bands"][band][date_key]["strategy_A"]["z_std"],
                         entry["bands"][band][date_key]["strategy_A"]["pct_outside_pm5"],
                         entry["bands"][band][date_key]["strategy_B"]["z_mean"],
                         entry["bands"][band][date_key]["strategy_B"]["z_std"],
                         entry["bands"][band][date_key]["strategy_B"]["pct_outside_pm5"]))

        # Reviewed-pixel statistics, reported separately and never pooled.
        if review_mask is not None:
            for band in bands:
                product = SafeProduct(cfg["path_date1"])
                values = collect(product, band, blocks, valid_mask, review_mask)
                entry["reviewed_subset"][band] = {
                    "count": values["count"], "mean_R": values["mean"],
                    "std_R": values["std"],
                    "mean_DN": values["mean"] * QUANTIFICATION + OFFSET}
        report["pairs"][pair] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
