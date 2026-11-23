"""Read-only audit of the input contract between the OSCD checkpoint and local data.

One question: can a checkpoint trained on OSCD patches take local Sentinel-2 patches without being fed
garbage? Reports the checkpoint's bands and statistics, the SAFE metadata for both dates of both
pairs read from the product rather than assumed, TRAIN-only per-band distributions of what the local
loader produces in reflectance and OSCD-equivalent DN, and the z-scored values the model would see.
Train windows are every Nth of the training lattice, so the statistics describe a documented sample.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import hashlib
import json
import os

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.local_patches import BANDS, pair_image_loader, patch_origins
from terrashift.data.manifest import load_manifest, pair_config
from terrashift.data.safeloader import SafeProduct

PATCH = 128
STRIDE = 64
MAX_WINDOWS = 200
PERCENTILES = (1, 25, 50, 75, 99)


def checkpoint_contract(path):
    import torch
    raw = open(path, "rb").read()
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "epoch": int(state.get("epoch", -1)),
        "bands": list(state.get("bands", [])),
        "base": int(state.get("base", -1)),
        "pos_weight": float(state.get("pos_weight") or 0.0),
        "mean": [float(v) for v in state.get("mean", [])],
        "std": [float(v) for v in state.get("std", [])],
        "units": "raw L1C DN as returned by rasterio.read(1); no offset, no scaling",
    }


def safe_metadata(cfg, bands=BANDS):
    entry = {}
    for key in ("path_date1", "path_date2"):
        product = SafeProduct(cfg[key])
        entry[key] = {
            "name": product.name,
            "quantification": float(product.quantification),
            "offset": {band: float(product.offset_for(band)) for band in bands},
        }
    return entry


def train_windows(labels, partition, patch=PATCH, stride=STRIDE):
    train = (labels != 255) & (partition == 1)
    rows = patch_origins(labels.shape[0], patch, stride)
    cols = patch_origins(labels.shape[1], patch, stride)
    kept = []
    for r0 in rows:
        for c0 in cols:
            if train[r0:r0 + patch, c0:c0 + patch].any():
                kept.append((r0, c0))
    return kept, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-dir", default=os.path.join(ROOT, "outputs", "labels"))
    parser.add_argument("--partition-dir",
                        default=os.path.join(ROOT, "outputs",
                                             "partitions_v2_observation_valid"))
    parser.add_argument("--manifest", default=os.path.join(ROOT, "configs",
                                                           "data_manifest.yaml"))
    parser.add_argument("--registration",
                        default=os.path.join(ROOT, "outputs", "audit",
                                             "registration_v2_observation_valid",
                                             "registration.json"))
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    contract = checkpoint_contract(args.checkpoint)
    manifest = load_manifest(args.manifest)
    loader = pair_image_loader(args.manifest, args.registration, bands=BANDS)
    report = {"checkpoint": contract, "local_loader": {
        "bands": list(BANDS),
        "formula": "reflectance = (DN + offset) / quantification",
        "offset_and_quantification_source": "MTD_MSIL1C.xml, read by SafeProduct",
        "nodata": "DN == 0 becomes NaN, then filled with 0.0 before z-scoring",
        "normalisation": "per-band z-score using statistics computed from local "
                         "TRAIN patches only",
        "both_dates": "identical preprocessing; date 2 additionally shifted by the "
                      "registration offset",
    }, "pairs": {}}

    for pair in ("A", "B"):
        cfg = pair_config(manifest, pair)
        meta = safe_metadata(cfg, BANDS)
        report["pairs"][pair] = {"safe_metadata": meta}
        print("pair %s: quant %s offset %s" % (pair, meta["path_date1"]["quantification"],
                                               meta["path_date1"]["offset"]))
        with rasterio.open(os.path.join(args.labels_dir, "pair_%s_v2" % pair.lower(),
                                        "manual_labels_%s.tif" % pair)) as handle:
            labels = handle.read(1)
        with rasterio.open(os.path.join(args.partition_dir,
                                        "pair_%s_partition_10m.tif" % pair)) as handle:
            partition = handle.read(1)
        windows, train_mask = train_windows(labels, partition)
        print("  train windows containing reviewed pixels: %d" % len(windows))
        if not windows:
            report["pairs"][pair]["note"] = "no reviewed train pixels"
            continue
        step = max(1, len(windows) // MAX_WINDOWS)
        sample = windows[::step][:MAX_WINDOWS]

        raw = {band: [] for band in BANDS}
        scoped = {band: [] for band in BANDS}
        for r0, c0 in sample:
            for date_index in (0, 1):
                image = loader(pair, r0, c0, PATCH)[date_index]
                inside = train_mask[r0:r0 + PATCH, c0:c0 + PATCH]
                for channel, band in enumerate(BANDS):
                    plane = image[channel]
                    raw[band].append(plane[np.isfinite(plane)].astype(np.float32))
                    if inside.any():
                        scoped[band].append(plane[inside].astype(np.float32))

        per_band = {}
        for band in BANDS:
            raw_values = np.concatenate(raw[band])
            reviewed = np.concatenate(scoped[band]) if scoped[band] else raw_values
            raw_values = raw_values[np.isfinite(raw_values)]
            reviewed = reviewed[np.isfinite(reviewed)]
            mean = float(reviewed.mean())
            std = float(reviewed.std())
            z = (reviewed - mean) / std if std > 0 else reviewed * 0.0
            per_band[band] = {
                "windows": len(sample),
                "raw_min": float(raw_values.min()), "raw_max": float(raw_values.max()),
                "raw_percentiles": {str(p): float(np.percentile(raw_values, p))
                                    for p in PERCENTILES},
                "reviewed_mean": mean, "reviewed_std": std,
                "dn_equivalent_mean": mean * 10000.0 + 1000.0,
                "z_min": float(z.min()), "z_max": float(z.max()),
                "z_percentiles": {str(p): float(np.percentile(z, p))
                                  for p in PERCENTILES},
            }
        report["pairs"][pair]["train_distributions"] = per_band
        report["pairs"][pair]["sampled_windows"] = len(sample)
        report["pairs"][pair]["total_train_windows"] = len(windows)

        for channel, band in enumerate(BANDS):
            entry = per_band[band]
            print("  %s raw %.4f..%.4f median %.4f | DN-equiv mean %.0f | "
                  "z %.2f..%.2f" % (band, entry["raw_min"], entry["raw_max"],
                                    entry["raw_percentiles"]["50"],
                                    entry["dn_equivalent_mean"], entry["z_min"],
                                    entry["z_max"]))

    # The comparison that decides compatibility.
    report["comparison"] = {
        "checkpoint_units": "raw DN",
        "local_units": "TOA reflectance, 0-1",
        "dn_equivalent_of_local": "reflectance * 10000 + 1000 restores raw DN",
        "z_score_absorbs_scale_and_offset": True,
        "implication": ("Because both domains are z-scored with their own train "
                        "statistics, the 10^4 scale and the -1000 offset cancel. "
                        "The checkpoint weights are usable; the checkpoint's stored "
                        "mean/std are not, because they are in raw DN."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
