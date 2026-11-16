"""Georeferenced change prediction for one local Sentinel-2 pair.

This is the SAFE-level counterpart of ``scripts/predict_folder.py``, which only
accepts paired ``.npy`` arrays and produces no map coordinates.

The preprocessing contract is the one the local trainer uses, applied to the
whole tile instead of to 256 px patches:

1. four native 10 m bands, B02, B03, B04, B08, on the date 1 grid;
2. date 2 aligned to date 1 with the stored global translation;
3. per-band z-score with the mean and standard deviation stored in the
   checkpoint, which were computed from the training labels only;
4. no histogram matching, no water masking, no resampling. Training never
   applied them, so inference must not either.

The tile is processed in row strips with a halo, so the sliding-window
accumulators never hold a full 10,980 x 10,980 float32 array. Each strip is
predicted, the halo is cropped, and the result is written into a window of the
output rasters. CRS, affine transform, width and height are copied from the
date 1 band grid, so the output is a map product, not an array.

Outputs, into ``--output-dir``:
    prediction_prob_<PAIR>.tif    float32 change probability, nodata -9999
    prediction_mask_<PAIR>.tif    uint8, 1 = positive, 0 = negative, 255 = invalid
    prediction_valid_<PAIR>.tif   uint8, 1 = valid observation, 0 = invalid
    prediction_config.json        resolved configuration and counts

Nothing is overwritten: the script refuses to run when a target exists.

This script is a pipeline, not a result. Until a local model is trained on
reviewed labels, any checkpoint produced by a smoke run gives meaningless
predictions, and the printed statistics must not be reported as detection
performance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.grids import grid_from_band
from data.manifest import load_manifest, pair_paths
from data.register import align_to_reference, coverage_mask
from data.safeloader import SafeProduct
from data.validity import window_validity
from models.siamese_unet import SiameseUNet
from utils.inference import sliding_window_predict

BANDS = ("B02", "B03", "B04", "B08")
HALO = 256


def load_checkpoint(path, bands):
    """Read the model, its normalisation statistics, and its training metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint has no model_state_dict: %s" % path)
    stored_bands = tuple(checkpoint.get("bands", ()))
    if stored_bands != tuple(bands):
        raise ValueError("checkpoint bands %s do not match %s" % (stored_bands, tuple(bands)))
    mean = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(checkpoint["std"], dtype=np.float32).reshape(-1, 1, 1)
    if mean.shape[0] != len(bands):
        raise ValueError("checkpoint mean has %d values for %d bands"
                         % (mean.shape[0], len(bands)))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("checkpoint normalisation statistics are not usable")
    return checkpoint, mean, std


def read_strip(product, rows, cols, shift=None):
    """Read the four bands over one window, optionally shifted."""
    stack = []
    for band in BANDS:
        values, _, _ = product.read_band(band, window=(rows, cols))
        if shift is not None:
            values = align_to_reference(values, shift[0], shift[1])
        stack.append(values)
    return np.stack(stack).astype(np.float32)


def normalise(stack, mean, std):
    """Apply the training z-score and replace non-finite values with 0.

    Non-finite pixels are masked out by the validity raster, so the value the
    model sees there does not matter; zero is the mean of the standardised
    data, which keeps the convolution state neutral.
    """
    out = (stack - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["A", "B"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for this run's rasters")
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--block-rows", type=int, default=512,
                        help="core rows per strip; a 256 row halo is added")
    parser.add_argument("--window", default=None,
                        help="optional 'r0,r1,c0,c1' window for a smoke run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--status", default="transfer_smoke_not_a_result",
                        help="status tag written to the rasters and the config; a "
                             "checkpoint not trained on reviewed local labels is a "
                             "transfer smoke test, not a result")
    args = parser.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg["data"]

    manifest = load_manifest(Path(data_cfg["manifest"]))
    registration_path = Path(data_cfg["registration_summary"])
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    da, db = pair_paths(manifest, args.pair)
    p1, p2 = SafeProduct(da), SafeProduct(db)

    grid = grid_from_band(p1, "B08")
    if grid.crs is None:
        raise SystemExit("date 1 band has no CRS; refusing to write an unreferenced raster")

    targets = [args.output_dir / ("prediction_prob_%s.tif" % args.pair),
               args.output_dir / ("prediction_mask_%s.tif" % args.pair),
               args.output_dir / ("prediction_valid_%s.tif" % args.pair)]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise SystemExit("refusing to overwrite existing predictions: %s" % ", ".join(existing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    gpu_peak = {"value": 0.0, "stop": threading.Event()}

    def _sample_gpu():
        while not gpu_peak["stop"].is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                gpu_peak["value"] = max(gpu_peak["value"],
                                        float(out.stdout.strip().splitlines()[0]))
            except (OSError, subprocess.SubprocessError, IndexError, ValueError):
                return
            gpu_peak["stop"].wait(0.25)

    sampler = threading.Thread(target=_sample_gpu, daemon=True)
    if args.device == "cuda":
        sampler.start()

    checkpoint, mean, std = load_checkpoint(args.checkpoint, BANDS)
    base = int(checkpoint["base"])
    patch_size = int(checkpoint.get("run_config", {}).get("patch_size", 256))
    if patch_size % 8:
        raise SystemExit("patch size must be a multiple of 8 for this U-Net")

    shift = registration[args.pair]["initial"].get("applied_shift_px")
    shift = (0.0, 0.0) if shift is None else (float(shift[0]), float(shift[1]))

    if args.window:
        r0, r1, c0, c1 = (int(v) for v in args.window.split(","))
        if not (0 <= r0 < r1 <= grid.height and 0 <= c0 < c1 <= grid.width):
            raise SystemExit("window is outside the tile")
    else:
        r0, r1, c0, c1 = 0, grid.height, 0, grid.width

    model = SiameseUNet(in_channels=len(BANDS), base=base).to(torch.device(args.device))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    profile_prob = dict(driver="GTiff", height=r1 - r0, width=c1 - c0, count=1,
                        dtype="float32", crs=grid.crs,
                        transform=rasterio.windows.transform(
                            Window(c0, r0, c1 - c0, r1 - r0), grid.transform),
                        nodata=-9999.0, tiled=True, compress="deflate")
    profile_u8 = dict(profile_prob, dtype="uint8", nodata=255, compress="deflate")

    counts = {"windows": 0, "valid_pixels": 0, "predicted_positive_pixels": 0,
              "pixels": 0}

    with rasterio.open(targets[0], "w", **profile_prob) as prob_ds, \
            rasterio.open(targets[1], "w", **profile_u8) as mask_ds, \
            rasterio.open(targets[2], "w", **profile_u8) as valid_ds:

        block = max(patch_size, int(args.block_rows))
        for core_start in range(r0, r1, block):
            core_stop = min(core_start + block, r1)
            read_start = max(r0, core_start - HALO)
            read_stop = min(r1, core_stop + HALO)
            rows = (read_start, read_stop)
            cols = (c0, c1)

            first = read_strip(p1, rows, cols)
            second = read_strip(p2, rows, cols, shift=shift)
            x1 = torch.from_numpy(normalise(first, mean, std))
            x2 = torch.from_numpy(normalise(second, mean, std))

            probabilities, _ = sliding_window_predict(
                model, x1, x2, patch_size=patch_size, stride=args.stride,
                device=torch.device(args.device), threshold=None)

            valid, _ = window_validity(p1, (rows, cols), band=BANDS[0])
            valid2, _ = window_validity(p2, (rows, cols), band=BANDS[0])
            covered = coverage_mask(probabilities.shape, -shift[0], -shift[1])
            paired = valid & valid2 & covered

            top = core_start - read_start
            bottom = top + (core_stop - core_start)
            core_prob = probabilities[top:bottom]
            core_valid = paired[top:bottom]
            core_mask = np.where(core_valid, (core_prob >= args.threshold).astype(np.uint8),
                                 np.uint8(255))
            core_prob_out = np.where(core_valid, core_prob, -9999.0).astype(np.float32)

            window = Window(c0, core_start, c1 - c0, core_stop - core_start)
            prob_ds.write(core_prob_out, 1, window=window)
            mask_ds.write(core_mask, 1, window=window)
            valid_ds.write(core_valid.astype(np.uint8), 1, window=window)

            counts["windows"] += 1
            counts["pixels"] += int(core_prob.size)
            counts["valid_pixels"] += int(core_valid.sum())
            counts["predicted_positive_pixels"] += int((core_mask == 1).sum())
            print("  rows %d-%d: valid %d of %d, positive %d"
                  % (core_start, core_stop, int(core_valid.sum()), core_prob.size,
                     int((core_mask == 1).sum())))

    for path, tag in ((targets[0], "prediction_probability"),
                      (targets[1], "prediction_mask"),
                      (targets[2], "prediction_validity")):
        with rasterio.open(path, "r+") as ds:
            ds.update_tags(output_type=tag,
                           source_pair=args.pair,
                           threshold=str(args.threshold),
                           threshold_status="not_calibrated",
                           bands=",".join(BANDS),
                           patch_size=str(patch_size), stride=str(args.stride),
                           applied_shift_px=json.dumps(list(shift)),
                           normalisation="per-band z-score from checkpoint",
                           model="siamese_unet base=%d" % base,
                           checkpoint=Path(args.checkpoint).name,
                           status=args.status)

    gpu_peak["stop"].set()
    elapsed = time.perf_counter() - started_at
    import resource as _resource
    peak_rss_mib = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024.0

    resolved = {
        "pair": args.pair,
        "grid": {"crs": str(grid.crs), "width": c1 - c0, "height": r1 - r0,
                 "window": [r0, r1, c0, c1]},
        "bands": list(BANDS),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_initialization_mode": checkpoint.get("run_config", {}).get(
            "initialization_mode", "unknown"),
        "patch_size": patch_size, "stride": args.stride,
        "threshold": args.threshold,
        "applied_shift_px": list(shift),
        "device": args.device,
        "counts": counts,
        "valid_fraction": counts["valid_pixels"] / max(counts["pixels"], 1),
        "positive_fraction_of_valid": (counts["predicted_positive_pixels"]
                                       / max(counts["valid_pixels"], 1)),
        "status": args.status,
        "note": ("These predictions are not a detection result. Without a model "
                 "trained on reviewed local labels, the numbers are a smoke test "
                 "of the inference path only."),
        "elapsed_seconds": round(elapsed, 2),
        "pixels_per_second": round(counts["pixels"] / max(elapsed, 1e-9), 1),
        "peak_process_rss_mib": round(peak_rss_mib, 1),
        "peak_gpu_memory_mib": (round(gpu_peak["value"], 1)
                                if args.device == "cuda" else None),
        "outputs": [str(p) for p in targets],
    }
    (args.output_dir / "prediction_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resolved, indent=2))


if __name__ == "__main__":
    main()
