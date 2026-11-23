"""Georeferenced change prediction for one local Sentinel-2 pair.

SAFE-level counterpart of ``predict_folder.py``, which takes paired ``.npy`` arrays and produces no
coordinates. Preprocessing matches the local trainer: four 10 m bands on the date 1 grid, date 2
shifted by the stored translation, per-band z-score from the checkpoint statistics, and no histogram
matching, water masking or resampling, because training never applied them.

The tile is processed in row strips with a halo, so no full-tile float32 array is held, and CRS and
transform are copied from the date 1 grid. A checkpoint from a smoke run gives meaningless numbers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.grids import grid_from_band
from terrashift.data.manifest import load_manifest, pair_paths
from terrashift.registration import align_to_reference, coverage_mask
from terrashift.data.safeloader import SafeProduct
from terrashift.data.validity import window_validity
from terrashift.models.siamese_unet import SiameseUNet
from terrashift.inference.sliding import sliding_window_predict

BANDS = ("B02", "B03", "B04", "B08")
HALO = 256


class HalfInputModel(torch.nn.Module):
    """Cast inputs to half before the forward pass.

    This is the execution method that was evaluated: the model weights are converted with
    ``model.half()`` and the inputs are cast to match. It is deliberately not
    ``torch.autocast``: autocast leaves some operations in float32 and therefore computes a
    different function from the one measured. sliding_window_predict converts the returned
    logits to float32 before applying sigmoid, so the accumulator stays float32.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image1, image2):
        return self.model(image1.half(), image2.half())


def resolve_execution(device, precision, batch_size=None, cuda_available=None):
    """Validate an execution request and return the batch size to use.

    A CUDA request with no working GPU raises rather than falling back, because a silent
    fallback makes a GPU run indistinguishable from a CPU run in the output artefacts.
    fp16 is only defined for the CUDA path: on CPU it would be a different computation
    from the one that was evaluated.
    """
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    if device == "cuda" and not available:
        raise SystemExit("--device cuda was requested but torch.cuda.is_available() is "
                         "False; refusing to fall back to CPU silently")
    if precision == "fp16" and device != "cuda":
        raise SystemExit("--precision fp16 requires --device cuda; the CPU path is fp32 only")
    resolved = batch_size if batch_size is not None else (8 if device == "cuda" else 1)
    if resolved < 1:
        raise SystemExit("--inference-batch-size must be at least 1")
    return resolved


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



def _run_note(status: str) -> str:
    """Factual note for the run config.

    The wording depends on ``status``: a run tagged as a smoke test used a checkpoint
    that was not trained on reviewed local labels, so its numbers say nothing about
    detection quality. A run with the selected local checkpoint does not carry that
    caveat, but it still produces candidate regions rather than detections.
    """
    note = ("These predictions are change scores: candidate regions for human review. "
            "They are not confirmed detections, and this run does not establish "
            "scene-wide precision.")
    if "smoke" in status:
        note += (" The checkpoint was not trained on reviewed local labels, so this is a "
                 "transfer smoke test of the inference path, not a detection result.")
    return note

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["A", "B"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for this run's rasters")
    parser.add_argument("--normalisation", type=Path, default=None,
                        help="per-pair statistics JSON; overrides the checkpoint's "
                             "single set, which is required when training used "
                             "pair-specific statistics")
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                        help="fp16 requires --device cuda and uses model.half() with half "
                             "inputs, the method that was evaluated")
    parser.add_argument("--inference-batch-size", type=int, default=None,
                        help="windows per forward pass; defaults to 1 on cpu and 8 on cuda")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="record per-stage timings in the run config; off by default")
    parser.add_argument("--block-rows", type=int, default=512,
                        help="core rows per strip; a 256 row halo is added")
    parser.add_argument("--window", default=None,
                        help="optional 'r0,r1,c0,c1' window for a smoke run")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="cpu is the unchanged default path; cuda fails loudly if "
                             "CUDA is unavailable rather than falling back")
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

    inference_batch = resolve_execution(args.device, args.precision,
                                        args.inference_batch_size)

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
    run_config = checkpoint.get("run_config") or {}
    if run_config.get("normalisation_file") and args.normalisation is None:
        raise SystemExit(
            "this checkpoint was trained with per-pair normalisation (%s); pass "
            "--normalisation so inference uses the same statistics"
            % run_config["normalisation_file"])
    if args.normalisation is not None:
        # Training normalises each pair with its own statistics, so inference must
        # use the same ones or it scores a model on differently scaled inputs.
        from terrashift.preprocessing.normalisation import NormalisationError, pair_statistics, require_statistics
        try:
            payload = require_statistics(args.normalisation, BANDS, (args.pair,))
        except NormalisationError as error:
            raise SystemExit("normalisation: %s" % error) from error
        if args.pair not in payload["pairs"]:
            raise SystemExit("normalisation file has no statistics for pair %s" % args.pair)
        mean, std = pair_statistics(payload, args.pair)
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

    run_device = torch.device(args.device)
    model = SiameseUNet(in_channels=len(BANDS), base=base).to(run_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    if args.precision == "fp16":
        model = model.half()
        model = HalfInputModel(model).to(run_device)
        model.eval()

    profile_prob = dict(driver="GTiff", height=r1 - r0, width=c1 - c0, count=1,
                        dtype="float32", crs=grid.crs,
                        transform=rasterio.windows.transform(
                            Window(c0, r0, c1 - c0, r1 - r0), grid.transform),
                        nodata=-9999.0, tiled=True, compress="deflate")
    profile_u8 = dict(profile_prob, dtype="uint8", nodata=255, compress="deflate")

    counts = {"windows": 0, "valid_pixels": 0, "predicted_positive_pixels": 0,
              "pixels": 0}
    stage_totals = {"jp2_read_s": 0.0, "normalise_s": 0.0, "inference_s": 0.0,
                    "validity_s": 0.0, "write_s": 0.0}

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

            _t0 = time.perf_counter()
            first = read_strip(p1, rows, cols)
            second = read_strip(p2, rows, cols, shift=shift)
            _t1 = time.perf_counter()
            x1 = torch.from_numpy(normalise(first, mean, std))
            x2 = torch.from_numpy(normalise(second, mean, std))
            _t2 = time.perf_counter()

            probabilities, _ = sliding_window_predict(
                model, x1, x2, patch_size=patch_size, stride=args.stride,
                device=run_device, threshold=None, batch_size=inference_batch)
            if run_device.type == "cuda":
                torch.cuda.synchronize()
            _t3 = time.perf_counter()

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

            _t4 = time.perf_counter()
            # The output raster is cropped to the requested window, so the write offset
            # must be window-relative. With the full tile (r0 = c0 = 0) this is a no-op.
            window = Window(c0 - c0, core_start - r0, c1 - c0, core_stop - core_start)
            prob_ds.write(core_prob_out, 1, window=window)
            mask_ds.write(core_mask, 1, window=window)
            valid_ds.write(core_valid.astype(np.uint8), 1, window=window)
            _t5 = time.perf_counter()
            if args.profile:
                stage_totals["jp2_read_s"] += _t1 - _t0
                stage_totals["normalise_s"] += _t2 - _t1
                stage_totals["inference_s"] += _t3 - _t2
                stage_totals["validity_s"] += _t4 - _t3
                stage_totals["write_s"] += _t5 - _t4

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
                           normalisation=("per-pair z-score from "
                                          + str(args.normalisation) if args.normalisation
                                          else "per-band z-score from checkpoint"),
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
        "precision": args.precision,
        "inference_batch_size": inference_batch,
        "fp16_method": ("model.half() with half inputs" if args.precision == "fp16"
                        else None),
        "stage_timings_s": ({key: round(value, 3) for key, value in stage_totals.items()}
                            if args.profile else None),
        "counts": counts,
        "valid_fraction": counts["valid_pixels"] / max(counts["pixels"], 1),
        "positive_fraction_of_valid": (counts["predicted_positive_pixels"]
                                       / max(counts["valid_pixels"], 1)),
        "status": args.status,
        "note": _run_note(args.status),
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
