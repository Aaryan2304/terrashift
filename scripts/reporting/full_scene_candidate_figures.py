"""Display figures for full-scene candidate mapping.

Downsampling happens only here, at read time, and only for display. Every number in the
analysis comes from native-resolution rasters. RGB is read one band at a time through
SafeProduct with an explicit out_shape so no full-scene array is ever held in RAM.

    python scripts/reporting/full_scene_candidate_figures.py --output <dir>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.manifest import load_manifest, pair_config
from terrashift.data.safeloader import SafeProduct

PAIRS = ("A", "B")
THRESHOLDS = (0.03, 0.05, 0.10)
OVERVIEW = 1100          # display pixels on the long side
CLOSE = 640              # native-resolution close-up size
NODATA = -9999.0


def suffix(threshold):
    return "%03d" % int(round(threshold * 100))


def memory_mb():
    info = open("/proc/meminfo").read()
    return (int(info.split("MemAvailable:")[1].split()[0]) // 1024,
            int(info.split("MemTotal:")[1].split()[0]) // 1024)


def guard(stage, floor_mb=1200):
    available, total = memory_mb()
    print("[mem] %-32s available %5d MB / %d MB" % (stage, available, total))
    if available < floor_mb:
        raise SystemExit("stage %s: MemAvailable %d MB below floor" % (stage, available))


def rgb(product, out_shape=None, window=None):
    """True colour, percentile-stretched, one band at a time."""
    stack = [product.read_band(band, window=window, out_shape=out_shape)[0]
             for band in ("B04", "B03", "B02")]
    data = np.stack(stack, axis=-1).astype(np.float32)
    del stack
    for channel in range(3):
        plane = data[..., channel]
        finite = np.isfinite(plane)
        if not finite.any():
            continue
        low, high = np.percentile(plane[finite], [2, 98])
        data[..., channel] = np.clip((plane - low) / max(high - low, 1e-6), 0, 1)
    return data


def save(figure, path):
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print("  wrote", path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    args = parser.parse_args()
    out = Path(args.output)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)

    for pair in PAIRS:
        guard("figures %s" % pair)
        cfg = pair_config(manifest, pair)
        before_product = SafeProduct(cfg["path_date1"])
        after_product = SafeProduct(cfg["path_date2"])
        scores_path = out / "rasters" / ("pair_%s_scores.tif" % pair)
        with rasterio.open(scores_path) as handle:
            height, width = handle.height, handle.width
            scale = max(1, int(np.ceil(max(height, width) / OVERVIEW)))
            shape = (height // scale, width // scale)

        # ---- overview: before / after ------------------------------------
        figure, axes = plt.subplots(1, 2, figsize=(16, 8.2))
        for axis, product, label in ((axes[0], before_product, "before (date 1)"),
                                     (axes[1], after_product, "after (date 2)")):
            axis.imshow(rgb(product, out_shape=shape))
            axis.set_title(label, fontsize=11)
            axis.axis("off")
        figure.suptitle("Pair %s full scene, true colour (display %dx downsampled)"
                        % (pair, scale), fontsize=12)
        figure.tight_layout()
        save(figure, figures / ("pair_%s_rgb_overview.png" % pair))

        # ---- continuous score overview -----------------------------------
        with rasterio.open(scores_path) as handle:
            scores = handle.read(1, out_shape=shape, resampling=Resampling.nearest)
        with rasterio.open(out / "rasters" / ("pair_%s_valid_mask.tif" % pair)) as handle:
            valid = handle.read(1, out_shape=shape, resampling=Resampling.nearest).astype(bool)
        display = np.where(valid, scores, np.nan)
        figure, axis = plt.subplots(figsize=(9, 8))
        image = axis.imshow(display, cmap="magma", vmin=0, vmax=1)
        axis.set_title("Pair %s continuous model score (sigmoid output, not calibrated)\n"
                       "invalid pixels transparent" % pair, fontsize=10)
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.035, label="model score")
        figure.tight_layout()
        save(figure, figures / ("pair_%s_score_overview.png" % pair))
        del scores, valid, display

        # ---- candidate masks at three thresholds -------------------------
        background = rgb(after_product, out_shape=shape)
        figure, axes = plt.subplots(1, 3, figsize=(19, 6.6))
        for axis, threshold in zip(axes, THRESHOLDS):
            with rasterio.open(out / "rasters" / ("pair_%s_candidates_%s.tif"
                                                  % (pair, suffix(threshold)))) as handle:
                mask = handle.read(1, out_shape=shape, resampling=Resampling.nearest)
            shown = np.where(mask == 255, np.nan, mask.astype(float))
            axis.imshow(background, alpha=0.55)
            axis.imshow(np.ma.masked_invalid(shown), cmap="autumn", vmin=0, vmax=1,
                        alpha=0.75)
            axis.set_title("threshold %.2f" % threshold, fontsize=11)
            axis.axis("off")
            del mask, shown
        figure.suptitle("Pair %s candidate masks, same extent, display %dx downsampled "
                        "(counts from native resolution)" % (pair, scale), fontsize=12)
        figure.tight_layout()
        save(figure, figures / ("pair_%s_candidate_comparison.png" % pair))
        del background

        # ---- close-ups around the largest candidate component ------------
        inventory = list(csv.DictReader(open(
            out / "components" / ("pair_%s_component_inventory_%s.csv" % (pair, suffix(0.10))))))
        if inventory:
            biggest = max(inventory, key=lambda r: int(r["pixels"]))
            centre_row, centre_col = int(float(biggest["centroid_row"])), int(float(biggest["centroid_col"]))
            row0 = max(0, min(height - CLOSE, centre_row - CLOSE // 2))
            col0 = max(0, min(width - CLOSE, centre_col - CLOSE // 2))
            window = Window(col0, row0, CLOSE, CLOSE)
            figure, axes = plt.subplots(2, 4, figsize=(19, 9.6))
            axes[0, 0].imshow(rgb(before_product, window=window))
            axes[0, 0].set_title("before", fontsize=9)
            axes[0, 1].imshow(rgb(after_product, window=window))
            axes[0, 1].set_title("after", fontsize=9)
            with rasterio.open(scores_path) as handle:
                crop = handle.read(1, window=window)
            crop = np.where(crop == NODATA, np.nan, crop)
            image = axes[0, 2].imshow(crop, cmap="magma", vmin=0, vmax=1)
            axes[0, 2].set_title("model score", fontsize=9)
            figure.colorbar(image, ax=axes[0, 2], fraction=0.04)
            for index, threshold in enumerate(THRESHOLDS):
                with rasterio.open(out / "rasters" / ("pair_%s_candidates_%s.tif"
                                                      % (pair, suffix(threshold)))) as handle:
                    mask = handle.read(1, window=window)
                shown = np.where(mask == 255, np.nan, mask.astype(float))
                axes[0, 3].imshow(rgb(after_product, window=window), alpha=0.5)
                axes[0, 3].imshow(np.ma.masked_invalid(shown), cmap="autumn", vmin=0,
                                  vmax=1, alpha=0.7)
                axes[0, 3].set_title("0.03 / 0.05 / 0.10 overlay", fontsize=9)
                axes[1, index].imshow(np.ma.masked_invalid(shown), cmap="autumn",
                                      vmin=0, vmax=1)
                axes[1, index].set_title("candidate mask @ %.2f" % threshold, fontsize=9)
                del mask, shown
            axes[1, 3].imshow(crop, cmap="magma", vmin=0, vmax=1)
            axes[1, 3].contour(np.nan_to_num(crop, nan=0.0), levels=[0.03, 0.05, 0.10],
                               colors=["#00ff66", "#ffd400", "#ff2222"], linewidths=0.8)
            axes[1, 3].set_title("0.03 / 0.05 / 0.10 contours", fontsize=9)
            for axis in axes.ravel():
                axis.axis("off")
            figure.suptitle("Pair %s close-up around the largest candidate region "
                            "(native resolution, %d px window)" % (pair, CLOSE), fontsize=11)
            figure.tight_layout()
            save(figure, figures / ("pair_%s_closeup_largest.png" % pair))
            del crop

        # ---- component size examples -------------------------------------
        if inventory:
            by_size = sorted(inventory, key=lambda r: -int(r["pixels"]))
            picks = [("largest", by_size[0])]
            if len(by_size) > 2:
                picks.append(("median", by_size[len(by_size) // 2]))
            picks.append(("smallest", by_size[-1]))
            figure, axes = plt.subplots(1, len(picks), figsize=(6.2 * len(picks), 6.2))
            for axis, (label, row) in zip(np.atleast_1d(axes), picks):
                size = max(96, min(512, int(np.sqrt(int(row["pixels"]))) * 6))
                centre_row = int(float(row["centroid_row"]))
                centre_col = int(float(row["centroid_col"]))
                r0 = max(0, min(height - size, centre_row - size // 2))
                c0 = max(0, min(width - size, centre_col - size // 2))
                window = Window(c0, r0, size, size)
                axis.imshow(rgb(after_product, window=window))
                with rasterio.open(out / "rasters" / ("pair_%s_candidates_010.tif" % pair)) as handle:
                    mask = handle.read(1, window=window)
                shown = np.where(mask == 255, np.nan, mask.astype(float))
                axis.imshow(np.ma.masked_invalid(shown), cmap="autumn", vmin=0, vmax=1,
                            alpha=0.65)
                axis.set_title("%s component: %d px, %.4f km2\nscore %.3f"
                               % (label, int(row["pixels"]), float(row["area_km2"]),
                                  float(row["mean_score"])), fontsize=9)
                axis.axis("off")
                del mask, shown
            figure.suptitle("Pair %s candidate components at 0.10 — size examples"
                            % pair, fontsize=11)
            figure.tight_layout()
            save(figure, figures / ("pair_%s_component_examples.png" % pair))
        del before_product, after_product
    print("figures complete")


if __name__ == "__main__":
    main()
