"""Supplementary mask comparison at the thresholds that produce reviewable volume."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.manifest import load_manifest, pair_config
from terrashift.data.safeloader import SafeProduct

THRESHOLDS = (0.5, 0.7, 0.9)
OVERVIEW = 1100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    args = parser.parse_args()
    out = Path(args.output)
    manifest = load_manifest(args.manifest)
    for pair in ("A", "B"):
        cfg = pair_config(manifest, pair)
        after = SafeProduct(cfg["path_date2"])
        with rasterio.open(out / "rasters" / ("pair_%s_scores.tif" % pair)) as handle:
            height, width = handle.height, handle.width
        scale = max(1, int(np.ceil(max(height, width) / OVERVIEW)))
        shape = (height // scale, width // scale)
        stack = [after.read_band(band, out_shape=shape)[0] for band in ("B04", "B03", "B02")]
        background = np.stack(stack, axis=-1).astype(np.float32)
        del stack
        for channel in range(3):
            plane = background[..., channel]
            finite = np.isfinite(plane)
            low, high = np.percentile(plane[finite], [2, 98])
            background[..., channel] = np.clip((plane - low) / max(high - low, 1e-6), 0, 1)
        figure, axes = plt.subplots(1, 3, figsize=(19, 6.6))
        for axis, threshold in zip(axes, THRESHOLDS):
            path = out / "rasters" / ("pair_%s_candidates_%03d.tif"
                                      % (pair, int(round(threshold * 100))))
            with rasterio.open(path) as handle:
                mask = handle.read(1, out_shape=shape, resampling=Resampling.nearest)
            shown = np.where(mask == 255, np.nan, mask.astype(float))
            axis.imshow(background, alpha=0.6)
            axis.imshow(np.ma.masked_invalid(shown), cmap="autumn", vmin=0, vmax=1, alpha=0.8)
            axis.set_title("threshold %.2f" % threshold, fontsize=11)
            axis.axis("off")
            del mask, shown
        figure.suptitle("Pair %s candidate masks at the reviewable thresholds "
                        "(display %dx downsampled; counts from native resolution)"
                        % (pair, scale), fontsize=12)
        figure.tight_layout()
        target = out / "figures" / ("pair_%s_candidate_comparison_supplementary.png" % pair)
        figure.savefig(target, dpi=130)
        plt.close(figure)
        print("wrote", target.name)
        del background


if __name__ == "__main__":
    main()
