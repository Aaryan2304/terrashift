"""Build a visual review panel for a pair's before/after and score tail.

The rendered RGB views use B04/B03/B02 with one common percentile stretch.
The score overlay is the top 0.1% of valid scores after morphology. It is only
an analyst-review layer, not a validated change map.

``--score-dir`` must point at a baseline run that applied the paired
observation-valid mask; the score raster's nodata value then marks exactly the
pixels that were never eligible.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.manifest import load_manifest, pair_paths
from data.safeloader import SafeProduct
from utils.geometry import clean_binary_mask
from utils.imagery import rgb, stretch_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["A", "B"], required=True)
    parser.add_argument("--score-dir", default=os.path.join(ROOT, "outputs", "baseline"),
                        help="directory holding change_score_<PAIR>.tif")
    parser.add_argument("--out", default=None,
                        help="PNG path; defaults to outputs/figures/pair_<PAIR>_visual_review.png")
    parser.add_argument("--percentile", type=float, default=99.9)
    args = parser.parse_args()

    manifest = load_manifest()
    s1, s2 = pair_paths(manifest, args.pair)
    p1 = SafeProduct(s1)
    p2 = SafeProduct(s2)
    with rasterio.open(p1.band_files["B08"]) as ds:
        scale = 20.0 / ds.res[0]
        shape = (int(round(ds.height / scale)), int(round(ds.width / scale)))
    a = rgb(p1, shape)
    b = rgb(p2, shape)
    a, b = stretch_pair(a, b)

    score_path = os.path.join(args.score_dir, "change_score_%s.tif" % args.pair)
    if not os.path.isfile(score_path):
        parser.error("score raster not found: %s" % score_path)
    with rasterio.open(score_path) as ds:
        score = ds.read(1, masked=True).filled(np.nan)
    valid = np.isfinite(score)
    threshold = float(np.percentile(score[valid], args.percentile))
    overlay = (score > threshold) & valid
    overlay = clean_binary_mask(overlay, valid=valid, min_area_px=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(a); axes[0].set_title("Before — B04/B03/B02")
    axes[1].imshow(b); axes[1].set_title("After — B04/B03/B02")
    axes[2].imshow(b)
    axes[2].imshow(np.ma.masked_where(~overlay, overlay), cmap="autumn", alpha=0.75,
                   vmin=0, vmax=1)
    axes[2].set_title("After + top %.1f%% score tail (review only)" % (100.0 - args.percentile))
    for ax in axes: ax.axis("off")
    fig.suptitle("Pair %s — 20 m display; shared 2–98%% RGB stretch" % args.pair)
    out = args.out or os.path.join(ROOT, "outputs", "figures",
                                   "pair_%s_visual_review.png" % args.pair)
    if os.path.exists(out):
        parser.error("refusing to overwrite existing figure: %s (pass --out)" % out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out, "score threshold", threshold,
          "overlay pixels", int(overlay.sum()))


if __name__ == "__main__":
    main()
