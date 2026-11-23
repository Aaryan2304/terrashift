from pathlib import Path
"""Render the split mask over the two dates, so the partition can be inspected.

The overlay colours are:

    1 train       green
    2 validation  orange
    3 test        blue
    0 guard band  grey

Value 0 is both the guard band and the raster nodata value, so an area outside
the analysed footprint is drawn the same way as a guard band. Both mean "not
eligible for a patch".

The figure is a review aid. It does not prove that a split is geographically
independent; it shows where the boundary falls so a human can check it against
roads, built-up areas and the coastline.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.manifest import load_manifest, pair_paths
from terrashift.data.safeloader import SafeProduct
from terrashift.geospatial.imagery import rgb, stretch_pair

COLORS = {
    1: (0.17, 0.63, 0.17, 0.45),
    2: (1.00, 0.50, 0.05, 0.45),
    3: (0.12, 0.47, 0.71, 0.45),
    0: (0.45, 0.45, 0.45, 0.35),
}
NAMES = {1: "train", 2: "validation", 3: "test", 0: "guard band / outside"}


def colourise(split):
    """RGBA overlay for a split raster."""
    rgba = np.zeros(split.shape + (4,), dtype=np.float32)
    for code, colour in COLORS.items():
        rgba[split == code] = colour
    return rgba


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["A", "B"], required=True)
    parser.add_argument("--partition-dir", default=os.path.join(ROOT, "outputs", "partitions"))
    parser.add_argument("--out", default=None,
                        help="PNG path; defaults to outputs/figures/partition_<PAIR>.png")
    args = parser.parse_args()

    part_path = os.path.join(args.partition_dir, "pair_%s_partition_20m.tif" % args.pair)
    if not os.path.isfile(part_path):
        parser.error("partition raster not found: %s" % part_path)

    with rasterio.open(part_path) as ds:
        split = ds.read(1)
        tags = ds.tags()

    manifest = load_manifest()
    s1, s2 = pair_paths(manifest, args.pair)
    first = SafeProduct(s1)
    second = SafeProduct(s2)
    shape = split.shape
    a = rgb(first, shape)
    b = rgb(second, shape)
    a, b = stretch_pair(a, b)
    overlay = colourise(split)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(a)
    axes[0].set_title("Date 1")
    axes[1].imshow(b)
    axes[1].set_title("Date 2")
    axes[2].imshow(b)
    axes[2].imshow(overlay)
    axes[2].set_title("Date 2 + split (axis %s, %s)" %
                      (tags.get("split_axis", "?"), tags.get("train_from", "?")))
    handles = [Patch(facecolor=COLORS[c][:3], alpha=0.6, label=NAMES[c])
               for c in (1, 2, 3, 0)]
    axes[2].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Pair %s partition over the imagery (20 m grid)" % args.pair)

    out = args.out or os.path.join(ROOT, "outputs", "figures",
                                   "partition_%s.png" % args.pair)
    if os.path.exists(out):
        parser.error("refusing to overwrite existing figure: %s (pass --out)" % out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)

    counts = {NAMES[c]: int((split == c).sum()) for c in (1, 2, 3, 0)}
    print("wrote", out)
    print("split pixel counts on the 20 m grid:", json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
