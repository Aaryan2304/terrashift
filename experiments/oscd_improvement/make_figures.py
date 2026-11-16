"""Render representative predictions with false positives and false negatives.

For each validation city this picks the largest false-positive cluster and the
largest false-negative cluster at the chosen operating point, crops around them,
and draws four views: date 1, date 2, truth against prediction, and an error map.

    python experiments/oscd_improvement/make_figures.py \
        --predictions experiments/oscd_improvement/predictions/baseline_seed42_val.npz \
        --threshold 0.90 --min-area 64 \
        --output-dir experiments/oscd_improvement/results/figures

Read-only with respect to the baseline outputs.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage as ndi
from rasterio.windows import Window

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utils.imagery import RGB_BANDS
from utils.metrics import metrics_from_counts

CROP = 160


def load(path):
    data = np.load(path)
    cities = sorted({key.split("_", 1)[1] for key in data.files if key.startswith("prob_")})
    return {city: (data["prob_" + city], data["truth_" + city]) for city in cities}


def image_dirs(index_path):
    """Map city to its two rectified image folders from the patch index."""
    import csv
    dirs = {}
    with open(index_path, newline="") as handle:
        for row in csv.DictReader(handle):
            dirs.setdefault(row["city"], (row["image1_dir"], row["image2_dir"]))
    return dirs


def read_rgb(directory, window):
    channels = []
    for band in RGB_BANDS:
        with rasterio.open(os.path.join(directory, band + ".tif")) as dataset:
            channels.append(dataset.read(1, window=window).astype(np.float32))
    return np.stack(channels, axis=-1)


def stretch(first, second):
    limits = []
    for channel in range(3):
        sample = np.concatenate([first[..., channel].ravel(), second[..., channel].ravel()])
        sample = sample[np.isfinite(sample)]
        limits.append(np.percentile(sample, [2, 98]))

    def apply(image):
        out = np.zeros_like(image, dtype=np.float32)
        for channel, (low, high) in enumerate(limits):
            out[..., channel] = np.clip((image[..., channel] - low) / max(high - low, 1e-9),
                                        0.0, 1.0)
        return np.nan_to_num(out)

    return apply(first), apply(second)


def clean(mask, min_area):
    if min_area <= 0:
        return mask
    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def biggest_cluster(mask, minimum=30):
    """Centre of the largest connected part of a mask, or None."""
    labels, count = ndi.label(mask)
    if count == 0:
        return None
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    label = int(np.argmax(sizes))
    if sizes[label] < minimum:
        return None
    return tuple(int(v) for v in ndi.center_of_mass(mask, labels, label))


def crop_slice(centre, shape):
    row = max(0, min(shape[0] - CROP, centre[0] - CROP // 2))
    col = max(0, min(shape[1] - CROP, centre[1] - CROP // 2))
    return slice(row, row + CROP), slice(col, col + CROP)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--index", default=os.path.join(ROOT, "outputs", "patches",
                                                        "oscd_original_index.csv"))
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise SystemExit("refusing to write into a non-empty %s" % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    data = load(args.predictions)
    dirs = image_dirs(args.index)
    summary = {"threshold": args.threshold, "min_area_px": args.min_area, "cities": {}}

    for city, (prob, truth) in data.items():
        prediction = clean(prob >= args.threshold, args.min_area)
        target = truth == 1
        fp = prediction & ~target
        fn = ~prediction & target
        tp, f = int((prediction & target).sum()), int(fp.sum())
        m, tn = int(fn.sum()), int((~prediction & ~target).sum())
        pixel = metrics_from_counts(tp, f, m, tn)
        summary["cities"][city] = {"precision": pixel["precision"],
                                   "recall": pixel["recall"], "f1": pixel["f1"],
                                   "tp": tp, "fp": f, "fn": m}

        image1_dir, image2_dir = dirs[city]
        for kind, mask in (("false_positives", fp), ("false_negatives", fn)):
            centre = biggest_cluster(mask)
            if centre is None:
                print("%-10s %-16s nothing to show" % (city, kind))
                continue
            rows, cols = crop_slice(centre, prob.shape)
            window = Window(cols.start, rows.start,
                            cols.stop - cols.start, rows.stop - rows.start)
            a = read_rgb(image1_dir, window)
            b = read_rgb(image2_dir, window)
            a, b = stretch(a, b)

            figure, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
            axes[0].imshow(a, interpolation="nearest"); axes[0].set_title("date 1")
            axes[1].imshow(b, interpolation="nearest"); axes[1].set_title("date 2")
            axes[2].imshow(b, interpolation="nearest")
            axes[2].imshow(np.ma.masked_where(~target[rows, cols],
                                              np.ones_like(fn[rows, cols], dtype=float)),
                           cmap="Greens", alpha=0.45)
            axes[2].imshow(np.ma.masked_where(~prediction[rows, cols],
                                              np.ones_like(fn[rows, cols], dtype=float)),
                           cmap="Reds", alpha=0.30)
            axes[2].set_title("truth (green) vs prediction (red)")
            error = np.zeros((*fn[rows, cols].shape, 3), dtype=np.float32)
            error[fn[rows, cols]] = (0.20, 0.35, 1.00)     # missed change, blue
            error[fp[rows, cols]] = (1.00, 0.20, 0.20)     # false alarm, red
            hit = (prediction & target)[rows, cols]
            error[hit] = (0.20, 0.85, 0.30)                # correct detection, green
            axes[3].imshow(b, interpolation="nearest", alpha=0.55)
            axes[3].imshow(error, interpolation="nearest", alpha=0.85)
            axes[3].set_title("TP green, FP red, FN blue")
            for axis in axes:
                axis.set_xticks([]); axis.set_yticks([])
            figure.suptitle("%s — %s at threshold %.2f, min area %d px"
                            % (city, kind.replace("_", " "), args.threshold, args.min_area))
            path = os.path.join(args.output_dir, "%s_%s.png" % (city, kind))
            figure.savefig(path, dpi=120)
            plt.close(figure)
            print("wrote", path)

    with open(os.path.join(args.output_dir, "figure_metrics.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print("wrote", os.path.join(args.output_dir, "figure_metrics.json"))


if __name__ == "__main__":
    main()
