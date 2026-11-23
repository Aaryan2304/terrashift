"""Distribution comparison, spatial concentration and representative regions.

Reads only the saved score rasters, validity masks, reviewed masks and component
inventories. No inference, no training, no annotation edits.

    python scripts/reporting/full_scene_diagnostic_visuals.py --output outputs/evaluation/full_scene_diagnostic_audit_v1
"""

from __future__ import annotations

import argparse
import csv
import json
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

LABELS = "outputs/evaluation/local_manual_v2/validation_threshold_analysis/_fine_tuned_labels.npy"
VECTORS = {"A": "outputs/evaluation/local_manual_v2_seed43_comparison/_local_0.834_s43_scores.npy",
           "B": "outputs/evaluation/local_manual_v2_seed43_comparison/_local_0.834_s43_scores.npy"}
SPLIT_A = 2759

PAIRS = ("A", "B")
FULL = "outputs/inference/full_scene_candidate_analysis_v1"
BLOCKS = 12
CROP = 640
NODATA = -9999.0


def memory_mb():
    info = open("/proc/meminfo").read()
    return int(info.split("MemAvailable:")[1].split()[0]) // 1024


def guard(stage, floor=1200):
    available = memory_mb()
    print("[mem] %-30s available %5d MB" % (stage, available))
    if available < floor:
        raise SystemExit("stage %s: memory below floor" % stage)


def block_stats(out_root, pair):
    """Mean score and high-score fraction per block, read windowed."""
    path = Path(FULL) / "rasters" / ("pair_%s_scores.tif" % pair)
    with rasterio.open(path) as handle:
        height, width = handle.height, handle.width
    rows = []
    for br in range(BLOCKS):
        for bc in range(BLOCKS):
            r0 = br * height // BLOCKS
            r1 = (br + 1) * height // BLOCKS
            c0 = bc * width // BLOCKS
            c1 = (bc + 1) * width // BLOCKS
            with rasterio.open(path) as handle:
                block = handle.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            values = block[block != NODATA]
            if not values.size:
                continue
            rows.append({"pair": pair, "block_row": br, "block_col": bc,
                         "row_start": r0, "col_start": c0,
                         "valid_pixels": int(values.size),
                         "mean_score": round(float(values.mean()), 6),
                         "median_score": round(float(np.median(values)), 6),
                         "p90_score": round(float(np.percentile(values, 90)), 6),
                         "fraction_ge_0p10": round(float((values >= 0.10).mean()), 6),
                         "fraction_ge_0p50": round(float((values >= 0.50).mean()), 6),
                         "fraction_ge_0p90": round(float((values >= 0.90).mean()), 6)})
            del block, values
    return rows, (height, width)


def rgb_pair(pair, manifest, row0, row1, col0, col1):
    cfg = pair_config(manifest, pair)
    planes = []
    for key in ("path_date1", "path_date2"):
        product = SafeProduct(cfg[key])
        window = ((row0, row1), (col0, col1))
        stack = [product.read_band(band, window=window)[0] for band in ("B04", "B03", "B02")]
        data = np.stack(stack, axis=-1).astype(np.float32)
        del stack
        for channel in range(3):
            plane = data[..., channel]
            finite = np.isfinite(plane)
            if not finite.any():
                continue
            low, high = np.percentile(plane[finite], [2, 98])
            data[..., channel] = np.clip((plane - low) / max(high - low, 1e-6), 0, 1)
        planes.append(data)
    return planes[0], planes[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    args = parser.parse_args()
    out = Path(args.output)
    (out / "full_scene_score_maps").mkdir(parents=True, exist_ok=True)
    (out / "representative_regions").mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)

    # ---- block statistics ------------------------------------------------
    all_blocks = []
    shapes = {}
    for pair in PAIRS:
        guard("blocks %s" % pair)
        rows, shape = block_stats(out, pair)
        all_blocks.extend(rows)
        shapes[pair] = shape
    if all_blocks:
        with open(out / "full_scene_block_statistics.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_blocks[0]))
            writer.writeheader()
            writer.writerows(all_blocks)
    print("wrote full_scene_block_statistics.csv (%d blocks)" % len(all_blocks))

    # ---- distribution comparison figure ----------------------------------
    hist = json.loads((out / "full_scene_histograms.json").read_text())
    class_rows = list(csv.DictReader(open(out / "validation_class_score_statistics.csv")))
    consistency_path = out / "consistency_detail.json"
    edges = np.array(hist["edges"])
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    labels_all = np.load(LABELS)
    vectors = np.load(VECTORS["B"]).astype(np.float32)
    for axis, pair in zip(axes, PAIRS):
        full_hist = np.array(hist["histograms"][pair], dtype=float)
        total = full_hist.sum()
        axis.plot(edges[:-1], full_hist / max(total, 1), lw=1.6, color="#333333",
                  label="full scene, all valid pixels (n=%s)" % "{:,}".format(int(total)))
        sl = slice(0, SPLIT_A) if pair == "A" else slice(SPLIT_A, None)
        lab = labels_all[sl]
        val = vectors[sl]
        for label, colour, name in ((1, "#c0392b", "reviewed change (1)"),
                                    (0, "#1f6f3f", "reviewed no-change (0)")):
            selected = val[lab == label]
            class_hist, _ = np.histogram(selected, bins=100, range=(0.0, 1.0))
            axis.plot(edges[:-1], class_hist / max(selected.size, 1), lw=1.6, color=colour,
                      label="%s n=%s, median %.3f"
                      % (name, "{:,}".format(selected.size), float(np.median(selected))))
        axis.set_xlabel("model score"); axis.set_ylabel("fraction of valid pixels per bin")
        axis.set_title("Pair %s — full scene vs reviewed classes" % pair, fontsize=10)
        axis.grid(alpha=0.3); axis.legend(fontsize=8)
    figure.suptitle("Score distributions: full scene against the two reviewed classes "
                    "(100 bins over 0–1)", fontsize=11)
    figure.tight_layout()
    figure.savefig(out / "score_distribution_comparison.png", dpi=150)
    plt.close(figure)

    # ---- score maps ------------------------------------------------------
    for pair in PAIRS:
        guard("map %s" % pair)
        height, width = shapes[pair]
        scale = max(1, int(np.ceil(max(height, width) / 1100)))
        shape = (height // scale, width // scale)
        with rasterio.open(Path(FULL) / "rasters" / ("pair_%s_scores.tif" % pair)) as handle:
            scores = handle.read(1, out_shape=shape, resampling=Resampling.nearest)
        shown = np.where(scores == NODATA, np.nan, scores)
        figure, axis = plt.subplots(figsize=(9, 8))
        image = axis.imshow(shown, cmap="magma", vmin=0, vmax=1)
        for k in range(1, BLOCKS):
            axis.axhline(k * shape[0] / BLOCKS - 0.5, color="cyan", lw=0.4, alpha=0.5)
            axis.axvline(k * shape[1] / BLOCKS - 0.5, color="cyan", lw=0.4, alpha=0.5)
        axis.set_title("Pair %s full-scene score (sigmoid output, not calibrated)\n"
                       "cyan grid = %dx%d analysis blocks" % (pair, BLOCKS, BLOCKS),
                       fontsize=9)
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.035, label="score")
        figure.tight_layout()
        figure.savefig(out / "full_scene_score_maps" / ("pair_%s_score_map.png" % pair), dpi=130)
        plt.close(figure)
        del scores, shown
        print("  wrote score map", pair)

    # ---- representative regions -----------------------------------------
    for pair in PAIRS:
        guard("regions %s" % pair)
        blocks = [b for b in all_blocks if b["pair"] == pair]
        by_mean = sorted(blocks, key=lambda b: float(b["mean_score"]))
        median_block = min(blocks, key=lambda b: abs(float(b["mean_score"])
                                                     - np.median([float(x["mean_score"])
                                                                  for x in blocks])))
        selections = [("lowest_mean_score", by_mean[0]),
                      ("median_mean_score", median_block),
                      ("highest_mean_score", by_mean[-1])]
        # large connected region at 0.50
        inventory_path = Path(FULL) / "components" / ("pair_%s_component_inventory_050.csv" % pair)
        if inventory_path.exists():
            inventory = list(csv.DictReader(open(inventory_path)))
            if inventory:
                biggest = max(inventory, key=lambda r: int(r["pixels"]))
                selections.append(("large_connected_region_at_0p50", {
                    "row_start": int(float(biggest["centroid_row"])),
                    "col_start": int(float(biggest["centroid_col"])),
                    "mean_score": float(biggest["mean_score"]),
                    "valid_pixels": int(biggest["pixels"])}))
        # fragmented high-score area: block with most component centroids at 0.90
        small_path = Path(FULL) / "components" / ("pair_%s_component_inventory_090.csv" % pair)
        if small_path.exists():
            small = list(csv.DictReader(open(small_path)))
            counts = {}
            for row in small:
                key = (int(float(row["centroid_row"])) * BLOCKS // shapes[pair][0],
                       int(float(row["centroid_col"])) * BLOCKS // shapes[pair][1])
                counts[key] = counts.get(key, 0) + 1
            if counts:
                (br, bc), count = max(counts.items(), key=lambda kv: kv[1])
                selections.append(("fragmented_high_score_area", {
                    "row_start": br * shapes[pair][0] // BLOCKS,
                    "col_start": bc * shapes[pair][1] // BLOCKS,
                    "mean_score": float("nan"), "valid_pixels": count}))

        for name, block in selections:
            height, width = shapes[pair]
            row0 = max(0, min(height - CROP, int(block["row_start"]) - CROP // 2))
            col0 = max(0, min(width - CROP, int(block["col_start"]) - CROP // 2))
            before, after = rgb_pair(pair, manifest, row0, row0 + CROP, col0, col0 + CROP)
            with rasterio.open(Path(FULL) / "rasters" / ("pair_%s_scores.tif" % pair)) as handle:
                crop = handle.read(1, window=Window(col0, row0, CROP, CROP))
            crop = np.where(crop == NODATA, np.nan, crop)
            figure, axes = plt.subplots(1, 4, figsize=(19, 5.2))
            axes[0].imshow(before); axes[0].set_title("before", fontsize=9)
            axes[1].imshow(after); axes[1].set_title("after", fontsize=9)
            image = axes[2].imshow(crop, cmap="magma", vmin=0, vmax=1)
            axes[2].set_title("model score", fontsize=9)
            figure.colorbar(image, ax=axes[2], fraction=0.04)
            axes[3].imshow(after)
            overlay = np.where(np.isnan(crop), np.nan, (crop >= 0.10).astype(float))
            axes[3].imshow(np.ma.masked_invalid(overlay), cmap="autumn", vmin=0, vmax=1,
                           alpha=0.55)
            axes[3].set_title("score >= 0.10 overlay", fontsize=9)
            for axis in axes:
                axis.axis("off")
            figure.suptitle("Pair %s — %s (row %d, col %d), %d px at native resolution"
                            % (pair, name.replace("_", " "), row0, col0, CROP), fontsize=10)
            figure.tight_layout()
            target = out / "representative_regions" / ("pair_%s_%s.png" % (pair, name))
            figure.savefig(target, dpi=130)
            plt.close(figure)
            print("  wrote", target.name)
            del before, after, crop, overlay

    (out / "visuals_summary.json").write_text(json.dumps(
        {"blocks": BLOCKS, "crop_px": CROP,
         "note": "all display images downsampled or windowed; metrics from native rasters"},
        indent=1) + "\n")
    print("visuals complete")


if __name__ == "__main__":
    main()
