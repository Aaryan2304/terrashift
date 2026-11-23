"""Full-scene diagnostic: does the full-scene score raster agree with the saved
validation score vector at the same reviewed pixels?

No inference, no training, no checkpoint access. The comparison is possible because both
paths use the same canonical tiling: ``terrashift.data.validation_windows.select_windows`` places
windows with ``inference_strips(512, 256)`` and ``window_grid`` over the full tile width,
which is what ``scripts/inference/predict_safe_pair.py`` runs. If the coordinate mapping is correct
the two score vectors must therefore agree to float rounding at every reviewed pixel.

    python scripts/evaluation/full_scene_diagnostic.py --output outputs/evaluation/full_scene_diagnostic_audit_v1 --stage consistency
    python scripts/evaluation/full_scene_diagnostic.py --output ... --stage statistics
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]

from terrashift.metrics import average_precision, precision_recall_arrays

PAIRS = ("A", "B")
NODATA = -9999.0
SCORES = {
    "A": "outputs/inference/full_scene_candidate_analysis_v1/rasters/pair_A_scores.tif",
    "B": "outputs/inference/full_scene_candidate_analysis_v1/rasters/pair_B_scores.tif",
}
VALID = {
    "A": "outputs/inference/full_scene_candidate_analysis_v1/rasters/pair_A_valid_mask.tif",
    "B": "outputs/inference/full_scene_candidate_analysis_v1/rasters/pair_B_valid_mask.tif",
}
VECTORS = {
    "A": "outputs/evaluation/local_manual_v2_seed43_comparison/_local_0.834_s43_scores.npy",
    "B": "outputs/evaluation/local_manual_v2_seed43_comparison/_local_0.834_s43_scores.npy",
}
LABELS = "outputs/evaluation/local_manual_v2/validation_threshold_analysis/_fine_tuned_labels.npy"
DIAG = {"A": "outputs/evaluation/local_manual_v2_seed43_comparison/spatial_audit/rasters/pair_A_oscd_diagnosis_0.50.tif",
        "B": "outputs/evaluation/local_manual_v2_seed43_comparison/spatial_audit/rasters/pair_B_oscd_diagnosis_0.50.tif"}
SPLIT_A = 2759
TOLERANCES = (1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def slice_for(pair):
    return slice(0, SPLIT_A) if pair == "A" else slice(SPLIT_A, None)


def reviewed_labels(pair):
    """Reviewed mask, labels and the stripe raster's column offset, from a diagnosis raster."""
    with rasterio.open(DIAG[pair]) as handle:
        codes = handle.read(1)
        stripe_transform = handle.transform
        stripe_shape = (handle.height, handle.width)
    with rasterio.open(SCORES[pair]) as handle:
        full_transform = handle.transform
        full_shape = (handle.height, handle.width)
    offset = int(round((stripe_transform.c - full_transform.c) / full_transform.a))
    reviewed = codes != 0
    labels = np.zeros(codes.shape, dtype=np.int8)
    labels[reviewed] = np.isin(codes[reviewed], (2, 4)).astype(np.int8)
    return reviewed, labels, offset, stripe_shape, full_shape, stripe_transform, full_transform


def describe(values):
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            "p05": float(np.percentile(values, 5)), "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95))}


def stage_consistency(out, args):
    labels_all = np.load(LABELS)
    rows_out = []
    detail = {}
    for pair in PAIRS:
        reviewed, labels, offset, stripe_shape, full_shape, st, ft = reviewed_labels(pair)
        expected_offset = 7736
        if offset != expected_offset:
            raise SystemExit("%s: stripe offset %d, expected %d" % (pair, offset, expected_offset))
        rows_idx, cols_idx = np.nonzero(reviewed)
        saved = np.load(VECTORS[pair])[slice_for(pair)].astype(np.float32)
        if saved.size != rows_idx.size:
            raise SystemExit("%s: vector %d != reviewed %d" % (pair, saved.size, rows_idx.size))
        if not np.array_equal(labels[reviewed], labels_all[slice_for(pair)]):
            raise SystemExit("%s: recovered labels differ from the saved label vector" % pair)

        r0, r1 = int(rows_idx.min()), int(rows_idx.max()) + 1
        c0, c1 = int(cols_idx.min()), int(cols_idx.max()) + 1
        with rasterio.open(SCORES[pair]) as handle:
            block = handle.read(1, window=Window(c0 + offset, r0, c1 - c0, r1 - r0))
        full = block[rows_idx - r0, cols_idx - c0].astype(np.float32)
        del block
        if not np.all(np.isfinite(full)):
            raise SystemExit("%s: full-scene scores contain NoData at reviewed pixels" % pair)

        diff = np.abs(full - saved)
        correlation = float(np.corrcoef(full, saved)[0, 1])
        # where are the largest differences?
        row_mod = rows_idx % 128
        col_global = cols_idx + offset
        col_mod = col_global % 128
        near_scene_edge = ((rows_idx < 256) | (rows_idx >= full_shape[0] - 256)
                           | (col_global < 256) | (col_global >= full_shape[1] - 256))
        near_stripe_edge = ((cols_idx < 256) | (cols_idx >= stripe_shape[1] - 256))
        big = diff > 1e-3
        entry = {
            "pair": pair,
            "reviewed_pixels": int(saved.size),
            "full_scene_shape": list(full_shape),
            "stripe_shape": list(stripe_shape),
            "col_offset_px": offset,
            "col_offset_m": float(offset * ft.a),
            "stripe_transform_origin": [float(st.c), float(st.f)],
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "median_abs_diff": float(np.median(diff)),
            "p95_abs_diff": float(np.percentile(diff, 95)),
            "p99_abs_diff": float(np.percentile(diff, 99)),
            "pearson_r": correlation,
            "identical_within_1e6": int((diff <= TOLERANCES[0]).sum()),
            "full_vs_validation_mean": [float(full.mean()), float(saved.mean())],
            "full_vs_validation_median": [float(np.median(full)), float(np.median(saved))],
            "pixels_exceeding": {str(t): int((diff > t).sum()) for t in TOLERANCES},
            "fraction_exceeding_1e-3": float(big.mean()),
            "big_diff_near_scene_edge_fraction": float(near_scene_edge[big].mean()) if big.any() else 0.0,
            "big_diff_near_stripe_edge_fraction":
                float(near_stripe_edge[big].mean()) if big.any() else 0.0,
            "row_mod128_of_big": np.bincount(row_mod[big], minlength=128).argmax().item()
            if big.any() else -1,
            "col_mod128_of_big": np.bincount(col_mod[big], minlength=128).argmax().item()
            if big.any() else -1,
        }
        rows_out.append(entry)
        detail[pair] = {"diff": diff, "full": full, "saved": saved, "labels": labels[reviewed],
                        "rows": rows_idx, "cols": cols_idx, "offset": offset}
        print("[%s] n=%d max|d|=%.3e mean=%.3e p99=%.3e r=%.6f exceed1e-3=%d"
              % (pair, saved.size, entry["max_abs_diff"], entry["mean_abs_diff"],
                 entry["p99_abs_diff"], correlation, entry["pixels_exceeding"]["0.001"]))

    with open(out / "validation_score_consistency.csv", "w", newline="") as handle:
        fields = ["pair", "reviewed_pixels", "col_offset_px", "col_offset_m",
                  "max_abs_diff", "mean_abs_diff", "median_abs_diff", "p95_abs_diff",
                  "p99_abs_diff", "pearson_r", "identical_within_1e6",
                  "full_vs_validation_mean", "full_vs_validation_median",
                  "pixels_exceeding", "fraction_exceeding_1e-3",
                  "big_diff_near_scene_edge_fraction", "big_diff_near_stripe_edge_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)
    (out / "consistency_detail.json").write_text(json.dumps(rows_out, indent=1) + "\n")

    # ---- per-class statistics on reviewed pixels, full-scene scores ------
    class_rows = []
    for pair in PAIRS:
        data = detail[pair]
        for label in (1, 0):
            selected = data["labels"] == label
            for source, values in (("full_scene", data["full"]), ("validation_vector", data["saved"])):
                stats = describe(values[selected])
                stats.update({"pair": pair, "label": label, "source": source})
                stats["ap_on_reviewed"] = None
                class_rows.append(stats)
        # AP from the existing implementation, both score sources
        for source, values in (("full_scene", data["full"]), ("validation_vector", data["saved"])):
            ap = float(average_precision(data["labels"], values))
            for row in class_rows:
                if row["pair"] == pair and row["source"] == source:
                    row["ap_on_reviewed"] = ap
    with open(out / "validation_class_score_statistics.csv", "w", newline="") as handle:
        fields = ["pair", "label", "source", "count", "mean", "median", "std", "min", "max",
                  "p05", "p10", "p25", "p75", "p90", "p95", "ap_on_reviewed"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(class_rows)
    print("wrote validation_score_consistency.csv and validation_class_score_statistics.csv")
    return detail


def stage_statistics(out, args):
    labels_all = np.load(LABELS)
    rows = []
    histograms = {}
    coverage = {}
    for pair in PAIRS:
        with rasterio.open(SCORES[pair]) as handle:
            height, width = handle.height, handle.width
            transform = handle.transform
        hist = np.zeros(100, dtype=np.int64)
        total = 0
        chunk = 1024
        sums = 0.0
        mins, maxs = np.inf, -np.inf
        collect = []
        for r0 in range(0, height, chunk):
            r1 = min(r0 + chunk, height)
            with rasterio.open(SCORES[pair]) as handle:
                block = handle.read(1, window=Window(0, r0, width, r1 - r0))
            values = block[block != NODATA]
            if values.size:
                hist += np.histogram(values, bins=100, range=(0.0, 1.0))[0]
                total += int(values.size)
                sums += float(values.sum())
                mins = min(mins, float(values.min()))
                maxs = max(maxs, float(values.max()))
                if len(collect) < 8:
                    collect.append(np.percentile(values, [10, 50, 90]))
            del block, values
        # exact quantiles need one pass with a sampled approximation; use the histogram
        edges = np.linspace(0.0, 1.0, 101)
        cumulative = np.cumsum(hist)
        quantiles = {}
        for q in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
            target = q / 100.0 * total
            idx = int(np.searchsorted(cumulative, target, side="left"))
            quantiles["q%02d" % q] = round(float(edges[min(idx, 99)]), 3)
        # candidate coverage on the same histogram
        for threshold in [round(0.01 * i, 2) for i in range(1, 100)]:
            bin_index = min(int(threshold * 100), 99)
            coverage.setdefault(threshold, {})[pair] = float(hist[bin_index:].sum() / max(total, 1))
        mean = sums / max(total, 1)
        variance = 0.0
        for i, count in enumerate(hist):
            mid = (edges[i] + edges[i + 1]) / 2
            variance += count * (mid - mean) ** 2
        rows.append({"pair": pair, "valid_pixels": total, "mean": round(mean, 6),
                     "median_from_hist": quantiles["q50"],
                     "std_from_hist": round(float(np.sqrt(variance / max(total, 1))), 6),
                     "min": round(mins, 6), "max": round(maxs, 6), **quantiles})
        histograms[pair] = hist.tolist()
        print("[%s] full scene: %d px mean %.4f median %.3f std %.4f"
              % (pair, total, mean, quantiles["q50"], float(np.sqrt(variance / total))))
    with open(out / "full_scene_score_statistics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "full_scene_histograms.json").write_text(json.dumps(
        {"histograms": histograms, "edges": list(np.linspace(0, 1, 101)),
         "note": "100 bins over 0..1, valid pixels only"}, indent=1) + "\n")
    (out / "full_scene_candidate_coverage.json").write_text(json.dumps(coverage, indent=1) + "\n")
    print("wrote full_scene_score_statistics.csv, histograms, coverage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True, choices=["consistency", "statistics"])
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "consistency":
        stage_consistency(out, args)
    else:
        stage_statistics(out, args)


if __name__ == "__main__":
    main()
