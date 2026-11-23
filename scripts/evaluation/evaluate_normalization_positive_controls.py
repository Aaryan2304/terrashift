"""Compare normalisation methods on independently reviewed structural change.

The positive controls are the original OSCD labels: the dataset authors' annotations of new
buildings and roads, not derived from our scores.

Scope limits: OSCD tiles only, so a result bounds the correction method and says nothing
about Pair A or Pair B; development regions only (`mumbai` shares tile T43QBB with Pair A and
is excluded, the test regions are never read); raw DNs with no CRS, so no absolute
reflectance is claimed, acceptable because the score divides by each band's standard
deviation and both corrections are scale-invariant.

Metrics: `auroc`, `delta_auroc`, `preservation` (mean change score after/before),
`mae_stable` (band error on no-change pixels). Nothing is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml

ROOT = Path(__file__).resolve().parents[2]

from terrashift.preprocessing.normalize import (apply_histogram, apply_linear, fit_global,
                                  fit_histogram, reference_scaled_change_score,
                                  select_pifs)
from terrashift.metrics import roc_auc

BANDS = ("B02", "B03", "B04", "B08")
IMAGES = "Onera Satellite Change Detection dataset - Images"
TRAIN_LABELS = "Onera Satellite Change Detection dataset - Train Labels"


def read_region(images_root, city):
    """Four bands for both dates, plus the human label mask."""
    bands = {}
    for date in (1, 2):
        stack = []
        for band in BANDS:
            path = images_root / city / ("imgs_%d_rect" % date) / (band + ".tif")
            with rasterio.open(path) as ds:
                stack.append(ds.read(1).astype(np.float32))
        bands[date] = stack
    shapes = {array.shape for date in bands for array in bands[date]}
    if len(shapes) != 1:
        raise ValueError("bands of %s are not on one shape: %s" % (city, shapes))
    label_path = (images_root.parent / TRAIN_LABELS / city / "cm" / (city + "-cm.tif"))
    with rasterio.open(label_path) as ds:
        raw = ds.read(1)
    values = set(int(v) for v in np.unique(raw))
    if values == {1, 2}:
        labels = (raw == 2).astype(np.int64)
    elif values == {0, 1}:
        labels = (raw == 1).astype(np.int64)
    else:
        raise ValueError("unsupported OSCD label values for %s: %s" % (city, sorted(values)))
    return bands[1], bands[2], labels


def evaluate_method(name, ref, mov_corrected, labels, valid, score_before):
    score, complete = reference_scaled_change_score(ref, mov_corrected, valid)
    both = complete & (labels >= 0)
    truth = labels[both]
    after = score[both]
    before = score_before[both]
    changed = truth == 1
    unchanged = truth == 0

    def mean_on(values, mask):
        return float(values[mask].mean()) if mask.any() else float("nan")

    return {
        "method": name,
        "auroc_before": roc_auc(truth, before),
        "auroc_after": roc_auc(truth, after),
        "delta_auroc": roc_auc(truth, after) - roc_auc(truth, before),
        "mean_score_change_before": mean_on(before, changed),
        "mean_score_change_after": mean_on(after, changed),
        "preservation": (mean_on(after, changed) / mean_on(before, changed)
                         if mean_on(before, changed) not in (0.0, float("nan")) else float("nan")),
        "mean_score_unchanged_before": mean_on(before, unchanged),
        "mean_score_unchanged_after": mean_on(after, unchanged),
        "contrast_before": (mean_on(before, changed) / mean_on(before, unchanged)
                            if mean_on(before, unchanged) else float("nan")),
        "contrast_after": (mean_on(after, changed) / mean_on(after, unchanged)
                           if mean_on(after, unchanged) else float("nan")),
        "n_change_pixels": int(changed.sum()),
        "n_unchanged_pixels": int(unchanged.sum()),
    }


def stable_mae(ref, mov, unchanged):
    diffs = [np.abs(r[unchanged] - m[unchanged]).mean() for r, m in zip(ref, mov)]
    return float(np.mean(diffs))


def run_region(images_root, city, pif_fraction):
    ref, mov, labels = read_region(images_root, city)
    valid = np.ones(labels.shape, dtype=bool)
    for band in ref + mov:
        valid &= np.isfinite(band) & (band > 0)
    if labels.shape != valid.shape:
        raise ValueError("label shape differs from the image shape for %s" % city)

    initial, complete = reference_scaled_change_score(ref, mov, valid)
    pifs = select_pifs(initial, complete, fraction=pif_fraction)
    if int(pifs.sum()) < 100:
        raise ValueError("too few pseudo-invariant pixels for %s" % city)

    rows = []

    global_params = fit_global(ref, mov, pifs)
    rows.append(evaluate_method("global", ref, apply_linear(mov, global_params),
                                labels, valid, initial))
    hist_tables = fit_histogram(ref, mov, pifs)
    rows.append(evaluate_method("histogram", ref, apply_histogram(mov, hist_tables),
                                labels, valid, initial))
    rows.append(evaluate_method("none", ref, [m.copy() for m in mov],
                                labels, valid, initial))

    # Stable-land error, on the same basis the earlier comparison used.
    ref_complete = complete
    unchanged_full = (labels == 0) & ref_complete
    for row, corrected in zip(rows,
                              (apply_linear(mov, global_params),
                               apply_histogram(mov, hist_tables),
                               mov)):
        row["mae_stable_after"] = stable_mae(ref, corrected, unchanged_full)
    baseline_mae = stable_mae(ref, mov, unchanged_full)
    for row in rows:
        row["mae_stable_before"] = baseline_mae
        row["mae_reduction_pct"] = (100.0 * (baseline_mae - row["mae_stable_after"])
                                    / baseline_mae if baseline_mae else float("nan"))
        row["city"] = city
        row["pif_pixels"] = int(pifs.sum())
        row["valid_pixels"] = int(complete.sum())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for the CSV and JSON")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--splits", type=Path,
                        default=ROOT / "configs/oscd_region_splits.yaml")
    parser.add_argument("--pif-fraction", type=float, default=0.5)
    parser.add_argument("--cities", nargs="*", default=None,
                        help="override the development region list")
    args = parser.parse_args()

    csv_path = args.output / "positive_controls.csv"
    json_path = args.output / "positive_controls_summary.json"
    existing = [str(p) for p in (csv_path, json_path) if p.exists()]
    if existing:
        raise SystemExit("refusing to overwrite: %s" % ", ".join(existing))
    args.output.mkdir(parents=True, exist_ok=True)

    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    data_root = Path(cfg["data"]["oscd_original_root"])
    images_root = data_root / IMAGES
    splits = yaml.safe_load(args.splits.read_text(encoding="utf-8"))
    cities = args.cities or (list(splits["train"]) + list(splits["validation"]))
    forbidden = set(splits["test"]) | set(splits.get("excluded", {}))
    overlap = set(cities) & forbidden
    if overlap:
        raise SystemExit("these regions are test or excluded and must not be read: %s"
                         % sorted(overlap))

    rows = []
    for city in cities:
        try:
            region_rows = run_region(images_root, city, args.pif_fraction)
        except (FileNotFoundError, ValueError) as exc:
            print("  skip %s: %s" % (city, exc))
            continue
        rows.extend(region_rows)
        best = max(region_rows, key=lambda r: r["delta_auroc"])
        print("  %-12s change px %6d  delta AUROC global %+.4f  histogram %+.4f  "
              "preservation global %.3f  histogram %.3f"
              % (city, region_rows[0]["n_change_pixels"],
                 region_rows[0]["delta_auroc"], region_rows[1]["delta_auroc"],
                 region_rows[0]["preservation"], region_rows[1]["preservation"]))

    if not rows:
        raise SystemExit("no region produced a result")

    fields = ["city", "method", "n_change_pixels", "n_unchanged_pixels", "valid_pixels",
              "pif_pixels", "auroc_before", "auroc_after", "delta_auroc",
              "mean_score_change_before", "mean_score_change_after", "preservation",
              "mean_score_unchanged_before", "mean_score_unchanged_after",
              "contrast_before", "contrast_after",
              "mae_stable_before", "mae_stable_after", "mae_reduction_pct"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})

    summary = {"method": {}, "regions": len({r["city"] for r in rows}),
               "pif_fraction": args.pif_fraction,
               "data": "OSCD rectified bands, digital numbers, no CRS"}
    for method in ("none", "global", "histogram"):
        subset = [r for r in rows if r["method"] == method]
        summary["method"][method] = {
            "n_regions": len(subset),
            "median_delta_auroc": float(np.median([r["delta_auroc"] for r in subset])),
            "mean_delta_auroc": float(np.mean([r["delta_auroc"] for r in subset])),
            "median_preservation": float(np.median([r["preservation"] for r in subset])),
            "min_preservation": float(np.min([r["preservation"] for r in subset])),
            "median_auroc_after": float(np.median([r["auroc_after"] for r in subset])),
            "median_mae_reduction_pct": float(np.median([r["mae_reduction_pct"] for r in subset])),
            "regions_with_lower_auroc": int(sum(1 for r in subset if r["delta_auroc"] < 0)),
        }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nwrote", csv_path)
    print("wrote", json_path)
    for method, values in summary["method"].items():
        print("  %-10s median delta AUROC %+0.4f  median preservation %.3f  "
              "min preservation %.3f  median AUROC after %.4f  median MAE reduction %.1f%%"
              % (method, values["median_delta_auroc"], values["median_preservation"],
                 values["min_preservation"], values["median_auroc_after"],
                 values["median_mae_reduction_pct"]))


if __name__ == "__main__":
    main()
