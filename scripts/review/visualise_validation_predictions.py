"""Validation-only spatial diagnostics for the fine-tuned checkpoint.

Produces, per pair, on the georeferenced validation stripe: a probability raster, a
diagnosis raster (TN/TP/FP/FN with unreviewed as nodata), prediction rasters at 0.50
and at the validation F1-maximising threshold, and a four-panel PNG.

Unreviewed pixels are never painted as "no change": they are nodata in the rasters and
transparent in the overlays.

    python scripts/review/visualise_validation_predictions.py \
        --analysis outputs/evaluation/local_manual_v2/validation_threshold_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from matplotlib import pyplot as plt
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.sessions import PAIRS, build_datasets, load_model, predict_pair
from terrashift.geospatial.rendering import rgb_background

CLASS_COLOURS = {2: (0.10, 0.75, 0.20), 3: (0.95, 0.15, 0.15), 4: (0.15, 0.35, 0.95)}
LEGEND = [("True positive", CLASS_COLOURS[2]), ("False positive", CLASS_COLOURS[3]),
          ("False negative", CLASS_COLOURS[4])]


def transform_for(labels_path, col_offset):
    with rasterio.open(labels_path) as handle:
        base = handle.transform
        crs = handle.crs
        height = handle.height
    return Affine(base.a, base.b, base.c + col_offset * base.a,
                  base.d, base.e, base.f), crs, height


def write_raster(path, array, transform, crs, dtype, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                       width=array.shape[1], count=1, dtype=dtype, crs=crs,
                       transform=transform, nodata=nodata, compress="deflate") as dst:
        dst.write(array.astype(dtype), 1)


def overlay(axis, rgb, diagnosis, title):
    axis.imshow(rgb)
    mask = np.isin(diagnosis, list(CLASS_COLOURS))
    coloured = np.zeros((*diagnosis.shape, 3))
    for value, colour in CLASS_COLOURS.items():
        coloured[diagnosis == value] = colour
    axis.imshow(np.dstack([coloured, mask.astype(float) * 0.75]))
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--fine-tuned",
                        default="outputs/checkpoints/local_manual_v2/seed_42/best.pt")
    parser.add_argument("--normalisation", default="outputs/dataset/v2/normalisation.json")
    parser.add_argument("--labels-dir", default="outputs/labels")
    parser.add_argument("--partition-dir", default="outputs/partitions_v2_observation_valid")
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    parser.add_argument("--registration",
                        default="outputs/audit/registration_v2_observation_valid/registration.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = Path(args.analysis)
    thresholds = json.loads((out / "threshold_selection.json").read_text())["thresholds"]
    metrics = json.loads((out / "validation_metrics.json").read_text())
    datasets, _, _ = build_datasets(args)
    model, _ = load_model(args.fine_tuned, torch.device(args.device))
    raster_dir = out / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for pair in PAIRS:
        entry = datasets[pair]
        acc, scores, labels = predict_pair(model, pair, entry, torch.device(args.device))
        target = entry["target"]
        covered = acc.count > 0
        prob = np.where(covered, acc.total / np.maximum(acc.count, 1), np.nan)
        reviewed = target != IGNORE_VALUE
        transform, crs, _ = transform_for(entry["labels_path"], entry["col_offset"])

        t_global = thresholds["combined"]["threshold"]
        t_pair = thresholds[pair]["threshold"]
        write_raster(raster_dir / ("pair_%s_probability.tif" % pair), prob, transform,
                     crs, "float32", nodata=-9999)

        diagnostics = {}
        for name, threshold in (("thr050", 0.5), ("thr_f1max", t_pair)):
            pred = (prob >= threshold) & reviewed
            diag = np.zeros(target.shape, dtype=np.uint8)     # 0 = unreviewed
            diag[reviewed & (target == 0) & ~pred] = 1        # TN
            diag[reviewed & (target == 1) & pred] = 2         # TP
            diag[reviewed & (target == 0) & pred] = 3         # FP
            diag[reviewed & (target == 1) & ~pred] = 4        # FN
            diagnostics[name] = diag
            write_raster(raster_dir / ("pair_%s_diagnosis_%s.tif" % (pair, name)), diag,
                         transform, crs, "uint8", nodata=0)
            prediction = np.full(target.shape, 255, dtype=np.uint8)
            prediction[reviewed] = pred[reviewed].astype(np.uint8)
            write_raster(raster_dir / ("pair_%s_prediction_%s.tif" % (pair, name)),
                         prediction, transform, crs, "uint8", nodata=255)

        rgb = rgb_background(args, pair, entry, target.shape)
        figure, axes = plt.subplots(2, 2, figsize=(13, 9))
        heat = axes[0, 0].imshow(np.where(reviewed, prob, np.nan), cmap="magma",
                                 vmin=0, vmax=1)
        axes[0, 0].set_title("Predicted probability (unreviewed transparent)", fontsize=9)
        axes[0, 0].axis("off")
        figure.colorbar(heat, ax=axes[0, 0], fraction=0.035)

        truth = np.full(target.shape, np.nan)
        truth[reviewed] = target[reviewed]
        axes[0, 1].imshow(rgb)
        axes[0, 1].imshow(np.ma.masked_invalid(truth), cmap="cool", vmin=0, vmax=1,
                          alpha=0.6)
        axes[0, 1].set_title("Ground truth (reviewed pixels only)", fontsize=9)
        axes[0, 1].axis("off")

        overlay(axes[1, 0], rgb, diagnostics["thr050"],
                "Errors at 0.50")
        overlay(axes[1, 1], rgb, diagnostics["thr_f1max"],
                "Errors at validation F1-max threshold %.2f" % t_pair)
        handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=10,
                              markerfacecolor=colour, label=label, markeredgecolor="k")
                   for label, colour in LEGEND]
        figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
        figure.suptitle("Pair %s — validation windows %d, reviewed pixels %d"
                        % (pair, len(entry["dataset"]), scores.size), fontsize=11)
        figure.tight_layout(rect=(0, 0.04, 1, 0.97))
        figure.savefig(out / ("pair_%s_diagnostics.png" % pair), dpi=140)
        plt.close(figure)

        summary[pair] = {"reviewed_pixels": int(scores.size),
                         "probability_range": [float(np.nanmin(prob)), float(np.nanmax(prob))],
                         "f1max_threshold": t_pair,
                         "f1max_threshold_global": t_global,
                         "tn": int((diagnostics["thr050"] == 1).sum()),
                         "tp": int((diagnostics["thr050"] == 2).sum()),
                         "fp": int((diagnostics["thr050"] == 3).sum()),
                         "fn": int((diagnostics["thr050"] == 4).sum())}
        print("pair", pair, summary[pair])

    (out / "spatial_diagnostics.json").write_text(json.dumps(
        {"pairs": summary,
         "colours": {k: list(v) for k, v in CLASS_COLOURS.items()},
         "legend": "2=TP green, 3=FP red, 4=FN blue, 1=TN and 0=unreviewed not painted",
         "note": "unreviewed pixels are nodata, never shown as confirmed no-change"},
        indent=1) + "\n")
    print("wrote", raster_dir)



if __name__ == "__main__":
    main()
