"""Spatial validation audit for three checkpoints.

Read-only: nothing is trained, no checkpoint is written, and normalisation, data, labels and existing
evaluation output are untouched. Only the validation split is used.

The three checkpoints are ``oscd``, ``local_0.834_s43`` and ``local_1.0_s43``, named after the
positive weights used in the paired-seed report. Saved 1-D score vectors drive every metric, and the
rasters come from the same canonical lattice pass and are checked against those vectors pixel for
pixel, so a raster and its vector cannot drift apart.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from rasterio.transform import Affine
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.manifest import load_manifest, pair_config
from terrashift.data.safeloader import SafeProduct
from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.sessions import PAIRS, build_datasets, load_model, predict_pair

CHECKPOINTS = {
    "oscd": ("outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt",
             "outputs/evaluation/local_manual_v2/validation_threshold_analysis/_oscd_scores.npy"),
    "local_0.834_s43": ("outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt",
                        "outputs/evaluation/local_manual_v2_seed43_comparison/_local_0.834_s43_scores.npy"),
    "local_1.0_s43": ("outputs/checkpoints/local_manual_v2_seed43_posweight_1p0/seed_43/best.pt",
                      "outputs/evaluation/local_manual_v2_seed43_comparison/_local_1.0_s43_scores.npy"),
}
FINE_TUNED = ("local_0.834_s43", "local_1.0_s43")
SPLIT_A = 2759
COMPONENT_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50)
MAP_THRESHOLDS = (0.05, 0.10, 0.20, 0.50)
EXPECTED_COMPONENT_SIZES = [154, 121, 116, 97, 91, 48, 140, 10749, 424]

# one colour scheme everywhere; unreviewed is never painted
DIAG_COLOURS = {1: (0.20, 0.20, 0.20),   # TN
                2: (0.10, 0.75, 0.20),   # TP
                3: (0.95, 0.15, 0.15),   # FP
                4: (0.15, 0.35, 0.95)}   # FN
DIAG_NAMES = {1: "TN", 2: "TP", 3: "FP", 4: "FN"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform_for(labels_path, col_offset):
    with rasterio.open(labels_path) as handle:
        base, crs = handle.transform, handle.crs
        height, width = handle.height, handle.width
    return Affine(base.a, base.b, base.c + col_offset * base.a,
                  base.d, base.e, base.f), crs, height, width


def write_raster(path, array, transform, crs, dtype, nodata=None, compress="deflate"):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                       width=array.shape[1], count=1, dtype=dtype, crs=crs,
                       transform=transform, nodata=nodata, compress=compress) as dst:
        dst.write(array.astype(dtype), 1)


def rgb_crop(args, pair, row0, row1, col0, col1, col_offset):
    """True colour for one date over a window, percentile-stretched on that window."""
    cfg = pair_config(load_manifest(args.manifest), pair)
    planes = []
    for key in ("path_date1", "path_date2"):
        product = SafeProduct(cfg[key])
        window = ((row0, row1), (col0 + col_offset, col1 + col_offset))
        bands = [product.read_band(band, window=window)[0] for band in ("B04", "B03", "B02")]
        rgb = np.stack(bands, axis=-1).astype(np.float32)
        out = np.zeros_like(rgb)
        for channel in range(3):
            plane = rgb[..., channel]
            finite = np.isfinite(plane)
            if not finite.any():
                continue
            low, high = np.percentile(plane[finite], [2, 98])
            out[..., channel] = np.clip((plane - low) / max(high - low, 1e-6), 0, 1)
        planes.append(out)
    return planes[0], planes[1]


def classification(scores, labels, threshold):
    pred = scores >= threshold
    truth = labels == 1
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "predicted_positive_fraction": (tp + fp) / labels.size}


def bbox(mask, pad, shape):
    rows, cols = np.nonzero(mask)
    return (max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad + 1, shape[0]),
            max(int(cols.min()) - pad, 0), min(int(cols.max()) + pad + 1, shape[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-vector",
                        default="outputs/evaluation/local_manual_v2/validation_threshold_analysis/_fine_tuned_labels.npy")
    parser.add_argument("--normalisation", default="outputs/dataset/v2/normalisation.json")
    parser.add_argument("--labels-dir", default="outputs/labels")
    parser.add_argument("--partition-dir", default="outputs/partitions_v2_observation_valid")
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    parser.add_argument("--registration",
                        default="outputs/audit/registration_v2_observation_valid/registration.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing to overwrite %s" % out)
    for sub in ("rasters", "maps", "figures", "components"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    labels_vector = np.load(args.labels_vector)
    if labels_vector.size != SPLIT_A + 22650:
        raise SystemExit("unexpected label vector length %d" % labels_vector.size)

    datasets, _, _ = build_datasets(args)
    hashes = {name: sha256(path) for name, (path, _) in CHECKPOINTS.items()}
    hashes["normalisation"] = sha256(args.normalisation)

    # ---- one canonical pass per checkpoint, checked against the saved vectors ----
    probability, diagnosis_source = {}, {}
    consistency = {}
    for name, (checkpoint, vector_path) in CHECKPOINTS.items():
        saved = np.load(vector_path)
        model, state = load_model(checkpoint, device)
        probability[name] = {}
        consistency[name] = {"epoch": int(state.get("epoch", -1)),
                             "pos_weight": state.get("pos_weight")}
        for pair in PAIRS:
            acc, scores, labels = predict_pair(model, pair, datasets[pair], device)
            target = datasets[pair]["target"]
            covered = (acc.count > 0) & (target != IGNORE_VALUE)
            raster = np.where(acc.count > 0, acc.total / np.maximum(acc.count, 1), np.nan)
            from_raster = raster[covered].astype(np.float32)
            expected = saved[:SPLIT_A] if pair == "A" else saved[SPLIT_A:]
            if not np.array_equal(labels, (target[covered]).astype(np.int64)):
                raise SystemExit("%s pair %s: label vector mismatch" % (name, pair))
            if from_raster.size != expected.size:
                raise SystemExit("%s pair %s: raster has %d pixels, vector %d"
                                 % (name, pair, from_raster.size, expected.size))
            delta = float(np.max(np.abs(from_raster - expected)))
            if delta > 1e-6:
                raise SystemExit("%s pair %s: raster and saved vector differ by %g"
                                 % (name, pair, delta))
            consistency[name][pair] = {"reviewed_pixels": int(from_raster.size),
                                       "max_abs_delta_vs_saved_vector": delta,
                                       "raster_matches_saved_vector": True}
            probability[name][pair] = raster
            print("%-16s pair %s  raster==vector (%d px, delta %.2e)"
                  % (name, pair, from_raster.size, delta))
        del model
        torch.cuda.empty_cache()

    labels_by_pair = {"A": labels_vector[:SPLIT_A], "B": labels_vector[SPLIT_A:]}

    # ---- TASK 1: Pair B connected components -------------------------------
    entry = datasets["B"]
    targetB = entry["target"]
    reviewedB = targetB != IGNORE_VALUE
    change = reviewedB & (targetB == 1)
    comp, n_comp = ndi.label(change, structure=np.ones((3, 3)))
    sizes = {int(i): int((comp == i).sum()) for i in range(1, n_comp + 1)}
    total_change = int(sum(sizes.values()))
    if total_change != labels_by_pair["B"].size - int((labels_by_pair["B"] == 0).sum()):
        raise SystemExit("component pixels %d do not match the reviewed change count"
                         % total_change)
    if total_change != int((labels_by_pair["B"] == 1).sum()):
        raise SystemExit("component pixels %d != positives %d"
                         % (total_change, int((labels_by_pair["B"] == 1).sum())))
    if sorted(sizes.values()) != sorted(EXPECTED_COMPONENT_SIZES):
        raise SystemExit("component sizes %s differ from the audited set %s"
                         % (sorted(sizes.values()), sorted(EXPECTED_COMPONENT_SIZES)))

    rows_by_pixel = np.nonzero(reviewedB)[0], np.nonzero(reviewedB)[1]
    component_of_pixel = comp[rows_by_pixel].astype(np.int16)   # 0 = reviewed no-change
    order = np.argsort(-np.array([sizes[i] for i in sizes]))

    component_rows = []
    for checkpoint in CHECKPOINTS:
        scores = probability[checkpoint]["B"][reviewedB].astype(np.float32)
        if not np.array_equal(scores, np.load(CHECKPOINTS[checkpoint][1])[SPLIT_A:]):
            raise SystemExit("%s Pair B vector differs from the saved scores" % checkpoint)
        for index in sizes:
            values = scores[component_of_pixel == index]
            row = {"checkpoint": checkpoint, "component": index,
                   "component_pixels": int(values.size),
                   "rank_by_size": int(list(order).index(index - 1) + 1),
                   "mean_score": float(values.mean()),
                   "median_score": float(np.median(values)),
                   "std_score": float(values.std()),
                   "min_score": float(values.min()),
                   "max_score": float(values.max())}
            for threshold in COMPONENT_THRESHOLDS:
                row["recall_%.2f" % threshold] = float((values >= threshold).mean())
                row["fraction_flagged_%.2f" % threshold] = float((values >= threshold).mean())
            component_rows.append(row)
    with open(out / "components" / "component_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)

    # pixel-weighted totals, computed independently of any averaging of recalls
    weighted = {}
    for checkpoint in CHECKPOINTS:
        scores = probability[checkpoint]["B"][reviewedB].astype(np.float32)
        labels = labels_by_pair["B"]
        weighted[checkpoint] = {}
        for threshold in COMPONENT_THRESHOLDS:
            global_row = classification(scores, labels, threshold)
            change_hits = int((scores[labels == 1] >= threshold).sum())
            component_hits = sum(int((scores[component_of_pixel == i] >= threshold).sum())
                                 for i in sizes)
            if change_hits != component_hits:
                raise SystemExit("%s: component hits %d != change hits %d at %.2f"
                                 % (checkpoint, component_hits, change_hits, threshold))
            weighted[checkpoint][threshold] = {
                "pixel_weighted_recall_all_change": change_hits / int((labels == 1).sum()),
                "component_pixels": total_change,
                "components_fully_recalled": sum(
                    1 for i in sizes
                    if (scores[component_of_pixel == i] >= threshold).all()),
                "global_precision": global_row["precision"],
                "global_recall": global_row["recall"],
                "predicted_positive_fraction": global_row["predicted_positive_fraction"]}
    with open(out / "components" / "component_weighted_totals.csv", "w", newline="") as handle:
        fields = ["checkpoint", "threshold", "pixel_weighted_recall_all_change",
                  "component_pixels", "components_fully_recalled", "global_precision",
                  "global_recall", "predicted_positive_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for checkpoint, per_threshold in weighted.items():
            for threshold, row in per_threshold.items():
                writer.writerow(dict(checkpoint=checkpoint, threshold=threshold, **row))

    # ---- TASK 2: spatial confusion maps ------------------------------------
    confusion_rows = []
    for pair in PAIRS:
        target = datasets[pair]["target"]
        reviewed = target != IGNORE_VALUE
        for checkpoint in CHECKPOINTS:
            raster = probability[checkpoint][pair]
            for threshold in MAP_THRESHOLDS:
                pred = (raster >= threshold) & reviewed
                diag = np.zeros(target.shape, dtype=np.uint8)
                diag[reviewed & (target == 0) & ~pred] = 1
                diag[reviewed & (target == 1) & pred] = 2
                diag[reviewed & (target == 0) & pred] = 3
                diag[reviewed & (target == 1) & ~pred] = 4
                transform, crs, _, _ = transform_for(datasets[pair]["labels_path"],
                                                    datasets[pair]["col_offset"])
                write_raster(out / "rasters" / ("pair_%s_%s_diagnosis_%.2f.tif"
                                                % (pair, checkpoint, threshold)),
                             diag, transform, crs, "uint8", nodata=0)
                confusion_rows.append({
                    "pair": pair, "checkpoint": checkpoint, "threshold": threshold,
                    "tp": int((diag == 2).sum()), "fp": int((diag == 3).sum()),
                    "fn": int((diag == 4).sum()), "tn": int((diag == 1).sum()),
                    "unreviewed": int((diag == 0).sum()),
                    "reviewed": int(reviewed.sum())})
    with open(out / "maps" / "confusion_map_counts.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confusion_rows[0]))
        writer.writeheader()
        writer.writerows(confusion_rows)

    # one figure per pair: rows = checkpoints, columns = thresholds
    for pair in PAIRS:
        target = datasets[pair]["target"]
        reviewed = target != IGNORE_VALUE
        row0, row1, col0, col1 = bbox(reviewed, 0, target.shape)
        before, after = rgb_crop(args, pair, row0, row1, col0, col1,
                                datasets[pair]["col_offset"])
        figure, axes = plt.subplots(len(CHECKPOINTS), len(MAP_THRESHOLDS),
                                    figsize=(4.0 * len(MAP_THRESHOLDS), 3.6 * len(CHECKPOINTS)))
        for r, checkpoint in enumerate(CHECKPOINTS):
            raster = probability[checkpoint][pair]
            for c, threshold in enumerate(MAP_THRESHOLDS):
                axis = axes[r, c]
                axis.imshow(after[row0:row1, col0:col1])
                pred = (raster >= threshold) & reviewed
                diag = np.zeros(target.shape, dtype=np.uint8)
                diag[reviewed & (target == 0) & ~pred] = 1
                diag[reviewed & (target == 1) & pred] = 2
                diag[reviewed & (target == 0) & pred] = 3
                diag[reviewed & (target == 1) & ~pred] = 4
                crop = diag[row0:row1, col0:col1]
                painted = np.zeros((*crop.shape, 3))
                alpha = np.zeros(crop.shape)
                for value, colour in DIAG_COLOURS.items():
                    selected = crop == value
                    painted[selected] = colour
                    alpha[selected] = 0.70
                axis.imshow(np.dstack([painted, alpha]))
                counts = {DIAG_NAMES[v]: int((diag == v).sum()) for v in DIAG_NAMES}
                axis.set_title("%s @ %.2f\nTP %d  FP %d  FN %d"
                               % (checkpoint, threshold, counts["TP"], counts["FP"],
                                  counts["FN"]), fontsize=8)
                axis.axis("off")
        handles = [Patch(facecolor=colour, label=DIAG_NAMES[value])
                   for value, colour in DIAG_COLOURS.items()]
        handles.append(Patch(facecolor=(1, 1, 1, 0), edgecolor="k",
                             label="not reviewed (transparent)"))
        figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
        figure.suptitle("Pair %s validation confusion maps — reviewed pixels only, "
                        "unreviewed transparent" % pair, fontsize=11)
        figure.tight_layout(rect=(0, 0.05, 1, 0.97))
        figure.savefig(out / "maps" / ("pair_%s_confusion_maps.png" % pair), dpi=130)
        plt.close(figure)

    # ---- TASK 3: georeferenced probability rasters -------------------------
    alignment, references = {}, {}
    for pair in PAIRS:
        labels_path = datasets[pair]["labels_path"]
        with rasterio.open(labels_path) as handle:
            reference = {"crs": str(handle.crs), "transform": list(handle.transform)[:6],
                         "height": handle.height, "width": handle.width,
                         "bounds": [float(v) for v in handle.bounds]}
        references[pair] = reference
        transform, crs, height, width = transform_for(labels_path, datasets[pair]["col_offset"])
        if (height, width) != datasets[pair]["target"].shape:
            raise SystemExit("pair %s raster shape %s != dataset shape %s"
                             % (pair, (height, width), datasets[pair]["target"].shape))
        for checkpoint in FINE_TUNED:
            raster = probability[checkpoint][pair].astype(np.float32)
            path = out / "rasters" / ("pair_%s_%s_probability.tif" % (pair, checkpoint))
            write_raster(path, np.where(np.isfinite(raster), raster, -9999), transform,
                         crs, "float32", nodata=-9999)
            write_raster(out / "rasters" / ("pair_%s_%s_reviewed_mask.tif" % (pair, checkpoint)),
                         (datasets[pair]["target"] != IGNORE_VALUE).astype(np.uint8),
                         transform, crs, "uint8", nodata=255)
            with rasterio.open(path) as written:
                check = {"crs_matches_labels": str(written.crs) == reference["crs"],
                         "transform_matches_labels": all(
                             abs(a - b) < 1e-9 for a, b in
                             zip(list(written.transform)[:6], reference["transform"])),
                         "shape_matches_labels": written.height == reference["height"]
                         and written.width == reference["width"],
                         "nodata": written.nodata, "dtype": written.dtypes[0],
                         "count": written.count}
                data = written.read(1)
                check["valid_pixels"] = int((data != -9999).sum())
                check["reviewed_pixels"] = int((datasets[pair]["target"] != IGNORE_VALUE).sum())
                check["probability_range"] = [float(data[data != -9999].min()),
                                              float(data[data != -9999].max())]
                check["labels_window_origin_matches"] = abs(
                    reference["transform"][2] - list(written.transform)[2]
                    - datasets[pair]["col_offset"] * reference["transform"][0]) < 1e-9
            alignment["%s_%s" % (pair, checkpoint)] = check
    (out / "rasters" / "geospatial_alignment.json").write_text(
        json.dumps({"labels_reference": references, "written": alignment,
                    "note": ("written rasters share the label raster grid, shifted in x by "
                             "the validation stripe column offset")}, indent=1) + "\n")

    # ---- TASK 4: visual comparison -----------------------------------------
    biggest = max(sizes, key=lambda i: sizes[i])
    figure_count = 0

    def panel(name, pair, row0, row1, col0, col1, thresholds=(0.20, 0.50), title=""):
        nonlocal figure_count
        target = datasets[pair]["target"]
        reviewed = target != IGNORE_VALUE
        before, after = rgb_crop(args, pair, row0, row1, col0, col1,
                                datasets[pair]["col_offset"])
        truth = np.full(target.shape, np.nan)
        truth[reviewed] = target[reviewed]
        columns = 2 + 2 * len(CHECKPOINTS) + 1
        figure, axes = plt.subplots(2, columns, figsize=(2.9 * columns, 6.4),
                                    squeeze=False)
        for t, (threshold_row, label) in enumerate(zip(thresholds,
                                                       ["threshold %.2f" % x for x in thresholds])):
            axes[t, 0].imshow(before[row0:row1, col0:col1])
            axes[t, 0].set_title("before (date 1)", fontsize=8)
            axes[t, 1].imshow(after[row0:row1, col0:col1])
            axes[t, 1].set_title("after (date 2)", fontsize=8)
            axes[t, 2].imshow(after[row0:row1, col0:col1])
            axes[t, 2].imshow(np.ma.masked_invalid(truth[row0:row1, col0:col1]),
                              cmap="cool", vmin=0, vmax=1, alpha=0.55)
            axes[t, 2].set_title("reference annotation", fontsize=8)
            for c, checkpoint in enumerate(CHECKPOINTS):
                raster = probability[checkpoint][pair]
                axes[t, 3 + 2 * c].imshow(raster[row0:row1, col0:col1], cmap="magma",
                                          vmin=0, vmax=1)
                axes[t, 3 + 2 * c].set_title("%s probability" % checkpoint, fontsize=7)
                binary = (raster >= threshold_row).astype(float)
                binary[~reviewed] = np.nan
                axes[t, 4 + 2 * c].imshow(after[row0:row1, col0:col1])
                axes[t, 4 + 2 * c].imshow(np.ma.masked_invalid(binary[row0:row1, col0:col1]),
                                          cmap="autumn", vmin=0, vmax=1, alpha=0.55)
                axes[t, 4 + 2 * c].set_title("%s @ %.2f" % (checkpoint, threshold_row),
                                             fontsize=7)
            for axis in axes[t]:
                axis.axis("off")
        figure.suptitle(title, fontsize=10)
        figure.tight_layout()
        figure.savefig(out / "figures" / ("%s.png" % name), dpi=125)
        plt.close(figure)
        figure_count += 1

    if True:
        row0, row1, col0, col1 = bbox(comp == biggest, 48, targetB.shape)
        panel("pair_B_largest_component", "B", row0, row1, col0, col1,
              title="Pair B largest change component (%d px) — component %d"
                    % (sizes[biggest], biggest))
    for index in [i for i in order if sizes[i] != sizes[biggest]][:8]:
        row0, row1, col0, col1 = bbox(comp == index, 40, targetB.shape)
        panel("pair_B_component_%02d" % index, "B", row0, row1, col0, col1,
              title="Pair B component %d (%d px)" % (index, sizes[index]))

    # representative false-positive areas at 0.50, one figure per fine-tuned checkpoint
    for checkpoint in FINE_TUNED:
        raster = probability[checkpoint]["B"]
        pred = (raster >= 0.5) & reviewedB
        fp = pred & (targetB == 0)
        blobs, n_blobs = ndi.label(fp, structure=np.ones((3, 3)))
        blob_sizes = ndi.sum(fp, blobs, index=range(1, n_blobs + 1)) if n_blobs else []
        top = np.argsort(-np.asarray(blob_sizes))[:3] + 1 if n_blobs else []
        for rank, blob in enumerate(top):
            row0, row1, col0, col1 = bbox(blobs == blob, 40, targetB.shape)
            panel("pair_B_%s_false_positive_%d" % (checkpoint, rank + 1), "B",
                  row0, row1, col0, col1,
                  title="Pair B false-positive area %d for %s (%d px at 0.50)"
                        % (rank + 1, checkpoint, int((blobs == blob).sum())))
        print("%s: %d FP blobs, top sizes %s"
              % (checkpoint, n_blobs,
                 [int(x) for x in np.sort(blob_sizes)[::-1][:5]] if n_blobs else []))

    targetA = datasets["A"]["target"]
    reviewedA = targetA != IGNORE_VALUE
    row0, row1, col0, col1 = bbox(reviewedA & (targetA == 1), 32, targetA.shape)
    panel("pair_A_comparison", "A", row0, row1, col0, col1,
          title="Pair A change region — same panels, reviewed pixels only")

    # ---- per-pair threshold metrics ---------------------------------------
    threshold_rows = []
    for pair in PAIRS:
        for checkpoint in CHECKPOINTS:
            scores = probability[checkpoint][pair][datasets[pair]["target"] != IGNORE_VALUE]
            for threshold in MAP_THRESHOLDS + (0.30,):
                row = classification(scores.astype(np.float32), labels_by_pair[pair], threshold)
                row.update({"pair": pair, "checkpoint": checkpoint})
                threshold_rows.append(row)
    with open(out / "threshold_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)

    seed43 = {}
    for pair in PAIRS:
        scores_a = probability["local_0.834_s43"][pair][datasets[pair]["target"] != IGNORE_VALUE]
        scores_b = probability["local_1.0_s43"][pair][datasets[pair]["target"] != IGNORE_VALUE]
        labels = labels_by_pair[pair]
        for threshold in MAP_THRESHOLDS:
            row_a = classification(scores_a.astype(np.float32), labels, threshold)
            row_b = classification(scores_b.astype(np.float32), labels, threshold)
            seed43["%s_%.2f" % (pair, threshold)] = {
                "precision_0.834": row_a["precision"], "precision_1.0": row_b["precision"],
                "recall_0.834": row_a["recall"], "recall_1.0": row_b["recall"],
                "tp_0.834": row_a["tp"], "tp_1.0": row_b["tp"],
                "fp_0.834": row_a["fp"], "fp_1.0": row_b["fp"]}
        agree = float(((scores_a >= 0.5) == (scores_b >= 0.5)).mean())
        seed43["%s_pixel_agreement_at_0.50" % pair] = agree

    payload = {
        "checkpoints": {name: {"path": path, "sha256": hashes[name],
                               "saved_vector": CHECKPOINTS[name][1]}
                        for name, (path, _) in CHECKPOINTS.items()},
        "normalisation_sha256": hashes["normalisation"],
        "raster_vector_consistency": consistency,
        "components": {"count": n_comp, "sizes": sizes,
                       "total_change_pixels": total_change,
                       "matches_audited_sizes": sorted(sizes.values()) == sorted(EXPECTED_COMPONENT_SIZES)},
        "weighted_totals": weighted,
        "geospatial_alignment": alignment,
        "seed43_comparison": seed43,
        "figures": figure_count,
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "numpy": np.__version__, "rasterio": rasterio.__version__},
        "test_split_accessed": False,
        "threshold_selected": False,
    }
    (out / "spatial_audit.json").write_text(json.dumps(payload, indent=1) + "\n")
    print("components:", n_comp, sizes, "sum", total_change)
    print("wrote", out)


if __name__ == "__main__":
    main()
