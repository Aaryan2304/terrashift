"""Resume the spatial validation audit after an interruption.

No model inference, no CUDA, no training. Probability rasters are rebuilt from the saved score vectors
and the reviewed mask, and the rebuild is proven by reproducing every saved diagnosis raster bit for
bit before anything is written. Stages update a manifest, so a later interruption resumes instead of
repeating work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from rasterio.transform import Affine
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.manifest import load_manifest, pair_config
from terrashift.data.safeloader import SafeProduct

CHECKPOINTS = ("oscd", "local_0.834_s43", "local_1.0_s43")
FINE_TUNED = ("local_0.834_s43", "local_1.0_s43")
CHECKPOINT_PATHS = {
    "oscd": "outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt",
    "local_0.834_s43": "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt",
    "local_1.0_s43": "outputs/checkpoints/local_manual_v2_seed43_posweight_1p0/seed_43/best.pt",
}
SCORES = {
    "oscd": "outputs/evaluation/local_manual_v2/validation_threshold_analysis/_oscd_scores.npy",
    "local_0.834_s43": "outputs/evaluation/local_manual_v2_seed43_comparison/_local_0.834_s43_scores.npy",
    "local_1.0_s43": "outputs/evaluation/local_manual_v2_seed43_comparison/_local_1.0_s43_scores.npy",
}
LABELS = "outputs/evaluation/local_manual_v2/validation_threshold_analysis/_fine_tuned_labels.npy"
LABEL_RASTER = {"A": "outputs/labels/pair_a_v2/manual_labels_A.tif",
                "B": "outputs/labels/pair_b_v2/manual_labels_B.tif"}
SPLIT_A = 2759
MAP_THRESHOLDS = (0.05, 0.10, 0.20, 0.50)
PANEL_THRESHOLDS = (0.20, 0.50)
DECIMATE = 12
DIAG_COLOURS = {1: (0.20, 0.20, 0.20), 2: (0.10, 0.75, 0.20),
                3: (0.95, 0.15, 0.15), 4: (0.15, 0.35, 0.95)}
DIAG_NAMES = {1: "TN", 2: "TP", 3: "FP", 4: "FN"}
PAIRS = ("A", "B")
GEOTIFFS = FINE_TUNED   # checkpoints with a written probability raster


def memory_mb():
    info = open("/proc/meminfo").read()
    total = int(info.split("MemTotal:")[1].split()[0]) // 1024
    available = int(info.split("MemAvailable:")[1].split()[0]) // 1024
    return available, total


def guard(stage, limit_mb=1500):
    available, total = memory_mb()
    print("[mem] %-30s available %5d MB / %d MB" % (stage, available, total))
    if available < limit_mb:
        raise SystemExit("stage %s: MemAvailable %d MB below the %d MB floor, stopping"
                         % (stage, available, limit_mb))


def sha256(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def slice_for(pair):
    return slice(0, SPLIT_A) if pair == "A" else slice(SPLIT_A, None)


def diagnosis_path(out, pair, checkpoint, threshold):
    return out / "rasters" / ("pair_%s_%s_diagnosis_%.2f.tif" % (pair, checkpoint, threshold))


def reviewed_and_truth(out, pair):
    """Reviewed mask and reference labels recovered from a saved diagnosis raster.

    TP and FN are change, TN and FP are no-change, code 0 is unreviewed. The mask and
    labels are then cross-checked against all 24 rasters, so a single corrupt file cannot
    pass unnoticed.
    """
    with rasterio.open(diagnosis_path(out, pair, "oscd", 0.50)) as handle:
        base = handle.read(1)
    reviewed = base != 0
    truth = np.zeros(base.shape, dtype=np.int8)
    truth[reviewed] = np.isin(base[reviewed], (2, 4)).astype(np.int8)
    for checkpoint in CHECKPOINTS:
        for threshold in MAP_THRESHOLDS:
            with rasterio.open(diagnosis_path(out, pair, checkpoint, threshold)) as handle:
                other = handle.read(1)
            if not np.array_equal(other != 0, reviewed):
                raise SystemExit("%s %s %.2f: reviewed mask differs between rasters"
                                 % (pair, checkpoint, threshold))
            if not np.array_equal(np.isin(other, (2, 4)).astype(np.int8)[reviewed],
                                 truth[reviewed]):
                raise SystemExit("%s %s %.2f: reference labels differ between rasters"
                                 % (pair, checkpoint, threshold))
    return reviewed, truth


def probability_from_vector(pair, checkpoint, reviewed, rows, cols):
    scores = np.load(SCORES[checkpoint])[slice_for(pair)].astype(np.float32)
    if scores.size != int(reviewed.sum()):
        raise SystemExit("%s %s: vector %d != reviewed %d"
                         % (pair, checkpoint, scores.size, int(reviewed.sum())))
    raster = np.full(reviewed.shape, np.nan, dtype=np.float32)
    raster[rows, cols] = scores
    return raster


def stripe_geometry(out, pair):
    """Stripe column offset and pixel size, read from a written diagnosis raster header."""
    with rasterio.open(diagnosis_path(out, pair, "oscd", 0.50)) as handle:
        written, shape = handle.transform, (handle.height, handle.width)
        crs = handle.crs
    with rasterio.open(LABEL_RASTER[pair]) as handle:
        base = handle.transform
        reference = {"crs": str(handle.crs), "transform": list(handle.transform)[:6],
                     "height": handle.height, "width": handle.width,
                     "bounds": [float(v) for v in handle.bounds]}
    col_offset = int(round((written.c - base.c) / base.a))
    transform = Affine(base.a, base.b, base.c + col_offset * base.a,
                       base.d, base.e, base.f)
    return col_offset, transform, crs, shape, reference


def rgb_pair(pair, manifest, row0, row1, col0, col1, col_offset, out_shape=None):
    cfg = pair_config(manifest, pair)
    planes = []
    for key in ("path_date1", "path_date2"):
        product = SafeProduct(cfg[key])
        window = ((row0, row1), (col0 + col_offset, col1 + col_offset))
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
        planes.append(data)
    return planes[0], planes[1]


def bbox(mask, pad, shape):
    rows, cols = np.nonzero(mask)
    return (max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad + 1, shape[0]),
            max(int(cols.min()) - pad, 0), min(int(cols.max()) + pad + 1, shape[1]))


def save(figure, path):
    figure.savefig(path, dpi=125)
    plt.close(figure)
    print("  wrote", path.name)


def update_manifest(out, stage, payload):
    path = out / "spatial_audit_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"stages": {}}
    manifest["stages"][stage] = payload
    path.write_text(json.dumps(manifest, indent=1) + "\n")


# --------------------------------------------------------------------------- stages

def stage_verify(out, args):
    guard("verify")
    labels = np.load(LABELS)
    checks = {}

    present = sorted(p.name for p in (out / "rasters").glob("*_diagnosis_*.tif"))
    expected = len(PAIRS) * len(CHECKPOINTS) * len(MAP_THRESHOLDS)
    if len(present) != expected:
        raise SystemExit("expected %d diagnosis rasters, found %d" % (expected, len(present)))
    checks["diagnosis_rasters"] = {"count": len(present), "expected": expected}

    rows = list(csv.DictReader(open(out / "components" / "component_metrics.csv")))
    components = sorted({int(r["component"]) for r in rows})
    oscd_rows = [r for r in rows if r["checkpoint"] == "oscd"]
    checks["component_metrics"] = {
        "rows": len(rows),
        "components": components,
        "checkpoints": sorted({r["checkpoint"] for r in rows}),
        "sizes_sum": int(sum(int(r["component_pixels"]) for r in oscd_rows)),
        "positive_pixels_reviewed": int((labels[slice_for("B")] == 1).sum()),
        "rows_per_checkpoint": {c: sum(1 for r in rows if r["checkpoint"] == c)
                                for c in CHECKPOINTS}}
    if len(components) != 9:
        raise SystemExit("expected 9 components, found %d" % len(components))
    if checks["component_metrics"]["sizes_sum"] != checks["component_metrics"]["positive_pixels_reviewed"]:
        raise SystemExit("component sizes %d != reviewed positives %d"
                         % (checks["component_metrics"]["sizes_sum"],
                            checks["component_metrics"]["positive_pixels_reviewed"]))
    weighted = list(csv.DictReader(open(out / "components" / "component_weighted_totals.csv")))
    checks["component_weighted_totals"] = {"rows": len(weighted),
                                           "checkpoints": sorted({r["checkpoint"] for r in weighted}),
                                           "thresholds": sorted({float(r["threshold"]) for r in weighted})}
    if len(weighted) != len(CHECKPOINTS) * len((0.05, 0.10, 0.20, 0.30, 0.50)):
        raise SystemExit("unexpected weighted-total row count %d" % len(weighted))
    for row in weighted:
        if abs(float(row["pixel_weighted_recall_all_change"])
               - float(row["global_recall"])) > 1e-12:
            raise SystemExit("weighted recall %.6f != global recall %.6f"
                             % (float(row["pixel_weighted_recall_all_change"]),
                                float(row["global_recall"])))

    counts = list(csv.DictReader(open(out / "maps" / "confusion_map_counts.csv")))
    mismatched = []
    for row in counts:
        with rasterio.open(diagnosis_path(out, row["pair"], row["checkpoint"],
                                          float(row["threshold"]))) as handle:
            data = handle.read(1)
        for code, key in ((2, "tp"), (3, "fp"), (4, "fn"), (1, "tn"), (0, "unreviewed")):
            if int((data == code).sum()) != int(row[key]):
                mismatched.append("%s/%s/%.2f/%s" % (row["pair"], row["checkpoint"],
                                                     float(row["threshold"]), key))
        del data
    checks["confusion_counts_vs_rasters"] = {
        "rows": len(counts), "mismatch_count": len(mismatched), "mismatches": mismatched[:10]}
    if mismatched:
        raise SystemExit("confusion counts disagree with rasters: %s" % mismatched[:5])

    proof = {}
    for pair in PAIRS:
        reviewed, truth = reviewed_and_truth(out, pair)
        if not np.array_equal(truth[reviewed], labels[slice_for(pair)]):
            raise SystemExit("%s: recovered labels != saved label vector" % pair)
        rows_idx, cols_idx = np.nonzero(reviewed)
        for checkpoint in CHECKPOINTS:
            raster = probability_from_vector(pair, checkpoint, reviewed, rows_idx, cols_idx)
            ok = True
            for threshold in MAP_THRESHOLDS:
                pred = (raster >= threshold) & reviewed
                diag = np.zeros(reviewed.shape, dtype=np.uint8)
                diag[reviewed & (truth == 0) & ~pred] = 1
                diag[reviewed & (truth == 1) & pred] = 2
                diag[reviewed & (truth == 0) & pred] = 3
                diag[reviewed & (truth == 1) & ~pred] = 4
                with rasterio.open(diagnosis_path(out, pair, checkpoint, threshold)) as handle:
                    if not np.array_equal(diag, handle.read(1)):
                        ok = False
                del diag, pred
            proof["%s_%s" % (pair, checkpoint)] = ok
            del raster
        print("[verify] pair %s rebuild proof: %s"
              % (pair, all(v for k, v in proof.items() if k.startswith(pair + "_"))))
        del reviewed, truth, rows_idx, cols_idx
    checks["rebuild_proof"] = proof
    if not all(proof.values()):
        raise SystemExit("rebuild proof failed: %s" % proof)

    (out / "verification.json").write_text(json.dumps(checks, indent=1) + "\n")
    update_manifest(out, "verify", {"status": "complete", "checks": checks})
    print("[verify] all checks passed; score vectors + masks reproduce every raster")


def stage_rasters(out, args):
    guard("rasters")
    alignment = {}
    for pair in PAIRS:
        col_offset, transform, crs, shape, reference = stripe_geometry(out, pair)
        print("[rasters] pair %s stripe offset %d px, shape %s" % (pair, col_offset, shape))
        reviewed, truth = reviewed_and_truth(out, pair)
        if tuple(shape) != reviewed.shape:
            raise SystemExit("pair %s stripe %s != diagnosis shape %s"
                             % (pair, shape, reviewed.shape))
        rows_idx, cols_idx = np.nonzero(reviewed)
        for checkpoint in FINE_TUNED:
            raster = probability_from_vector(pair, checkpoint, reviewed, rows_idx, cols_idx)
            path = out / "rasters" / ("pair_%s_%s_probability.tif" % (pair, checkpoint))
            with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1],
                               count=1, dtype="float32", crs=crs, transform=transform,
                               nodata=-9999, compress="deflate") as dst:
                dst.write(np.where(np.isfinite(raster), raster, -9999).astype(np.float32), 1)
            with rasterio.open(out / "rasters" / ("pair_%s_%s_reviewed_mask.tif"
                                                  % (pair, checkpoint)),
                               "w", driver="GTiff", height=shape[0], width=shape[1],
                               count=1, dtype="uint8", crs=crs, transform=transform,
                               nodata=255, compress="deflate") as dst:
                dst.write(reviewed.astype(np.uint8), 1)
            with rasterio.open(path) as written:
                data = written.read(1)
                valid = data != -9999
                written_t, base_t = list(written.transform)[:6], reference["transform"]
                check = {
                    "crs": str(written.crs),
                    "crs_matches_label_raster": str(written.crs) == reference["crs"],
                    "pixel_size_matches_label_raster":
                        abs(written_t[0] - base_t[0]) < 1e-9
                        and abs(written_t[4] - base_t[4]) < 1e-9,
                    "row_origin_matches_label_raster":
                        abs(written_t[3] - base_t[3]) < 1e-9
                        and abs(written_t[5] - base_t[5]) < 1e-9,
                    "column_offset_matches_stripe":
                        abs(written_t[2] - (base_t[2] + col_offset * base_t[0])) < 1e-6,
                    "height_matches_label_raster": written.height == reference["height"],
                    "width_matches_stripe": written.width == shape[1],
                    "col_offset_pixels": col_offset,
                    "x_origin_shift_m": round(list(written.transform)[2]
                                              - reference["transform"][2], 6),
                    "dtype": written.dtypes[0], "nodata": written.nodata,
                    "valid_pixels": int(valid.sum()),
                    "reviewed_pixels": int(reviewed.sum()),
                    "valid_pixels_equal_reviewed": int(valid.sum()) == int(reviewed.sum()),
                    "probability_range": [float(data[valid].min()), float(data[valid].max())],
                    "values_match_score_vector": bool(np.array_equal(
                        data[valid].astype(np.float32),
                        np.load(SCORES[checkpoint])[slice_for(pair)])),
                    "unreviewed_is_nodata_not_zero": bool(np.isnan(raster[~reviewed]).all()),
                }
            failed = [k for k in ("crs_matches_label_raster",
                                  "pixel_size_matches_label_raster",
                                  "row_origin_matches_label_raster",
                                  "column_offset_matches_stripe",
                                  "height_matches_label_raster",
                                  "width_matches_stripe",
                                  "valid_pixels_equal_reviewed",
                                  "values_match_score_vector",
                                  "unreviewed_is_nodata_not_zero") if not check[k]]
            if failed:
                raise SystemExit("alignment check failed for %s %s: %s"
                                 % (pair, checkpoint, failed))
            alignment["%s_%s" % (pair, checkpoint)] = check
            print("  pair %s %-16s %d valid px, prob %.4f..%.4f, alignment ok"
                  % (pair, checkpoint, check["valid_pixels"],
                     check["probability_range"][0], check["probability_range"][1]))
            del raster, data, valid
        del reviewed, truth, rows_idx, cols_idx
    payload = {"written": alignment,
               "note": ("probability rasters share the label raster grid and CRS, shifted in x "
                        "by the validation stripe offset. Reviewed pixels only; every other "
                        "pixel is nodata, never a negative label.")}
    (out / "rasters" / "geospatial_alignment.json").write_text(json.dumps(payload, indent=1) + "\n")
    update_manifest(out, "rasters", {"status": "complete", "files": len(alignment) * 2,
                                     "alignment": alignment})
    print("[rasters] wrote %d GeoTIFFs" % (len(alignment) * 2))


def stage_figures(out, args):
    guard("figures")
    manifest_yaml = load_manifest(args.manifest)
    made = []
    col_offset, _, _, shape, _ = stripe_geometry(out, "B")

    # ---- Pair B confusion maps (Pair A's survived the restart) --------------
    reviewed, truth = reviewed_and_truth(out, "B")
    row0, row1, col0, col1 = bbox(reviewed, 0, shape)
    out_shape = ((row1 - row0) // DECIMATE, (col1 - col0) // DECIMATE)
    _, after_small = rgb_pair("B", manifest_yaml, row0, row1, col0, col1, col_offset,
                              out_shape=out_shape)
    index_r = (np.arange(out_shape[0]) * DECIMATE).clip(0, row1 - row0 - 1)
    index_c = (np.arange(out_shape[1]) * DECIMATE).clip(0, col1 - col0 - 1)
    figure, axes = plt.subplots(len(CHECKPOINTS), len(MAP_THRESHOLDS),
                                figsize=(4.0 * len(MAP_THRESHOLDS), 3.5 * len(CHECKPOINTS)))
    for r, checkpoint in enumerate(CHECKPOINTS):
        for c, threshold in enumerate(MAP_THRESHOLDS):
            guard("figure map %s %.2f" % (checkpoint, threshold))
            with rasterio.open(diagnosis_path(out, "B", checkpoint, threshold)) as handle:
                diag_full = handle.read(1)
            codes = np.bincount(diag_full.ravel(), minlength=5)
            diag = diag_full[row0:row1, col0:col1][np.ix_(index_r, index_c)]
            del diag_full
            axis = axes[r, c]
            axis.imshow(after_small)
            painted = np.zeros((*diag.shape, 3))
            alpha = np.zeros(diag.shape)
            for value, colour in DIAG_COLOURS.items():
                selected = diag == value
                painted[selected] = colour
                alpha[selected] = 0.70
            axis.imshow(np.dstack([painted, alpha]))
            axis.set_title("%s @ %.2f\nTP %d  FP %d  FN %d"
                           % (checkpoint, threshold, codes[2], codes[3], codes[4]),
                           fontsize=8)
            axis.axis("off")
            del diag, painted, alpha, codes
    handles = [Patch(facecolor=colour, label=DIAG_NAMES[value])
               for value, colour in DIAG_COLOURS.items()]
    handles.append(Patch(facecolor=(1, 1, 1, 0), edgecolor="k",
                         label="not reviewed (transparent)"))
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    figure.suptitle("Pair B validation confusion maps — counts at full resolution, "
                    "display decimated %dx, unreviewed transparent" % DECIMATE, fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(figure, out / "maps" / "pair_B_confusion_maps.png")
    made.append("maps/pair_B_confusion_maps.png")
    del after_small

    # ---- components ---------------------------------------------------------
    comp, n_comp = ndi.label(truth == 1, structure=np.ones((3, 3)))
    # truth == 1 already excludes unreviewed (truth is 0 there), so components are exact
    sizes = {int(i): int((comp == i).sum()) for i in range(1, n_comp + 1)}
    order = sorted(sizes, key=lambda i: -sizes[i])
    print("[figures] %d components, sizes desc %s, sum %d"
          % (n_comp, [sizes[i] for i in order], sum(sizes.values())))
    if n_comp != 9 or sum(sizes.values()) != 11940:
        raise SystemExit("component recount differs: %d components, %d pixels"
                         % (n_comp, sum(sizes.values())))

    oscd_cache = {}

    def panel(name, pair, row0, row1, col0, col1, title):
        guard("figure %s" % name)
        before, after = rgb_pair(pair, manifest_yaml, row0, row1, col0, col1, col_offset)
        truth_crop = truth[row0:row1, col0:col1].astype(float)
        truth_crop[~reviewed[row0:row1, col0:col1]] = np.nan
        columns = 3 + 2 * len(CHECKPOINTS)
        figure, axes = plt.subplots(len(PANEL_THRESHOLDS), columns,
                                    figsize=(2.85 * columns, 3.1 * len(PANEL_THRESHOLDS)),
                                    squeeze=False)
        rasters = {}
        for checkpoint in CHECKPOINTS:
            if checkpoint in GEOTIFFS:
                with rasterio.open(out / "rasters" / ("pair_%s_%s_probability.tif"
                                                      % (pair, checkpoint))) as handle:
                    block = handle.read(1, window=((row0, row1), (col0, col1)))
                rasters[checkpoint] = np.where(block == -9999, np.nan, block)
                del block
            else:
                # OSCD has no GeoTIFF by design (Task 3 asks for the fine-tuned pair only),
                # so one in-memory raster is built per pair and cropped for each panel.
                if pair not in oscd_cache:
                    rows_idx, cols_idx = np.nonzero(reviewed)
                    oscd_cache[pair] = probability_from_vector(pair, checkpoint, reviewed,
                                                               rows_idx, cols_idx)
                    del rows_idx, cols_idx
                    print("  [oscd] built in-memory probability for pair %s" % pair)
                rasters[checkpoint] = oscd_cache[pair][row0:row1, col0:col1]
        for t, threshold in enumerate(PANEL_THRESHOLDS):
            axes[t, 0].imshow(before); axes[t, 0].set_title("before (date 1)", fontsize=8)
            axes[t, 1].imshow(after); axes[t, 1].set_title("after (date 2)", fontsize=8)
            axes[t, 2].imshow(after)
            axes[t, 2].imshow(np.ma.masked_invalid(truth_crop), cmap="cool", vmin=0, vmax=1,
                              alpha=0.55)
            axes[t, 2].set_title("reference annotation", fontsize=8)
            for c, checkpoint in enumerate(CHECKPOINTS):
                axes[t, 3 + 2 * c].imshow(rasters[checkpoint], cmap="magma", vmin=0, vmax=1)
                axes[t, 3 + 2 * c].set_title("%s probability" % checkpoint, fontsize=7)
                binary = np.where(np.isnan(rasters[checkpoint]), np.nan,
                                  (rasters[checkpoint] >= threshold).astype(float))
                axes[t, 4 + 2 * c].imshow(after)
                axes[t, 4 + 2 * c].imshow(np.ma.masked_invalid(binary), cmap="autumn",
                                          vmin=0, vmax=1, alpha=0.55)
                axes[t, 4 + 2 * c].set_title("%s @ %.2f" % (checkpoint, threshold), fontsize=7)
            for axis in axes[t]:
                axis.axis("off")
        figure.suptitle(title + " — top row threshold 0.20, bottom 0.50; unreviewed "
                        "transparent", fontsize=10)
        figure.tight_layout()
        save(figure, out / "figures" / ("%s.png" % name))
        made.append("figures/%s.png" % name)
        del rasters, before, after, truth_crop
        guard("after %s" % name)

    row0, row1, col0, col1 = bbox(comp == order[0], 48, shape)
    panel("pair_B_largest_component", "B", row0, row1, col0, col1,
          "Pair B largest change component (id %d, %d px)" % (order[0], sizes[order[0]]))
    for index in order[1:]:
        row0, row1, col0, col1 = bbox(comp == index, 40, shape)
        panel("pair_B_component_%02d" % index, "B", row0, row1, col0, col1,
              "Pair B component %d (%d px)" % (index, sizes[index]))
    del comp

    # ---- representative false-positive areas -------------------------------
    fp_summary = {}
    for checkpoint in FINE_TUNED:
        guard("fp areas %s" % checkpoint)
        with rasterio.open(out / "rasters" / ("pair_B_%s_probability.tif"
                                              % checkpoint)) as handle:
            block = handle.read(1)
        prob = np.where(block == -9999, np.nan, block)
        del block
        fp = (prob >= 0.5) & reviewed & (truth == 0)
        blobs, n_blobs = ndi.label(fp, structure=np.ones((3, 3)))
        blob_sizes = np.bincount(blobs.ravel())[1:] if n_blobs else np.array([], dtype=int)
        fp_summary[checkpoint] = {"blobs": int(n_blobs),
                                  "largest": [int(x) for x in np.sort(blob_sizes)[::-1][:5]]}
        print("[figures] %s: %d FP blobs at 0.50, largest %s"
              % (checkpoint, n_blobs, fp_summary[checkpoint]["largest"]))
        del prob, fp
        for rank in range(min(3, n_blobs)):
            blob = int(np.argsort(-blob_sizes)[rank]) + 1
            row0, row1, col0, col1 = bbox(blobs == blob, 40, shape)
            panel("pair_B_%s_false_positive_%d" % (checkpoint, rank + 1), "B",
                  row0, row1, col0, col1,
                  "Pair B false-positive area %d for %s (%d px at 0.50)"
                  % (rank + 1, checkpoint, int((blobs == blob).sum())))
        del blobs, blob_sizes

    # ---- Pair A comparison --------------------------------------------------
    reviewed_a, truth_a = reviewed_and_truth(out, "A")
    row0, row1, col0, col1 = bbox(truth_a == 1, 32, shape)
    saved_reviewed, saved_truth = reviewed, truth
    reviewed, truth = reviewed_a, truth_a
    panel("pair_A_comparison", "A", row0, row1, col0, col1,
          "Pair A change region (%d reviewed change px)" % int(truth_a.sum()))
    reviewed, truth = saved_reviewed, saved_truth
    del reviewed_a, truth_a

    oscd_cache.clear()
    update_manifest(out, "figures", {"status": "complete", "files": made,
                                     "false_positive_blobs": fp_summary})
    print("[figures] %d figures written" % len(made))


def stage_tables(out, args):
    guard("tables")
    labels = np.load(LABELS)
    rows = []
    for pair in PAIRS:
        truth = labels[slice_for(pair)]
        change = truth == 1
        for checkpoint in CHECKPOINTS:
            scores = np.load(SCORES[checkpoint])[slice_for(pair)].astype(np.float32)
            for threshold in MAP_THRESHOLDS + (0.30,):
                pred = scores >= threshold
                tp = int((pred & change).sum()); fp = int((pred & ~change).sum())
                fn = int((~pred & change).sum()); tn = int((~pred & ~change).sum())
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                rows.append({"pair": pair, "checkpoint": checkpoint, "threshold": threshold,
                             "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "precision": precision, "recall": recall,
                             "f1": 2 * precision * recall / (precision + recall)
                             if precision + recall else 0.0,
                             "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
                             "predicted_positive_fraction": (tp + fp) / truth.size})
    with open(out / "threshold_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    available, total = memory_mb()
    summary = {
        "checkpoints": {name: {"sha256": sha256(path), "scores": SCORES[name],
                               "path": path}
                        for name, path in CHECKPOINT_PATHS.items()},
        "counts": {"reviewed_A": int(labels[:SPLIT_A].size),
                   "reviewed_B": int(labels[SPLIT_A:].size),
                   "positives_A": int((labels[:SPLIT_A] == 1).sum()),
                   "positives_B": int((labels[SPLIT_A:] == 1).sum())},
        "threshold_metrics_rows": len(rows),
        "no_inference_rerun": True,
        "device": "cpu",
        "test_split_accessed": False,
        "threshold_selected": False,
        "mem_available_mb_at_tables": available,
        "mem_total_mb": total,
    }
    (out / "spatial_audit.json").write_text(json.dumps(summary, indent=1) + "\n")
    update_manifest(out, "tables", {"status": "complete",
                                    "files": ["threshold_metrics.csv", "spatial_audit.json"]})
    print("[tables] wrote threshold_metrics.csv (%d rows) and spatial_audit.json" % len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", default="all",
                        choices=["verify", "rasters", "figures", "tables", "all"])
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_dir():
        raise SystemExit("output directory does not exist: %s" % out)
    stages = {"verify": stage_verify, "rasters": stage_rasters,
              "figures": stage_figures, "tables": stage_tables}
    order = ["verify", "rasters", "figures", "tables"] if args.stage == "all" else [args.stage]
    for name in order:
        print("=== stage %s ===" % name)
        stages[name](out, args)
    print("stages complete:", order)


if __name__ == "__main__":
    main()
