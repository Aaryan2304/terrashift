"""Build the approved local training dataset, one step at a time.

Subcommands, in order: ``normalise`` (per-pair per-band statistics from valid TRAIN pixels), ``index``
(256/128 windows, minimum reviewed fraction 0.002), ``materialise`` (write the selected patches) and
``verify`` (re-read and check).

Test labels are never read: the statistics skip test columns, the indexer skips the test split, and
the materialiser writes only rows present in the index.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.local_patches import (BANDS, build_local_patch_index, pair_image_loader,
                                write_local_patch_index)
from terrashift.data.manifest import load_manifest, pair_config
from terrashift.preprocessing.normalisation import (compute_pair_statistics, compute_window_statistics, load_statistics,
                                normalise_reflectance, pair_statistics,
                                save_statistics)
from terrashift.data.safeloader import SafeProduct

PATCH = 256
STRIDE = 128
TRAIN_FRACTION = 0.002
PAIRS = ("A", "B")


def paths(args, pair):
    return {
        "labels": Path(args.labels_dir) / ("pair_%s_v2" % pair.lower())
                  / ("manual_labels_%s.tif" % pair),
        "partition": Path(args.partition_dir) / ("pair_%s_partition_10m.tif" % pair),
        "validity": Path(args.validity_dir) / ("observation_valid_%s.tif" % pair),
    }


def cmd_normalise(args):
    manifest = load_manifest(args.manifest)
    out_path = Path(args.root) / "normalisation.json"
    if out_path.exists():
        raise SystemExit("refusing to overwrite %s" % out_path)
    payload = {}
    meta = {}

    if args.source == "windows":
        index_path = Path(args.root) / "train_index.csv"
        if not index_path.is_file():
            raise SystemExit("build the index before statistics from windows")
        rows = list(csv.DictReader(index_path.open(newline="")))
        validity_by_pair = {}
        shapes = {}
        for pair in PAIRS:
            validity_path = paths(args, pair)["validity"]
            validity_by_pair[pair] = None
            if validity_path.is_file():
                with rasterio.open(validity_path) as handle:
                    validity_by_pair[pair] = handle.read(1) > 0
                    shapes[pair] = validity_by_pair[pair].shape
            else:
                with rasterio.open(paths(args, pair)["labels"]) as handle:
                    shapes[pair] = (handle.height, handle.width)
        # The loader that materialises the patches, so date 2 carries the same
        # registration shift in both places.
        loader = pair_image_loader(args.manifest, args.registration, bands=BANDS)
        for pair in PAIRS:
            pair_rows = [row for row in rows if row["pair"] == pair]
            statistics, counts = compute_window_statistics(
                pair_rows, loader, validity_by_pair, BANDS, shapes=shapes)
            payload[pair] = statistics.summary()
            meta[pair] = {"source": "training windows, unique coordinates",
                          "windows": len(pair_rows), **counts["per_pair"][pair],
                          "nodata": counts["nodata"], "invalid": counts["invalid"]}
            for band in BANDS:
                entry = payload[pair][band]
                print("%s %s: n=%d mean %.6f std %.6f" % (
                    pair, band, entry["count"], entry["mean"], entry["std"]))
            print("   instances %d | unique %d | repeated %d (%.1f%%)"
                  % (counts["per_pair"][pair]["instances"],
                     counts["per_pair"][pair]["unique"],
                     counts["per_pair"][pair]["repeated"],
                     100.0 * counts["per_pair"][pair]["repeated"]
                     / max(1, counts["per_pair"][pair]["instances"])))
        save_statistics(out_path, payload, extra={
            "band_order": list(BANDS), "pooled_dates": True,
            "partition": "TRAIN only, inside the training windows",
            "unique_pixels": True,
            "counting": "each unique geographic coordinate once per date",
            "registration": "date 2 read through the same loader as the patches",
            "blocks": meta,
            "note": ("aligns per-band first and second moments with the training "
                     "domain; does not equalise higher moments, spatial correlation "
                     "or the per-date radiometric offset"),
        })
        print("wrote", out_path)
        return

    for pair in PAIRS:
        cfg = pair_config(manifest, pair)
        with rasterio.open(paths(args, pair)["partition"]) as handle:
            partition = handle.read(1)
        validity = None
        validity_path = paths(args, pair)["validity"]
        if validity_path.is_file():
            with rasterio.open(validity_path) as handle:
                validity = handle.read(1) > 0
        statistics, block_meta = compute_pair_statistics(
            SafeProduct(cfg["path_date1"]), SafeProduct(cfg["path_date2"]),
            partition, validity, BANDS)
        payload[pair] = statistics.summary()
        meta[pair] = block_meta
        for band in BANDS:
            entry = payload[pair][band]
            print("%s %s: n=%d mean %.6f std %.6f" % (
                pair, band, entry["count"], entry["mean"], entry["std"]))
    save_statistics(out_path, payload, extra={
        "band_order": list(BANDS),
        "pooled_dates": True,
        "partition": "TRAIN only",
        "unique_pixels": True,
        "blocks": meta,
        "note": ("aligns per-band first and second moments with the training domain; "
                 "does not equalise higher moments, spatial correlation or the "
                 "per-date radiometric offset"),
    })
    print("wrote", out_path)


def cmd_index(args):
    out_csv = Path(args.root) / "train_index.csv"
    if out_csv.exists():
        raise SystemExit("refusing to overwrite %s" % out_csv)
    rows = []
    for pair in PAIRS:
        label_dir = paths(args, pair)["labels"].parent
        pair_rows = build_local_patch_index(
            Path(args.manifest), label_dir, Path(args.partition_dir),
            patch_size=PATCH, stride=STRIDE,
            min_labeled_fraction={"train": TRAIN_FRACTION, "validation": 0.0},
            include_test=False, pairs=[pair])
        train_rows = [row for row in pair_rows if row["split"] == "train"]
        rows.extend(train_rows)
        print("pair %s: %d windows, %d train" % (pair, len(pair_rows), len(train_rows)))
    write_local_patch_index(rows, out_csv, PATCH, STRIDE,
                            {"train": TRAIN_FRACTION, "validation": 0.0},
                            include_test=False)
    meta_path = out_csv.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta.update({
        "validation_window_source": "terrashift.data.validation_windows.ValidationLatticeDataset",
        "validation_note": ("validation is not indexed here; it uses the inference "
                            "lattice and the validation-only target"),
        "normalisation": str((Path(args.root) / "normalisation.json").resolve()),
        "test_index_included": False,
        "test_label_pixels_read": False,
        "uses_human_labels_only": True,
        "uses_osm_labels": False,
        "buffer_or_cross_split_windows_included": False,
        "normalisation_scheme": "per-pair per-band z-score, pooled over both dates",
    })
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("wrote", out_csv)


def cmd_materialise(args):
    root = Path(args.root)
    out_dir = root / "patches"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit("refusing to overwrite %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalisation = load_statistics(root / "normalisation.json")
    manifest = load_manifest(args.manifest)
    loader = pair_image_loader(args.manifest, args.registration, bands=BANDS)
    rows = list(csv.DictReader((root / "train_index.csv").open(newline="")))
    records = []
    for index, row in enumerate(rows):
        pair = row["pair"]
        row_off, col_off = int(row["row_off"]), int(row["col_off"])
        mean, std = pair_statistics(normalisation, pair)
        image1, image2 = loader(pair, row_off, col_off, PATCH)
        image1 = normalise_reflectance(image1, mean, std)
        image2 = normalise_reflectance(image2, mean, std)
        with rasterio.open(row["label_path"]) as handle:
            label = handle.read(1, window=((row_off, row_off + PATCH),
                                           (col_off, col_off + PATCH)))
        if label.shape != (PATCH, PATCH):
            raise ValueError("label window has the wrong shape: %s" % (label.shape,))
        values = set(int(v) for v in np.unique(label))
        if not values <= {0, 1, 255}:
            raise ValueError("label window has unexpected values %s" % sorted(values))
        name = "%s_r%05d_c%05d" % (pair, row_off, col_off)
        np.save(out_dir / (name + "_date1.npy"), image1.astype(np.float16))
        np.save(out_dir / (name + "_date2.npy"), image2.astype(np.float16))
        np.save(out_dir / (name + "_label.npy"), label)
        records.append({
            "name": name, "pair": pair, "split": row["split"],
            "row_off": row_off, "col_off": col_off,
            "date1": "%s_date1.npy" % name, "date2": "%s_date2.npy" % name,
            "label": "%s_label.npy" % name,
            "reviewed_pixels": int((label != 255).sum()),
            "positive_pixels": int((label == 1).sum()),
            "ignored_pixels": int((label == 255).sum()),
            "normalisation_mean": [float(v) for v in mean.ravel()],
            "normalisation_std": [float(v) for v in std.ravel()],
        })
        if (index + 1) % 50 == 0:
            print("  %d/%d" % (index + 1, len(rows)))
    manifest_path = root / "dataset_manifest.json"
    if manifest_path.exists():
        raise SystemExit("refusing to overwrite %s" % manifest_path)
    manifest_path.write_text(json.dumps({
        "patch_size": PATCH, "stride": STRIDE,
        "train_min_labeled_fraction": TRAIN_FRACTION,
        "bands": list(BANDS), "dates": ["date1 (before)", "date2 (after)"],
        "dtype": "float16 normalised reflectance; label uint8 0/1/255",
        "normalisation": normalisation,
        "samples": records,
        "counts": {
            "patches": len(records),
            "positive_pixels": sum(r["positive_pixels"] for r in records),
            "reviewed_pixels": sum(r["reviewed_pixels"] for r in records),
            "ignored_pixels": sum(r["ignored_pixels"] for r in records),
        },
        "test_included": False,
        "validation_source": "inference lattice, not this dataset",
    }, indent=1) + "\n", encoding="utf-8")
    print("wrote", manifest_path)


def cmd_verify(args):
    root = Path(args.root)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    normalisation = manifest["normalisation"]
    failures = []
    stats = {}
    for record in manifest["samples"]:
        pair = record["pair"]
        # float32 from here on: float16 std over 256*256*4 values overflows.
        d1 = np.load(root / "patches" / record["date1"]).astype(np.float32)
        d2 = np.load(root / "patches" / record["date2"]).astype(np.float32)
        label = np.load(root / "patches" / record["label"])
        for name, array in (("date1", d1), ("date2", d2)):
            if array.shape != (len(BANDS), PATCH, PATCH):
                failures.append("%s %s shape %s" % (record["name"], name, array.shape))
            if not np.all(np.isfinite(array)):
                failures.append("%s %s is not finite" % (record["name"], name))
        if label.shape != (PATCH, PATCH):
            failures.append("%s label shape" % record["name"])
        if not set(int(v) for v in np.unique(label)) <= {0, 1, 255}:
            failures.append("%s label values" % record["name"])
        if record["reviewed_pixels"] == 0:
            failures.append("%s has no reviewed pixels" % record["name"])
        mean, std = pair_statistics(normalisation, pair)
        for name, array in (("date1", d1), ("date2", d2)):
            stats.setdefault(pair, {}).setdefault(name, []).append(
                (float(array.mean()), float(array.std())))
    summary = {}
    for pair, dates in stats.items():
        summary[pair] = {}
        for name, values in dates.items():
            means = np.array([v[0] for v in values])
            stds = np.array([v[1] for v in values])
            summary[pair][name] = {"mean_of_patch_means": float(means.mean()),
                                   "std_of_patch_means": float(means.std()),
                                   "mean_of_patch_stds": float(stds.mean())}
    report = {"patches": len(manifest["samples"]), "failures": failures[:20],
              "failure_count": len(failures), "patches_summary": summary,
              "counts": manifest["counts"]}
    out = root / "verification.json"
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1)[:2000])
    if failures:
        raise SystemExit("verification failed with %d problems" % len(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("normalise", "index", "materialise", "verify"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--labels-dir", default="outputs/labels")
    parser.add_argument("--partition-dir", default="outputs/partitions_v2_observation_valid")
    parser.add_argument("--validity-dir", default="outputs/baseline")
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    parser.add_argument("--source", choices=("windows", "partition"), default="windows",
                        help="population for the normalisation statistics")
    parser.add_argument("--registration",
                        default="outputs/audit/registration_v2_observation_valid/registration.json")
    args = parser.parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    {"normalise": cmd_normalise, "index": cmd_index,
     "materialise": cmd_materialise, "verify": cmd_verify}[args.command](args)


if __name__ == "__main__":
    main()
