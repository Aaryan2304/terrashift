"""Create buffered, geographically contiguous train/val/test partitions.

The split direction is selected from valid-land coverage only. No labels or
change scores are used to choose the direction. Date 1 and date 2 share the
same split raster because they cover the same local tile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import rasterio
import yaml
from rasterio.warp import Resampling, reproject, transform_geom
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box, mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.grids import (grid_from_dataset, require_expected_crs,
                        require_same_grid, resolution)
from data.manifest import load_manifest, pair_config
from data.safeloader import SafeProduct

SPLIT_CODES = {"train": 1, "validation": 2, "test": 3}
ORIENTATIONS = (("x", False), ("y", False), ("x", True), ("y", True))


def candidate_partition(shape, axis, reverse, buffer_pixels, train_fraction, validation_fraction):
    height, width = shape
    size = width if axis == "x" else height
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("partition fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be below one")
    first = int(round(train_fraction * size))
    second = int(round((train_fraction + validation_fraction) * size))
    if first <= buffer_pixels or size - second <= buffer_pixels or second - first <= 2 * buffer_pixels:
        raise ValueError("spatial extent is too small for fractions and buffer")

    coord = np.arange(size, dtype=np.int64)
    ordered = size - 1 - coord if reverse else coord
    axis_ids = np.where(ordered < first, 1, np.where(ordered < second, 2, 3)).astype(np.uint8)
    edges = (size - first, size - second) if reverse else (first, second)
    for edge in edges:
        axis_ids[max(0, edge - buffer_pixels):min(size, edge + buffer_pixels)] = 0

    if axis == "x":
        split_grid = np.broadcast_to(axis_ids[None, :], shape)
    else:
        split_grid = np.broadcast_to(axis_ids[:, None], shape)
    return split_grid, axis_ids


def score_partition(valid, split_grid):
    total = int(valid.sum())
    if total == 0:
        raise ValueError("baseline score raster has no valid land pixels")
    counts = {name: int((valid & (split_grid == code)).sum())
              for name, code in SPLIT_CODES.items()}
    fractions = {name: count / total for name, count in counts.items()}
    retained = sum(counts.values()) / total
    return counts, fractions, retained, min(fractions.values())


def _geometry_for_axis(axis, axis_ids, shape, transform):
    height, width = shape
    geometries = {}
    for name, code in SPLIT_CODES.items():
        positions = np.flatnonzero(axis_ids == code)
        if not positions.size:
            raise ValueError("partition %s has no pixels" % name)
        start, stop = int(positions[0]), int(positions[-1]) + 1
        window = Window(start, 0, stop - start, height) if axis == "x" else Window(0, start, width, stop - start)
        left, bottom, right, top = window_bounds(window, transform)
        geometries[name] = box(left, bottom, right, top)
    return geometries


def _write_geojson(path, pair, geometries, crs, metadata):
    features = []
    for name, geom in geometries.items():
        wgs_geom = transform_geom(crs, "EPSG:4326", mapping(geom), precision=8)
        features.append({
            "type": "Feature",
            "geometry": wgs_geom,
            "properties": {
                "pair": pair,
                "split": name,
                "split_code": SPLIT_CODES[name],
                **metadata,
            },
        })
    collection = {
        "type": "FeatureCollection",
        "name": "local_pair_%s_partitions" % pair,
        "crs_note": "RFC 7946 longitude/latitude coordinates (EPSG:4326)",
        "features": features,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection, indent=2) + "\n", encoding="utf-8")


def _write_raster(path, array, reference, tags):
    profile = reference.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0,
                   compress="deflate", predictor=2, tiled=True,
                   blockxsize=512, blockysize=512)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        dst.update_tags(**tags)


def build_pair(pair, manifest, score_path, output_dir, buffer_m, train_fraction,
               validation_fraction, validity_path=None,
               allow_score_mask_fallback=False):
    targets = (
        output_dir / ("pair_%s_partition_20m.tif" % pair),
        output_dir / ("pair_%s_partition_10m.tif" % pair),
        output_dir / ("pair_%s_partitions.geojson" % pair),
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite partition outputs; choose a new --output-dir: %s"
            % ", ".join(str(path) for path in existing))
    if not score_path.is_file():
        raise FileNotFoundError(score_path)
    with rasterio.open(score_path) as score_ds:
        score_grid = grid_from_dataset(score_ds)
        require_expected_crs(score_grid, pair_config(manifest, pair)["crs"],
                             label="score raster %s" % score_path)
        score = score_ds.read(1, masked=True)
        score_crs = score_ds.crs
        score_transform = score_ds.transform
        shape = score_ds.shape
        score_profile = score_ds.profile.copy()
        score_finite = (~np.ma.getmaskarray(score)) & np.isfinite(score.filled(np.nan))

        # The partition split must be chosen from the quality-valid footprint,
        # not from "the score raster happened to have a value here". The score
        # raster is a derived product; the validity raster is the record of
        # which pixels were ever eligible.
        if validity_path is not None and Path(validity_path).is_file():
            with rasterio.open(validity_path) as validity_ds:
                require_same_grid(grid_from_dataset(validity_ds), score_grid,
                                  "validity raster %s" % validity_path,
                                  "score raster %s" % score_path)
                valid = validity_ds.read(1) > 0
            valid_source = "observation_validity_raster"
        elif allow_score_mask_fallback:
            valid = score_finite.copy()
            valid_source = "change_score_dataset_mask_fallback"
        else:
            raise FileNotFoundError(
                "observation-validity raster not found: %s\n"
                "run scripts/run_classical_baseline.py with this output "
                "directory, or pass --allow-score-mask-fallback to accept the "
                "weaker change-score mask." % validity_path)
        # A validity raster cannot claim more than the score actually covers.
        valid &= score_finite
        valid_count = int(valid.sum())
        if score_crs is None:
            raise ValueError("partition source has no CRS: %s" % score_path)
        pixel_size = resolution(score_grid)[0]
        if abs(pixel_size - 20.0) > 1e-6:
            raise ValueError("expected a 20 m analysis grid, found %g m: %s"
                             % (pixel_size, score_path))
        buffer_pixels = int(math.ceil(buffer_m / pixel_size))
        candidates = []
        for rank, (axis, reverse) in enumerate(ORIENTATIONS):
            grid, axis_ids = candidate_partition(
                shape, axis, reverse, buffer_pixels,
                train_fraction, validation_fraction)
            counts, fractions, retained, minimum = score_partition(valid, grid)
            candidates.append({
                "rank": rank, "axis": axis, "reverse": reverse,
                "grid": grid, "axis_ids": axis_ids,
                "counts": counts, "fractions": fractions,
                "retained": retained, "minimum": minimum,
            })
        chosen = max(candidates, key=lambda item: (item["minimum"], item["retained"], -item["rank"]))
        split_grid = np.asarray(chosen["grid"], dtype=np.uint8).copy()
        split_codes = {
            "1": "train", "2": "validation", "3": "test", "0": "buffer_ignore"
        }
        transform_meta = {
            "axis": chosen["axis"],
            "train_from": "high-coordinate" if chosen["reverse"] else "low-coordinate",
            "train_fraction_of_extent": train_fraction,
            "validation_fraction_of_extent": validation_fraction,
            "test_fraction_of_extent": 1.0 - train_fraction - validation_fraction,
            "buffer_m_each_side_of_boundary": buffer_m,
            "valid_land_pixels_by_split": chosen["counts"],
            "valid_land_fraction_by_split": chosen["fractions"],
            "valid_land_retained_fraction": chosen["retained"],
            "baseline_valid_land_pixels": valid_count,
            "valid_pixels_used_only_for_axis_selection": True,
            "labels_used_for_partition_selection": False,
            "validity_source": valid_source,
        }
        tags = {
            "pair": pair,
            "partition_method": "contiguous_stripes_with_buffer_v1",
            "split_axis": chosen["axis"],
            "train_from": transform_meta["train_from"],
            "buffer_m_each_side": str(buffer_m),
            "split_codes_json": json.dumps(split_codes, sort_keys=True),
            "date1_and_date2_share_partition": "true",
            "selection_used_labels": "false",
            "selection_valid_land_counts_json": json.dumps(chosen["counts"], sort_keys=True),
            "validity_source": valid_source,
        }
        geometries = _geometry_for_axis(chosen["axis"], chosen["axis_ids"], shape, score_transform)
        raster_path, _, geojson_path = targets
        _write_raster(raster_path, split_grid, score_ds, tags)
        _write_geojson(geojson_path, pair, geometries, score_crs, transform_meta)

    pair_cfg = pair_config(manifest, pair)
    date1 = SafeProduct(pair_cfg["path_date1"])
    reference_path = date1.band_path("B02")
    ten_meter_path = targets[1]
    with rasterio.open(reference_path) as ref, rasterio.open(raster_path) as src:
        target = np.zeros((ref.height, ref.width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1), destination=target,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref.transform, dst_crs=ref.crs,
            resampling=Resampling.nearest, src_nodata=0, dst_nodata=0,
        )
        target_tags = dict(tags)
        target_tags["target_grid"] = "date1 native B02 grid"
        target_tags["reference_raster"] = str(reference_path)
        _write_raster(ten_meter_path, target, ref, target_tags)

    summary = {
        "pair": pair,
        "shape_20m": list(shape),
        "crs": str(score_crs),
        "split_axis": chosen["axis"],
        "train_from": transform_meta["train_from"],
        "buffer_pixels_20m_each_side": buffer_pixels,
        **transform_meta,
        "paths": {
            "partition_20m": str(raster_path),
            "partition_10m": str(ten_meter_path),
            "partitions_geojson": str(geojson_path),
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    parser.add_argument("--config", default="configs/local_partitions.yaml")
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--validity-dir", type=Path, default=None,
                        help="directory holding observation_valid_<PAIR>.tif; "
                             "defaults to --score-dir")
    parser.add_argument("--allow-score-mask-fallback", action="store_true",
                        help="accept the change-score dataset mask when no "
                             "observation-validity raster exists (weaker)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pairs", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--buffer-m", type=float, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    score_pattern = cfg["score_raster_pattern"]
    score_dir = args.score_dir
    validity_dir = args.validity_dir if args.validity_dir is not None else score_dir
    output_dir = args.output_dir if args.output_dir is not None else Path(cfg["output_dir"])
    if any((output_dir / name).exists() for name in ("partition_summary.json", "partition_summary.csv")):
        parser.error("refusing to overwrite partition summaries; choose a new --output-dir")
    buffer_m = args.buffer_m if args.buffer_m is not None else float(cfg["buffer_m_each_side_of_boundary"])
    train_fraction = args.train_fraction if args.train_fraction is not None else float(cfg["train_fraction_of_extent"])
    validation_fraction = (args.validation_fraction if args.validation_fraction is not None
                           else float(cfg["validation_fraction_of_extent"]))
    if buffer_m < 0:
        parser.error("buffer-m must be non-negative")
    manifest = load_manifest(args.manifest)
    summaries = []
    for pair in args.pairs:
        score_path = (score_dir / ("change_score_%s.tif" % pair)
                      if score_dir is not None else Path(score_pattern.format(pair=pair)))
        validity_path = (validity_dir / ("observation_valid_%s.tif" % pair)
                         if validity_dir is not None
                         else Path(score_pattern.format(pair=pair)).parent /
                         ("observation_valid_%s.tif" % pair))
        summaries.append(build_pair(
            pair, manifest, score_path, output_dir, buffer_m,
            train_fraction, validation_fraction,
            validity_path=validity_path,
            allow_score_mask_fallback=args.allow_score_mask_fallback))
    summary_path = output_dir / "partition_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "partition_summary.csv"
    fields = ["pair", "split_axis", "train_from", "buffer_m_each_side_of_boundary",
              "valid_land_pixels_by_split", "valid_land_fraction_by_split",
              "valid_land_retained_fraction", "partition_20m", "partition_10m", "partitions_geojson"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow({
                "pair": item["pair"], "split_axis": item["split_axis"],
                "train_from": item["train_from"],
                "buffer_m_each_side_of_boundary": item["buffer_m_each_side_of_boundary"],
                "valid_land_pixels_by_split": json.dumps(item["valid_land_pixels_by_split"], sort_keys=True),
                "valid_land_fraction_by_split": json.dumps(item["valid_land_fraction_by_split"], sort_keys=True),
                "valid_land_retained_fraction": item["valid_land_retained_fraction"],
                "partition_20m": item["paths"]["partition_20m"],
                "partition_10m": item["paths"]["partition_10m"],
                "partitions_geojson": item["paths"]["partitions_geojson"],
            })
        fh.flush()
    for item in summaries:
        print("Pair %s: axis=%s, train_from=%s, valid land by split=%s, retained=%.3f" % (
            item["pair"], item["split_axis"], item["train_from"],
            item["valid_land_pixels_by_split"], item["valid_land_retained_fraction"]))
        for key, path in item["paths"].items():
            print("  %s: %s" % (key, path))
    print("wrote", summary_path)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
