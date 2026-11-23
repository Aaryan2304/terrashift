"""Classical change-detection baseline: no learning, fixed by hand.

Grid and validity: 10 m analysis grid from date 1, plus the paired observation-valid mask over source
NoData, MSK_CLASSI, MSK_QUALIT and the registration footprint. The shoreline stays a water state,
never invalid.

Per pair: four 10 m bands of both dates, date 2 shifted by the measured translation and
histogram-matched on stable land, then a per-pixel score thresholded at a percentile. That threshold
is scale-invariant but only a scene-relative prior, not a calibrated operating point, since these
tiles carry no labels.

Outputs: validity, score, candidate mask (255 = invalid) and component CSV, never overwritten. The
candidate mask is exploratory.
"""

import argparse
import csv
import json
import os

import numpy as np
import rasterio
import yaml
from scipy import ndimage as ndi


from terrashift.data.grids import grid_from_band
from terrashift.data.manifest import load_manifest, pair_paths
from terrashift.registration import align_to_reference
from terrashift.data.safeloader import SafeProduct
from terrashift.data.validity import build_paired_validity, write_validity_raster
from terrashift.preprocessing.indices import ndwi
from terrashift.preprocessing.normalize import (apply_histogram, evaluate, fit_histogram,
                                  reference_scaled_change_score, select_pifs)
from terrashift.preprocessing.water import thresholds_from_config, water_masks
from terrashift.geospatial.geometry import clean_binary_mask

DEFAULT_OUT_DIR = "outputs/baseline"
# The masked registration is the authoritative estimate. It carries
# validity_masked=true and the per-window mask counts.
REG_PATH = "outputs/audit/registration_v2_observation_valid/registration.json"

# Pair name -> (path_date1, path_date2) is resolved from the manifest inside
# main(). The manifest needs SENTINEL_DATA_ROOT in the environment, so it must
# not be read at import time: importing this module for a unit test would then
# depend on the machine's data layout.
BANDS = ["B02", "B03", "B04", "B08"]
GRID_M = 20                      # analysis grid, metres
SOURCE_RES = 10                  # native resolution of the bands
MIN_COMPONENT_PX = 4             # at 20 m, 4 px = 1600 m2


def read_stack(product, shape):
    stack = []
    for b in BANDS:
        a, _, _ = product.read_band(b, out_shape=shape)
        stack.append(a)
    return stack


def candidate_mask_from_score(score, valid, percentile,
                              min_component_px=MIN_COMPONENT_PX):
    """Threshold, clean, and filter a change score on the valid footprint.

    Kept as a pure function so the invalidity contract can be tested without
    reading imagery: no invalid pixel can ever become a candidate, and
    morphology is re-clipped to ``valid`` afterwards. The on-disk mask then
    uses 255 for the ineligible pixels, so "unchanged" and "invalid" are
    different values.
    """
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        raise ValueError("no valid pixels remain for the candidate threshold")
    threshold = float(np.percentile(score[valid], percentile))
    mask = clean_binary_mask((score > threshold) & valid, valid=valid,
                             min_area_px=min_component_px)
    return mask, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="*", default=None,
                        help="pair names, e.g. A B; default is every pair")
    parser.add_argument("--config", default="config.yaml",
                        help="configuration file with change_score_percentile")
    parser.add_argument("--percentile", type=float, default=None,
                        help="override the configured candidate threshold percentile")
    parser.add_argument("--output-dir", default=None,
                        help="new directory for this run's outputs; existing "
                             "files are never replaced")
    parser.add_argument("--registration", default=REG_PATH,
                        help="registration summary with applied_shift_px")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    percentile = (cfg["preprocess"]["thresholds"]["change_score_percentile"]
                  if args.percentile is None else args.percentile)
    if not 0.0 < percentile < 100.0:
        parser.error("percentile must be between 0 and 100")

    out_dir = args.output_dir or DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    water_thresholds = thresholds_from_config(cfg)

    with open(args.registration) as fh:
        reg = json.load(fh)

    manifest = load_manifest()
    pairs = {name: pair_paths(manifest, name) for name in sorted(manifest["pairs"])}

    requested = args.pairs or list(pairs)
    unknown = set(requested) - set(pairs)
    if unknown:
        raise SystemExit("unknown pair(s): %s; choose A and/or B" %
                         ", ".join(sorted(unknown)))

    for name in requested:
        targets = [
            os.path.join(out_dir, "observation_valid_%s.tif" % name),
            os.path.join(out_dir, "change_score_%s.tif" % name),
            os.path.join(out_dir, "candidate_mask_%s.tif" % name),
            os.path.join(out_dir, "candidate_components_%s.csv" % name),
        ]
        existing = [p for p in targets if os.path.exists(p)]
        if existing:
            raise SystemExit(
                "refusing to overwrite existing baseline outputs: %s\n"
                "choose a new --output-dir so the previous run stays "
                "available for comparison." % ", ".join(existing))

        s1, s2 = pairs[name]
        print("=" * 80)
        print("PAIR", name)
        p1 = SafeProduct(s1)
        p2 = SafeProduct(s2)
        grid = grid_from_band(p1, "B08", GRID_M)
        if grid.width * grid.height > 60_000_000:
            raise SystemExit("output grid is too large: %dx%d" % (grid.height, grid.width))

        ref = read_stack(p1, (grid.height, grid.width))
        mov = read_stack(p2, (grid.height, grid.width))

        # --- registration: apply the measured shift to date 2
        shift = reg[name]["initial"].get("applied_shift_px")
        coverage_shift = None
        if shift is not None:
            dx_m = shift[0] * SOURCE_RES
            dy_m = shift[1] * SOURCE_RES
            grid_dx = dx_m / GRID_M
            grid_dy = dy_m / GRID_M
            mov = [align_to_reference(b, grid_dx, grid_dy) for b in mov]
            # align_to_reference calls apply_shift with the negated
            # displacement, so the coverage mask must use the same sign.
            coverage_shift = (-grid_dx, -grid_dy)
            print("  applied registration shift: %.2f m, %.2f m" % (dx_m, dy_m))

        # --- observation validity: NoData + quality flags + warp coverage
        validity = build_paired_validity(p1, p2, grid, shift=coverage_shift,
                                        bands=tuple(BANDS))
        paired = validity["paired"]
        print("  observation valid: %.1f%% of the analysis grid (%d of %d pixels)"
              % (100.0 * paired.mean(), int(paired.sum()), paired.size))
        for label in ("date1", "date2"):
            detail = validity["detail_" + label]
            nodata = sum(v for k, v in detail.items() if k.startswith("nodata_"))
            print("    %s invalid: %d (nodata %d, quality %d)"
                  % (label, detail["total_invalid"], nodata,
                     detail["total_invalid"] - nodata))

        # --- water state, classified only where both dates are valid
        w1 = ndwi(ref[1], ref[3])
        w2 = ndwi(mov[1], mov[3])
        wm = water_masks(w1, w2, ndwi_threshold=water_thresholds["ndwi_threshold"],
                         b08_1=ref[3], b08_2=mov[3],
                         nir_threshold=water_thresholds["nir_threshold"],
                         observation_valid=paired)
        land = wm["land"]
        print("  land pixels: %d (%.1f%% of the grid); dynamic shoreline kept "
              "as a water state: %d"
              % (land.sum(), land.mean() * 100, int(wm["dynamic_shoreline"].sum())))

        # --- normalisation on stable land
        initial_score, common_valid = reference_scaled_change_score(ref, mov, land)
        stable = select_pifs(initial_score, common_valid, fraction=0.5)
        tables = fit_histogram(ref, mov, stable)
        mov_n = apply_histogram(mov, tables)
        before = evaluate(ref, mov, land)
        after = evaluate(ref, mov_n, land)
        print("  normalisation MAE %.5f -> %.5f (%.1f%%)" % (
            before["mae_mean"], after["mae_mean"],
            100.0 * (before["mae_mean"] - after["mae_mean"]) / before["mae_mean"]))

        # --- change score on the paired-valid land footprint
        cs, valid = reference_scaled_change_score(ref, mov_n, land)
        p = np.percentile(cs[valid], [50, 90, 95, 99, 99.5, 99.9])
        print("  change score percentiles 50/90/95/99/99.5/99.9: " +
              " ".join("%.3f" % v for v in p))

        mask, threshold = candidate_mask_from_score(cs, valid, percentile)
        print("  percentile %.1f -> threshold %.4f -> %d candidate px "
              "(%.3f%% of valid land)" % (
                  percentile, threshold, mask.sum(),
                  mask.sum() / max(valid.sum(), 1) * 100))
        labels, n = ndi.label(mask)
        print("  components after morphology and area filter: %d" % n)

        # --- the validity raster is written first: it is the contract for the
        # score, the partition generator, and the annotation validator.
        validity_path = targets[0]
        write_validity_raster(
            validity_path, paired, grid,
            tags={"source_pair": name,
                  "valid_pixels": int(paired.sum()),
                  "total_pixels": int(paired.size),
                  "date1_invalid": validity["detail_date1"]["total_invalid"],
                  "date2_invalid": validity["detail_date2"]["total_invalid"],
                  "registration_shift_px": validity["shift_px"]})
        print("  wrote", validity_path)

        with rasterio.open(targets[1], "w", driver="GTiff", height=grid.height,
                           width=grid.width, count=1, dtype="float32",
                           transform=grid.transform, crs=grid.crs, nodata=-9999.0,
                           tiled=True, compress="deflate") as ds:
            ds.write(np.where(valid, cs, -9999.0).astype(np.float32), 1)
            ds.update_tags(
                score_method="euclidean_band_difference_scaled_by_reference_land_std",
                threshold_percentile=percentile,
                threshold_value=threshold,
                threshold_status="percentile_prior_not_calibrated",
                source_pair=name,
                validity_raster=os.path.basename(validity_path),
                validity_definition="nodata+msk_classi+msk_qualit+registration_coverage",
            )
        print("  wrote", targets[1])

        # 255 marks pixels that were never eligible, so "unchanged" and
        # "invalid" cannot be confused in the binary product.
        candidate = np.where(valid, mask.astype(np.uint8), np.uint8(255))
        with rasterio.open(targets[2], "w", driver="GTiff", height=grid.height,
                           width=grid.width, count=1, dtype="uint8",
                           transform=grid.transform, crs=grid.crs, nodata=255,
                           compress="deflate") as ds:
            ds.write(candidate, 1)
            ds.update_tags(
                output_type="exploratory_candidate_mask_not_a_detection",
                grid_m=GRID_M,
                valid_no_change_value="0",
                candidate_change_value="1",
                invalid_value="255",
                threshold_percentile=percentile,
                threshold_value=threshold,
                threshold_status="percentile_prior_not_calibrated",
                validity_raster=os.path.basename(validity_path),
                source_pair=name)
        print("  wrote", targets[2])

        # --- components, computed in raster-wide passes
        component_ids = np.arange(1, n + 1)
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        centroids = ndi.center_of_mass(mask, labels=labels, index=component_ids)
        means = ndi.mean(cs, labels=labels, index=component_ids)
        rows = []
        for i, (sz, (cy, cx), mean_score) in enumerate(
                zip(sizes[1:], centroids, means), start=1):
            if sz == 0:
                continue
            x_utm, y_utm = rasterio.transform.xy(grid.transform, cy, cx)
            rows.append({"pair": name, "component": i, "area_px": int(sz),
                         "area_m2": int(sz) * GRID_M * GRID_M,
                         "centroid_col": float(cx), "centroid_row": float(cy),
                         "centroid_easting": x_utm, "centroid_northing": y_utm,
                         "mean_change_score": float(mean_score)})
        rows.sort(key=lambda r: r["area_px"], reverse=True)
        fields = ["pair", "component", "area_px", "area_m2",
                  "centroid_col", "centroid_row", "centroid_easting",
                  "centroid_northing", "mean_change_score"]
        with open(targets[3], "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print("  wrote", targets[3], "(%d components)" % len(rows))
        if rows:
            print("  largest component: %d px = %.0f m2"
                  % (rows[0]["area_px"], rows[0]["area_m2"]))


if __name__ == "__main__":
    main()
