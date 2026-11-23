"""Write the water-state audit raster, summary and figure for one or both pairs.

Produces the ``outputs/audit/water_masks.tif`` and audit figures promised on day 1, from the corrected
pipeline: the same observation-validity mask and water rule as the classical baseline, on the same
20 m grid.

Classes: 0 invalid (quality flag, NoData or registration gap), 1 land, 2 stable water, 3 dynamic
shoreline. The shoreline keeps its own class, since water turning to land can be real reclamation and
folding it into invalid would delete a target.

Outputs are never overwritten: ``outputs/audit/water_masks_<PAIR>.tif``, ``water_summary_<PAIR>.json``
and ``outputs/figures/audit_pair_<PAIR>.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
import yaml

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.grids import grid_from_band
from terrashift.data.manifest import load_manifest, pair_paths
from terrashift.registration import align_to_reference
from terrashift.data.safeloader import SafeProduct
from terrashift.data.validity import build_paired_validity
from terrashift.preprocessing.indices import ndwi
from terrashift.preprocessing.water import classify, thresholds_from_config, water_masks
from terrashift.geospatial.imagery import rgb, stretch_pair

BANDS = ["B02", "B03", "B04", "B08"]
GRID_M = 20
SOURCE_RES = 10
CLASS_NAMES = {"0": "invalid", "1": "land", "2": "stable_water", "3": "dynamic_shoreline"}
CLASS_COLORS = ["#00000000", "#d9c8a9", "#1f77b4", "#ff7f0e"]


def read_stack(product, shape):
    return [product.read_band(b, out_shape=shape)[0] for b in BANDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="*", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "audit")
    parser.add_argument("--figure-dir", type=Path,
                        default=ROOT / "outputs" / "figures")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    water_thresholds = thresholds_from_config(cfg)
    manifest = load_manifest()
    registration = json.loads(Path(cfg["data"]["registration_summary"]).read_text())

    planned = []
    for pair in args.pairs:
        planned.append((
            args.output_dir / ("water_masks_%s.tif" % pair),
            args.output_dir / ("water_summary_%s.json" % pair),
            args.figure_dir / ("audit_pair_%s.png" % pair)))
    existing = [str(p) for group in planned for p in group if p.exists()]
    if existing:
        raise SystemExit("refusing to overwrite existing audit outputs: %s"
                         % ", ".join(existing))

    for pair, (raster_path, summary_path, figure_path) in zip(args.pairs, planned):
        s1, s2 = pair_paths(manifest, pair)
        p1, p2 = SafeProduct(s1), SafeProduct(s2)
        grid = grid_from_band(p1, "B08", GRID_M)
        shape = (grid.height, grid.width)

        shift_px = registration[pair]["initial"].get("applied_shift_px")
        grid_dx = 0.0 if shift_px is None else shift_px[0] * SOURCE_RES / GRID_M
        grid_dy = 0.0 if shift_px is None else shift_px[1] * SOURCE_RES / GRID_M

        ref = read_stack(p1, shape)
        mov = read_stack(p2, shape)
        if shift_px is not None:
            mov = [align_to_reference(b, grid_dx, grid_dy) for b in mov]

        validity = build_paired_validity(
            p1, p2, grid,
            shift=None if shift_px is None else (-grid_dx, -grid_dy),
            bands=tuple(BANDS))
        paired = validity["paired"]

        w1 = ndwi(ref[1], ref[3])
        w2 = ndwi(mov[1], mov[3])
        water = water_masks(w1, w2,
                            ndwi_threshold=water_thresholds["ndwi_threshold"],
                            b08_1=ref[3], b08_2=mov[3],
                            nir_threshold=water_thresholds["nir_threshold"],
                            observation_valid=paired)
        classes = classify(water, observation_valid=paired)

        counts = {name: int((classes == int(code)).sum())
                  for code, name in CLASS_NAMES.items()}
        fractions = {name: count / classes.size for name, count in counts.items()}

        profile = dict(driver="GTiff", height=shape[0], width=shape[1], count=1,
                       dtype="uint8", crs=grid.crs, transform=grid.transform,
                       nodata=0, compress="deflate", tiled=True)
        raster_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(raster_path, "w", **profile) as ds:
            ds.write(classes, 1)
            ds.update_tags(
                output_type="water_state_audit",
                class_0="invalid observation", class_1="land",
                class_2="stable water", class_3="dynamic shoreline",
                ndwi_threshold=str(water_thresholds["ndwi_threshold"]),
                nir_threshold=str(water_thresholds["nir_threshold"]),
                observation_validity="nodata+msk_classi+msk_qualit+coverage",
                source_pair=pair, grid_m=str(GRID_M))

        summary = {
            "pair": pair,
            "grid": {"crs": str(grid.crs), "width": shape[1], "height": shape[0],
                     "pixels": int(classes.size)},
            "thresholds": water_thresholds,
            "class_counts": counts,
            "class_fractions": fractions,
            "observation_valid_fraction": float(paired.mean()),
            "raster": str(raster_path),
            "figure": str(figure_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        before = rgb(p1, shape)
        after = rgb(p2, shape)
        before, after = stretch_pair(before, after)
        overlay = np.ma.masked_where(classes == 0, classes)

        fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), constrained_layout=True)
        axes[0].imshow(before)
        axes[0].set_title("Date 1")
        axes[1].imshow(after)
        axes[1].set_title("Date 2")
        axes[2].imshow(after)
        axes[2].imshow(overlay, cmap=ListedColormap(CLASS_COLORS), vmin=0, vmax=3, alpha=0.75)
        axes[2].set_title("Water state (green land, blue stable water, orange shoreline)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Pair %s water audit, %d m grid; NDWI > %.2f and B08 < %.2f"
                     % (pair, GRID_M, water_thresholds["ndwi_threshold"],
                        water_thresholds["nir_threshold"]))
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=120)
        plt.close(fig)

        print("pair %s: valid %.4f, stable water %.4f, shoreline %.4f, land %.4f"
              % (pair, paired.mean(), fractions["stable_water"],
                 fractions["dynamic_shoreline"], fractions["land"]))
        print("  wrote", raster_path)
        print("  wrote", summary_path)
        print("  wrote", figure_path)


if __name__ == "__main__":
    main()
