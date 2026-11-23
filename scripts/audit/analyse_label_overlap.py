from pathlib import Path
"""Overlay and measure the overlap between two annotation records.

Writes a QGIS-ready GeoJSON with both polygons and their intersection, a
before/after image crop around the overlap, and a JSON summary with the
attribute values, the intersection area in pixels and square metres, and the
overlap fraction for each record.

Read-only with respect to the master labels.

    python scripts/audit/analyse_label_overlap.py \
        --labels labels/manual_labels_A.geojson \
        --records A_8 A_9 --pair A \
        --output-dir outputs/labels/pair_a_qa/overlap_A8_A9
"""

import argparse
import json
import os

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.manifest import load_manifest, pair_paths
from terrashift.data.safeloader import SafeProduct
from terrashift.geospatial.imagery import RGB_BANDS

PADDING_M = 120.0


def read_rgb_window(product, window):
    channels = []
    for band in RGB_BANDS:
        with rasterio.open(product.band_files[band]) as dataset:
            channels.append(dataset.read(1, window=window).astype(np.float32))
    return np.stack(channels, axis=-1)


def stretch(first, second):
    limits = []
    for channel in range(3):
        sample = np.concatenate([first[..., channel].ravel(), second[..., channel].ravel()])
        sample = sample[np.isfinite(sample)]
        limits.append(np.percentile(sample, [2, 98]))

    def apply(image):
        out = np.zeros_like(image, dtype=np.float32)
        for channel, (low, high) in enumerate(limits):
            out[..., channel] = np.clip((image[..., channel] - low) / max(high - low, 1e-9),
                                        0.0, 1.0)
        return np.nan_to_num(out)

    return apply(first), apply(second)


def overlap_metrics(geom_a, geom_b, pixel_area_m2=100.0):
    """Intersection area in square metres, pixels, and each record's share."""
    intersection = geom_a.intersection(geom_b)
    area_m2 = float(intersection.area)
    return {
        "intersection_area_m2": round(area_m2, 1),
        "intersection_area_px": int(round(area_m2 / pixel_area_m2)),
        "share_of_first": round(area_m2 / float(geom_a.area), 4) if geom_a.area else 0.0,
        "share_of_second": round(area_m2 / float(geom_b.area), 4) if geom_b.area else 0.0,
        "iou": round(area_m2 / float(geom_a.union(geom_b).area), 4)
        if geom_a.union(geom_b).area else 0.0,
        "first_area_m2": round(float(geom_a.area), 1),
        "second_area_m2": round(float(geom_b.area), 1),
    }


def burned_overlap(geom_a, geom_b, transform, size=64):
    """Pixel-level overlap on a local grid, to confirm the geometric area."""
    minx = min(geom_a.bounds[0], geom_b.bounds[0])
    miny = min(geom_a.bounds[1], geom_b.bounds[1])
    maxx = max(geom_a.bounds[2], geom_b.bounds[2])
    maxy = max(geom_a.bounds[3], geom_b.bounds[3])
    width = int(np.ceil((maxx - minx) / abs(transform.a))) + 2
    height = int(np.ceil((maxy - miny) / abs(transform.e))) + 2
    local = rasterio.Affine(transform.a, 0.0, minx, 0.0, transform.e, maxy)
    a = rasterize([(geom_a.__geo_interface__, 1)], out_shape=(height, width),
                  transform=local, fill=0, dtype="uint8").astype(bool)
    b = rasterize([(geom_b.__geo_interface__, 1)], out_shape=(height, width),
                  transform=local, fill=0, dtype="uint8").astype(bool)
    return int((a & b).sum()), int(a.sum()), int(b.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--records", nargs=2, required=True)
    parser.add_argument("--pair", choices=["A", "B"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise SystemExit("refusing to write into a non-empty %s" % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.labels) as handle:
        payload = json.load(handle)
    by_id = {f["properties"]["record_id"]: f for f in payload["features"]}
    missing = [r for r in args.records if r not in by_id]
    if missing:
        raise SystemExit("records not found: %s" % ", ".join(missing))

    # The export is WGS84; measure in the pair's projected CRS.
    from pyproj import Transformer
    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    import shapely

    def metric(record_id):
        geom = shape(by_id[record_id]["geometry"])
        return shapely.transform(geom, lambda c: np.column_stack(
            to_metric.transform(c[:, 0], c[:, 1])))

    first, second = args.records
    geom_a, geom_b = metric(first), metric(second)
    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    px_overlap, px_a, px_b = burned_overlap(geom_a, geom_b, transform)
    metrics = overlap_metrics(geom_a, geom_b)
    metrics["pixel_overlap"] = px_overlap
    metrics["pixel_count_first"] = px_a
    metrics["pixel_count_second"] = px_b

    intersection = geom_a.intersection(geom_b)
    from shapely.ops import transform as shapely_transform
    to_degrees = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    to_wgs84 = lambda geom: shapely_transform(to_degrees.transform, geom)  # noqa: E731

    features = []
    for record_id, geom in ((first, geom_a), (second, geom_b)):
        props = dict(by_id[record_id]["properties"])
        props["overlay_role"] = "annotation"
        features.append({"type": "Feature", "properties": props,
                         "geometry": json.loads(json.dumps(to_wgs84(geom).__geo_interface__))})
    features.append({
        "type": "Feature",
        "properties": {
            "overlay_role": "intersection",
            "record_ids": "%s+%s" % (first, second),
            "area_m2": metrics["intersection_area_m2"],
            "area_px_10m": metrics["intersection_area_px"],
        },
        "geometry": json.loads(json.dumps(to_wgs84(intersection).__geo_interface__)),
    })
    overlay_path = os.path.join(args.output_dir, "overlap_%s_%s.geojson"
                                % (first, second))
    with open(overlay_path, "w") as handle:
        json.dump({"type": "FeatureCollection", "name": "overlap_%s_%s" % (first, second),
                   "features": features}, handle, indent=1)

    summary = {
        "pair": args.pair,
        "records": {
            first: {"label_class": by_id[first]["properties"]["label_class"],
                    "evidence": by_id[first]["properties"]["evidence"],
                    "confidence": by_id[first]["properties"]["confidence"],
                    "reviewer": by_id[first]["properties"]["reviewer"]},
            second: {"label_class": by_id[second]["properties"]["label_class"],
                     "evidence": by_id[second]["properties"]["evidence"],
                     "confidence": by_id[second]["properties"]["confidence"],
                     "reviewer": by_id[second]["properties"]["reviewer"]},
        },
        "metrics": metrics,
        "overlay": os.path.abspath(overlay_path),
    }

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        manifest = load_manifest()
        first_path, second_path = pair_paths(manifest, args.pair)
        p1, p2 = SafeProduct(first_path), SafeProduct(second_path)
        minx, miny, maxx, maxy = geom_a.union(geom_b).bounds
        minx -= PADDING_M; miny -= PADDING_M; maxx += PADDING_M; maxy += PADDING_M
        with rasterio.open(p1.band_files[RGB_BANDS[0]]) as dataset:
            inverse = ~dataset.transform
            col0, row0 = inverse * (minx, maxy)
            col1, row1 = inverse * (maxx, miny)
            window = Window(int(col0), int(row0), int(col1 - col0), int(row1 - row0))
        before = read_rgb_window(p1, window)
        after = read_rgb_window(p2, window)
        before, after = stretch(before, after)

        figure, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
        for axis, image, title in ((axes[0], before, "date 1"),
                                   (axes[1], after, "date 2")):
            axis.imshow(image, extent=(minx, maxx, miny, maxy), interpolation="nearest")
            for geom, colour, label in ((geom_a, "lime", first),
                                        (geom_b, "red", second),
                                        (intersection, "yellow", "intersection")):
                if geom.is_empty:
                    continue
                boundary = np.asarray(geom.exterior.coords) if geom.geom_type == "Polygon" \
                    else np.vstack([np.asarray(p.exterior.coords) for p in geom.geoms])
                axis.plot(boundary[:, 0], boundary[:, 1], color=colour, linewidth=1.4,
                          label=label)
            axis.set_title(title)
            axis.set_xticks([]); axis.set_yticks([])
        axes[0].legend(loc="lower right", fontsize=8)
        figure.suptitle("%s vs %s — intersection %.0f m² (%d px at 10 m)"
                        % (first, second, metrics["intersection_area_m2"],
                           metrics["intersection_area_px"]))
        figure_path = os.path.join(args.output_dir, "overlap_%s_%s.png" % (first, second))
        figure.savefig(figure_path, dpi=130)
        plt.close(figure)
        summary["figure"] = os.path.abspath(figure_path)

    summary_path = os.path.join(args.output_dir, "overlap_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=1)

    print("records:", first, by_id[first]["properties"]["label_class"],
          "|", second, by_id[second]["properties"]["label_class"])
    print("intersection: %.1f m² = %d px at 10 m (%.1f%% of %s, %.1f%% of %s)"
          % (metrics["intersection_area_m2"], metrics["intersection_area_px"],
             100 * metrics["share_of_first"], first,
             100 * metrics["share_of_second"], second))
    print("burned overlap: %d px (%d in %s, %d in %s)"
          % (px_overlap, px_a, first, px_b, second))
    print("IoU: %.4f" % metrics["iou"])
    print("wrote", overlay_path)
    print("wrote", summary_path)
    if "figure" in summary:
        print("wrote", summary["figure"])


if __name__ == "__main__":
    main()
