from pathlib import Path
"""Export the annotation GeoPackage to the GeoJSON the rasteriser consumes.

The rasteriser only accepts RFC 7946 GeoJSON in EPSG:4326. The master file is a
GeoPackage in the pair's projected CRS, so this converts it and writes a new
export; the master is opened read-only and never modified.

GDAL writes a legacy ``crs`` member into GeoJSON output. That member is valid for
GDAL but not for RFC 7946, and the pipeline's other exporters drop it, so it is
removed here too.

    python scripts/review/export_manual_labels.py \
        --source data/manual_annotations.gpkg --layer manual_annotations \
        --pair A --output labels/manual_labels_A_v2.geojson
"""

import argparse
import json
import os

ROOT = Path(__file__).resolve().parents[2]

PROPERTY_ORDER = ["record_id", "pair", "label_class", "reviewer", "reviewed_at",
                  "evidence_source", "evidence", "confidence"]


def to_payload(features, name="manual_labels"):
    """Wrap features as a FeatureCollection with no legacy crs member.

    ``features`` is a list of ``(properties, geometry_mapping)`` pairs, already in
    WGS84. Property order follows the schema so the export reads like the
    template.
    """
    out = []
    for properties, geometry in features:
        ordered = {key: properties[key] for key in PROPERTY_ORDER if key in properties}
        for key in properties:
            if key not in ordered:
                ordered[key] = properties[key]
        out.append({"type": "Feature", "properties": ordered, "geometry": geometry})
    return {"type": "FeatureCollection", "name": name, "features": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--pair", choices=["A", "B"], default=None,
                        help="assert every feature belongs to this pair")
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    import geopandas as gpd

    frame = gpd.read_file(args.source, layer=args.layer, engine="pyogrio")
    if frame.crs is None:
        raise SystemExit("source has no CRS; refusing to guess a projection")
    source_crs = frame.crs.to_string()
    if frame.crs.to_epsg() != 4326:
        frame = frame.to_crs("EPSG:4326")

    features = []
    for _, row in frame.iterrows():
        properties = {key: row[key] for key in frame.columns if key != "geometry"}
        if args.pair is not None and properties.get("pair") != args.pair:
            raise SystemExit("feature %s belongs to pair %s, expected %s"
                             % (properties.get("record_id"), properties.get("pair"),
                                args.pair))
        features.append((properties, row.geometry.__geo_interface__))

    name = args.name or os.path.splitext(os.path.basename(args.output))[0]
    payload = to_payload(features, name=name)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=1)

    print("read %d features from %s" % (len(features), args.source))
    print("crs %s -> EPSG:4326, no legacy crs member" % source_crs)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
