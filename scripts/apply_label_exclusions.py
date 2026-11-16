"""Apply an exclusion manifest to a manual annotation export.

Records named in the manifest are dropped from the output. The master file is
never touched: this reads it and writes a new export. Every applied exclusion is
reported, and the script refuses to write if an excluded ID is missing from the
source, so a typo cannot silently pass.

    python scripts/apply_label_exclusions.py \
        --labels labels/manual_labels_A.geojson \
        --exclusions labels/exclusions_A.json \
        --output labels/manual_labels_A_filtered.geojson \
        --report outputs/labels/pair_a_qa/exclusions_applied.json
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_manifest(path):
    with open(path) as handle:
        manifest = json.load(handle)
    if "exclusions" not in manifest or not isinstance(manifest["exclusions"], list):
        raise ValueError("manifest has no 'exclusions' list: %s" % path)
    ids = [item["record_id"] for item in manifest["exclusions"]]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest lists a record_id more than once")
    for item in manifest["exclusions"]:
        if not item.get("reason"):
            raise ValueError("exclusion %s has no reason" % item.get("record_id"))
    return manifest


def apply_exclusions(features, manifest, pair=None):
    """Split features into kept and excluded. Returns (kept, excluded, unknown)."""
    wanted = {item["record_id"]: item for item in manifest["exclusions"]}
    kept, excluded = [], []
    seen = set()
    for feature in features:
        props = feature.get("properties", {})
        if pair is not None and props.get("pair") != pair:
            raise ValueError("feature %s belongs to pair %s, expected %s"
                             % (props.get("record_id"), props.get("pair"), pair))
        record_id = props.get("record_id")
        if record_id in wanted:
            entry = dict(wanted[record_id])
            entry["label_class"] = props.get("label_class")
            excluded.append(entry)
            seen.add(record_id)
        else:
            kept.append(feature)
    unknown = sorted(set(wanted) - seen)
    return kept, excluded, unknown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--exclusions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--pair", default=None, help="assert every feature is this pair")
    args = parser.parse_args()

    for path in (args.output, args.report):
        if path and os.path.exists(path):
            raise SystemExit("refusing to overwrite %s" % path)

    manifest = load_manifest(args.exclusions)
    with open(args.labels) as handle:
        payload = json.load(handle)
    if payload.get("type") != "FeatureCollection":
        raise ValueError("input is not a GeoJSON FeatureCollection")

    kept, excluded, unknown = apply_exclusions(payload["features"], manifest, args.pair)
    if unknown:
        raise SystemExit("exclusions reference records absent from the labels: %s"
                         % ", ".join(unknown))

    output = {"type": "FeatureCollection", "name": payload.get("name", "labels"),
              "features": kept}
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=1)

    report = {
        "source": os.path.abspath(args.labels),
        "output": os.path.abspath(args.output),
        "input_features": len(payload["features"]),
        "kept_features": len(kept),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "policy": manifest.get("policy", ""),
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=1)

    print("input %d -> kept %d, excluded %d" % (len(payload["features"]),
                                                len(kept), len(excluded)))
    for item in excluded:
        print("  excluded %-6s %-14s %s" % (item["record_id"], item["reason"],
                                            item["detail"][:70] + "..."))
    print("wrote", args.output)
    if args.report:
        print("wrote", args.report)


if __name__ == "__main__":
    main()
