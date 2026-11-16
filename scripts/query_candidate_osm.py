"""Fetch current OSM context for a small ranked review set.

OSM is current mapping context, not historical ground truth. The selection
(5 largest + 5 highest peak-score areas per pair) is a review prioritisation
rule, not a final detection threshold.
"""

import csv
import json
import os
import sys
import time
from collections import Counter

import requests
from pyproj import Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.osm import fetch_osm

# The candidate CSV must come from the current baseline run. The pre-correction
# run in outputs/audit is superseded; CANDIDATE_DIR points at the corrected run.
CANDIDATE_DIR = os.path.join(ROOT, "outputs", "audit", "v2_observation_valid")
RAW_DIR = os.path.join(ROOT, "outputs", "audit", "osm_raw")
OUT_CSV = os.path.join(ROOT, "outputs", "audit", "osm_candidate_context.csv")
BUFFER_M = 150.0
TOP_N = 5


def read_candidates(pair):
    path = os.path.join(CANDIDATE_DIR, "review_candidates_%s_p99_9.csv" % pair)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "candidate CSV not found: %s\n"
            "Run scripts/rank_review_candidates.py for pair %s first, then set "
            "CANDIDATE_DIR in this script to that run's output directory."
            % (path, pair))
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_area = sorted(rows, key=lambda r: int(r["area_px"]), reverse=True)[:TOP_N]
    by_peak = sorted(rows, key=lambda r: float(r["max_score"]), reverse=True)[:TOP_N]
    selected = {}
    for rank, group in (("largest", by_area), ("highest_peak_score", by_peak)):
        for row in group:
            selected[row["component_id"]] = (row, rank)
    return list(selected.values())


def candidate_bbox_lonlat(row):
    to_geo = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    x0 = float(row["bbox_min_easting"]) - BUFFER_M
    y0 = float(row["bbox_min_northing"]) - BUFFER_M
    x1 = float(row["bbox_max_easting"]) + BUFFER_M
    y1 = float(row["bbox_max_northing"]) + BUFFER_M
    corners = [to_geo.transform(x, y) for x, y in
               ((x0, y0), (x0, y1), (x1, y0), (x1, y1))]
    lons, lats = zip(*corners)
    return min(lons), min(lats), max(lons), max(lats)


def write_results(rows):
    fields = sorted({key for row in rows for key in row})
    if not fields:
        return
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    rows_out = []
    for pair in ("A", "B"):
        for row, rank in read_candidates(pair):
            west, south, east, north = candidate_bbox_lonlat(row)
            entry = {
                "pair": pair,
                "component_id": row["component_id"],
                "ranking_basis": rank,
                "candidate_latitude": row["latitude"],
                "candidate_longitude": row["longitude"],
                "buffer_m": BUFFER_M,
                "west": west, "south": south, "east": east, "north": north,
                "status": "pending",
            }
            try:
                elements = fetch_osm(west, south, east, north)
                ways = [e for e in elements if e.get("type") == "way"]
                buildings = [e for e in ways if "building" in e.get("tags", {})]
                roads = [e for e in ways if "highway" in e.get("tags", {})]
                raw_path = os.path.join(RAW_DIR, "%s_component_%s.json" %
                                        (pair, row["component_id"]))
                with open(raw_path, "w") as raw_file:
                    json.dump({"attribution": "© OpenStreetMap contributors, ODbL",
                               "query_bbox_wsen": [west, south, east, north],
                               "elements": elements}, raw_file)
                building_types = Counter(e.get("tags", {}).get("building", "")
                                         for e in buildings)
                road_types = Counter(e.get("tags", {}).get("highway", "")
                                     for e in roads)
                entry.update({
                    "status": "ok",
                    "element_count": len(elements),
                    "way_count": len(ways),
                    "building_way_count": len(buildings),
                    "road_way_count": len(roads),
                    "building_tag_counts": json.dumps(building_types, sort_keys=True),
                    "road_tag_counts": json.dumps(road_types, sort_keys=True),
                    "raw_json": os.path.relpath(raw_path, ROOT),
                    "note": "current OSM context only; not historical evidence",
                })
                print(pair, row["component_id"], "buildings", len(buildings),
                      "roads", len(roads))
            except Exception as exc:
                entry.update({"status": "error", "error": repr(exc)})
                rows_out.append(entry)
                write_results(rows_out)
                print(pair, row["component_id"], "ERROR", repr(exc))
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if isinstance(exc, requests.ConnectionError) or status_code in (406, 429):
                    raise SystemExit(
                        "Stopped the OSM batch after a service/network error. "
                        "No alternate endpoint was used; the remaining candidates were not queried.") from exc
            else:
                rows_out.append(entry)
                time.sleep(2.0)

    write_results(rows_out)
    print("wrote", OUT_CSV, "rows", len(rows_out))
    print("Attribution: © OpenStreetMap contributors, ODbL.")


if __name__ == "__main__":
    main()
