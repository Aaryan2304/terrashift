"""Choose a 10-candidate calibration shortlist and write a review sheet.

Selection is diversity-first, not score-first:

1. only candidates in a ``train`` partition and clear of the guard band;
2. five size targets per pair, at the minimum, 25th, 50th, 75th and maximum
   component area, so the shortlist spans the whole size range instead of
   clustering in the most populated band;
3. at each target, look at the candidates nearest that target by area and take
   the one farthest from the ones already chosen, so the ten are spread across
   the imagery;
4. score never enters the ranking. It is measured afterwards and reported, so
   the spread can be checked rather than engineered.

Land-cover context comes from the water mask: the land, stable-water and
shoreline fractions within 1 km of the centroid. Navigation order is a greedy
nearest-neighbour tour, so panning between candidates is short.

Nothing is labelled here. The review sheet is written with empty decision
columns.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

QUEUE = os.path.join(ROOT, "review", "manual_review_v1", "review_queue.csv")
WATER_DIR = os.path.join(ROOT, "outputs", "audit")
DEFAULT_OUT = os.path.join(ROOT, "review", "calibration_v1")

PER_PAIR = 5
STRATA = 5
CONTEXT_RADIUS_PX = 50          # 1 km on the 20 m grid
WATER_INVALID, WATER_LAND, WATER_STABLE, WATER_SHORE = 0, 1, 2, 3

SHORTLIST_FIELDS = [
    "nav_order", "candidate_id", "pair", "partition", "size_rank",
    "centroid_lon", "centroid_lat", "centroid_easting", "centroid_northing",
    "bbox_width_m", "bbox_height_m", "bbox_area_m2",
    "component_area_px_20m", "component_area_m2",
    "mean_score", "max_score",
    "land_fraction_1km", "water_fraction_1km", "shoreline_fraction_1km",
    "context_class", "nearest_train_candidate_m", "selection_reason",
]

SHEET_FIELDS = [
    "nav_order", "candidate_id", "pair", "partition", "centroid_lon", "centroid_lat",
    "bbox_width_m", "bbox_height_m", "component_area_m2", "max_score",
    "interpretability", "apparent_class", "suitable_for_polygon",
    "reason", "evidence", "notes",
]


def load_queue(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["component_area_px_20m"] = int(row["component_area_px_20m"])
        row["component_area_m2"] = float(row["component_area_m2"])
        row["bbox_width_m"] = float(row["bbox_width_m"])
        row["bbox_height_m"] = float(row["bbox_height_m"])
        row["bbox_area_m2"] = float(row["bbox_area_m2"])
        row["mean_score"] = float(row["mean_score"])
        row["max_score"] = float(row["max_score"])
        row["centroid_easting"] = float(row["centroid_easting"])
        row["centroid_northing"] = float(row["centroid_northing"])
        row["centroid_lon"] = float(row["centroid_lon"])
        row["centroid_lat"] = float(row["centroid_lat"])
    return rows


def context_fractions(dataset, easting, northing, radius_px=CONTEXT_RADIUS_PX):
    """Land, water and shoreline fractions within a window around a point."""
    col, row = ~dataset.transform * (easting, northing)
    col, row = int(round(col)), int(round(row))
    col0 = max(0, col - radius_px)
    row0 = max(0, row - radius_px)
    col1 = min(dataset.width, col + radius_px)
    row1 = min(dataset.height, row + radius_px)
    if col1 <= col0 or row1 <= row0:
        return None
    window = dataset.read(1, window=((row0, row1), (col0, col1)))
    total = window.size
    if total == 0:
        return None
    land = float((window == WATER_LAND).sum()) / total
    water = float((window == WATER_STABLE).sum()) / total
    shore = float((window == WATER_SHORE).sum()) / total
    return {"land": land, "water": water, "shoreline": shore,
            "context_class": context_class(land, water, shore)}


def context_class(land, water, shore):
    """A coarse land-cover label for review variety, not a land-use map."""
    if water >= 0.5:
        return "water_dominated"
    if shore >= 0.10:
        return "coastal_margin"
    if water >= 0.10:
        return "mixed_inland_water"
    return "inland"


def annotate_context(rows, water_dir):
    for pair in ("A", "B"):
        path = os.path.join(water_dir, "water_masks_%s.tif" % pair)
        with rasterio.open(path) as dataset:
            for row in rows:
                if row["pair"] != pair:
                    continue
                fractions = context_fractions(dataset, row["centroid_easting"],
                                              row["centroid_northing"])
                if fractions is None:
                    row["land_fraction_1km"] = row["water_fraction_1km"] = ""
                    row["shoreline_fraction_1km"] = ""
                    row["context_class"] = "unknown"
                    continue
                row["land_fraction_1km"] = round(fractions["land"], 3)
                row["water_fraction_1km"] = round(fractions["water"], 3)
                row["shoreline_fraction_1km"] = round(fractions["shoreline"], 3)
                row["context_class"] = fractions["context_class"]
    return rows


def distance_m(row_a, row_b):
    return math.hypot(row_a["centroid_easting"] - row_b["centroid_easting"],
                      row_a["centroid_northing"] - row_b["centroid_northing"])


def nearest_train_distance(row, rows):
    others = [distance_m(row, other) for other in rows if other is not row]
    return min(others) if others else float("nan")


def strata_by_area(rows, strata=STRATA):
    """Five groups of near-equal size, ordered smallest to largest area."""
    ordered = sorted(rows, key=lambda r: (r["component_area_px_20m"],
                                          r["candidate_id"]))
    groups = []
    for index in range(strata):
        start = round(index * len(ordered) / strata)
        stop = round((index + 1) * len(ordered) / strata)
        groups.append(ordered[start:stop])
    return [group for group in groups if group]


def size_targets(rows, per_pair=PER_PAIR):
    """Component-area targets spanning the smallest to the largest candidate."""
    areas = np.array([row["component_area_px_20m"] for row in rows], dtype=float)
    if areas.size <= per_pair:
        return sorted(areas.tolist())
    percentiles = np.linspace(0, 100, per_pair)
    return [float(np.percentile(areas, p)) for p in percentiles]


def select(rows, per_pair=PER_PAIR, neighbours=5):
    """Diversity-first selection: size targets, then farthest-point inside each."""
    chosen = []
    targets = size_targets(rows, per_pair)
    for index, target in enumerate(targets, start=1):
        # The two extreme targets take the candidate closest to the extreme, so
        # the shortlist really contains the smallest and the largest. Middle
        # targets look at a few candidates and pick for spatial spread instead.
        window = 1 if index in (1, len(targets)) else neighbours
        nearest_by_area = sorted(
            rows, key=lambda r: (abs(r["component_area_px_20m"] - target),
                                 r["candidate_id"]))[:window]
        if chosen:
            scored = [(min(distance_m(candidate, other) for other in chosen), candidate)
                      for candidate in nearest_by_area]
        else:
            # Nothing selected yet: prefer the most isolated candidate overall, so
            # the tour does not start inside a cluster.
            scored = [(nearest_train_distance(candidate, rows), candidate)
                      for candidate in nearest_by_area]
        scored.sort(key=lambda item: (-item[0], item[1]["candidate_id"]))
        spread, pick = scored[0]
        pick["area_stratum"] = index
        pick["selection_reason"] = (
            "size target %d/%d at %d px: closest candidates by area, took the one "
            "%.0f m from the nearest of those already selected"
            % (index, len(targets), round(target), spread))
        chosen.append(pick)
    return chosen


def navigation_order(rows):
    """Greedy nearest-neighbour tour starting from the westernmost candidate."""
    if not rows:
        return []
    remaining = list(rows)
    current = min(remaining, key=lambda r: (r["centroid_easting"], r["candidate_id"]))
    order = [current]
    remaining.remove(current)
    while remaining:
        current = min(remaining, key=lambda r: (distance_m(order[-1], r),
                                                r["candidate_id"]))
        order.append(current)
        remaining.remove(current)
    return order


def shortlist_rows(rows):
    out = []
    for row in rows:
        out.append({
            "nav_order": row["nav_order"],
            "candidate_id": row["candidate_id"],
            "pair": row["pair"],
            "partition": row["partition"],
            "size_rank": row["area_stratum"],
            "centroid_lon": row["centroid_lon"],
            "centroid_lat": row["centroid_lat"],
            "centroid_easting": row["centroid_easting"],
            "centroid_northing": row["centroid_northing"],
            "bbox_width_m": row["bbox_width_m"],
            "bbox_height_m": row["bbox_height_m"],
            "bbox_area_m2": row["bbox_area_m2"],
            "component_area_px_20m": row["component_area_px_20m"],
            "component_area_m2": row["component_area_m2"],
            "mean_score": row["mean_score"],
            "max_score": row["max_score"],
            "land_fraction_1km": row["land_fraction_1km"],
            "water_fraction_1km": row["water_fraction_1km"],
            "shoreline_fraction_1km": row["shoreline_fraction_1km"],
            "context_class": row["context_class"],
            "nearest_train_candidate_m": round(nearest_train_distance(row, rows), 1)
            if len(rows) > 1 else "",
            "selection_reason": row["selection_reason"],
        })
    return out


def sheet_rows(rows):
    out = []
    for row in rows:
        out.append({
            "nav_order": row["nav_order"],
            "candidate_id": row["candidate_id"],
            "pair": row["pair"],
            "partition": row["partition"],
            "centroid_lon": row["centroid_lon"],
            "centroid_lat": row["centroid_lat"],
            "bbox_width_m": row["bbox_width_m"],
            "bbox_height_m": row["bbox_height_m"],
            "component_area_m2": row["component_area_m2"],
            "max_score": row["max_score"],
            "interpretability": "",
            "apparent_class": "",
            "suitable_for_polygon": "",
            "reason": "",
            "evidence": "",
            "notes": "",
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=QUEUE)
    parser.add_argument("--water-dir", default=WATER_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    shortlist_path = os.path.join(args.output_dir, "calibration_shortlist.csv")
    sheet_path = os.path.join(args.output_dir, "review_sheet.csv")
    json_path = os.path.join(args.output_dir, "calibration_summary.json")
    for path in (shortlist_path, sheet_path, json_path):
        if os.path.exists(path):
            parser.error("refusing to overwrite %s" % path)

    rows = load_queue(args.queue)
    eligible = [row for row in rows
                if row["partition"] == "train" and row["touches_guard_band"] != "True"]
    annotate_context(eligible, args.water_dir)
    print("eligible train candidates:", len(eligible))

    chosen = []
    nav = 0
    for pair in ("A", "B"):
        pool = [row for row in eligible if row["pair"] == pair]
        picked = select(pool)
        for row in navigation_order(picked):
            nav += 1
            row["nav_order"] = nav
            chosen.append(row)
        print("%s: %d selected, area %d-%d px, max_score %.3f-%.3f, contexts %s"
              % (pair, len(picked),
                 min(r["component_area_px_20m"] for r in picked),
                 max(r["component_area_px_20m"] for r in picked),
                 min(r["max_score"] for r in picked),
                 max(r["max_score"] for r in picked),
                 sorted({r["context_class"] for r in picked})))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(shortlist_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHORTLIST_FIELDS)
        writer.writeheader()
        writer.writerows(shortlist_rows(chosen))
    with open(sheet_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(sheet_rows(chosen))

    summary = {
        "candidates": len(chosen),
        "eligible_train_candidates": len(eligible),
        "per_pair": PER_PAIR,
        "strata_per_pair": STRATA,
        "context_radius_m": CONTEXT_RADIUS_PX * 20,
        "selected": [{"nav_order": r["nav_order"], "candidate_id": r["candidate_id"],
                      "pair": r["pair"], "area_px": r["component_area_px_20m"],
                      "max_score": r["max_score"], "context": r["context_class"]}
                     for r in chosen],
        "status": "calibration_shortlist_not_labels",
        "decisions_recorded": False,
    }
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=1)

    print("\nwrote", shortlist_path)
    print("wrote", sheet_path)
    print("wrote", json_path)


if __name__ == "__main__":
    main()
