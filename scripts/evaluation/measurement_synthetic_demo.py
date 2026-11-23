"""Worked example of the measurement functions on a synthetic fixture.

Every polygon is built in this script. None of it is an assignment result, and none of it may be
quoted as a detection, area or road length for the SkyServe sites. The only real numbers are the
registration displacements, taken from the masked registration result; the geometry they apply to
is synthetic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, box

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terrashift.geospatial.components import extract
from terrashift.geospatial.geometry import polygon_area_sensitivity, registration_displacement

PIXEL_M = 10.0
REGISTRATION = ROOT / "outputs/audit/registration_v2_observation_valid/registration.json"
STATUS = "SYNTHETIC_FIXTURE_NOT_A_RESULT"


def fixtures():
    """Named synthetic masks at 10 m, all inside one 220 x 220 array."""
    shape = (220, 220)
    masks = {}

    rectangle = np.zeros(shape, dtype=bool)
    rectangle[20:40, 20:40] = True                     # 20x20 px = 200 m x 200 m
    masks["rectangle_200m"] = rectangle

    narrow = np.zeros(shape, dtype=bool)
    narrow[60, 20:60] = True                           # 1 px wide, 40 px long
    masks["narrow_strip_1px"] = narrow

    diagonal = np.zeros(shape, dtype=bool)
    for step in range(40):
        diagonal[100 + step, 20 + step] = True         # 1 px diagonal, 40 px
    masks["diagonal_road_1px"] = diagonal

    branch = np.zeros(shape, dtype=bool)
    branch[160, 20:120] = True                         # 1,000 m trunk
    branch[160:200, 60] = True                         # 400 m branch
    branch[160:190, 100] = True                        # 300 m branch
    masks["branching_road"] = branch

    small = np.zeros(shape, dtype=bool)
    small[200:202, 200:202] = True                     # 2x2 px
    masks["small_2x2"] = small

    return masks


def polygon_fixtures():
    rectangle = box(0.0, 0.0, 200.0, 200.0)
    narrow = box(0.0, 0.0, 400.0, 10.0)
    ring = Polygon(list(box(0.0, 0.0, 200.0, 200.0).exterior.coords),
                   [list(box(50.0, 50.0, 150.0, 150.0).exterior.coords)])
    multipart = MultiPolygon([box(0.0, 0.0, 100.0, 100.0),
                              box(300.0, 0.0, 320.0, 20.0)])
    return {"rectangle_200m": rectangle, "narrow_strip_10m": narrow,
            "ring_100m_hole": ring, "multipart_two_parts": multipart}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/audit/measurement_synthetic")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "synthetic_measurements.json"
    csv_path = output_dir / "synthetic_measurements.csv"
    if json_path.exists() or csv_path.exists():
        raise SystemExit("refusing to overwrite %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = Affine(PIXEL_M, 0.0, 500000.0, 0.0, -PIXEL_M, 3000000.0)
    registration = json.loads(REGISTRATION.read_text())
    displacement = {pair: registration_displacement(registration, pair, PIXEL_M)
                    for pair in ("A", "B")}

    rows = []
    for name, mask in fixtures().items():
        for record in extract(mask, transform, PIXEL_M, min_area_px=1):
            rows.append({
                "fixture": name,
                "kind": "raster",
                "status": STATUS,
                "area_nominal_m2": record["area_m2"],
                "area_min_m2": record["area_min_m2"],
                "area_max_m2": record["area_max_m2"],
                "geometry_area_m2": record["geometry_area_m2"],
                "skeleton_length_m": record["skeleton_length_m"],
                "skeleton_length_min_m": record["skeleton_length_min_m"],
                "skeleton_length_max_m": record["skeleton_length_max_m"],
                "skeleton_length_eroded_m": record["skeleton_length_eroded_m"],
                "skeleton_length_dilated_m": record["skeleton_length_dilated_m"],
                "geometry_class": record["geometry_class"],
                "skeleton_px": record["skeleton_px"],
                "endpoints": record["skeleton_endpoints"],
                "junctions": record["skeleton_junctions"],
                "displacement_m": PIXEL_M,
                "note": "one-pixel boundary band",
            })

    for name, geometry in polygon_fixtures().items():
        for pair, label in ((None, "quantisation_only"), ("A", "plus_pair_A_residual"),
                            ("B", "plus_pair_B_residual")):
            d = PIXEL_M if pair is None else PIXEL_M + displacement[pair]["median_m"]
            result = polygon_area_sensitivity(geometry, d)
            rows.append({
                "fixture": name,
                "kind": "polygon",
                "status": STATUS,
                "area_nominal_m2": None if result["nominal_m2"] is None else round(result["nominal_m2"], 2),
                "area_min_m2": None if result["min_m2"] is None else round(result["min_m2"], 2),
                "area_max_m2": None if result["max_m2"] is None else round(result["max_m2"], 2),
                "geometry_area_m2": round(float(geometry.area), 2),
                "skeleton_length_m": None, "skeleton_length_min_m": None,
                "skeleton_length_max_m": None, "skeleton_length_eroded_m": None,
                "skeleton_length_dilated_m": None,
                "geometry_class": "unclassified",
                "skeleton_px": None, "endpoints": None, "junctions": None,
                "displacement_m": round(d, 4),
                "note": "%s; %s" % (label, "disappears under erosion"
                                    if result["disappeared"] else "survives erosion"),
            })

    fields = ["fixture", "kind", "status", "area_nominal_m2", "area_min_m2",
              "area_max_m2", "geometry_area_m2", "skeleton_length_m", "skeleton_length_min_m",
              "skeleton_length_max_m", "skeleton_length_eroded_m",
              "skeleton_length_dilated_m", "geometry_class",
              "skeleton_px", "endpoints", "junctions", "displacement_m", "note"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": STATUS,
        "warning": ("Synthetic geometry. Not a detection, not an area measurement of "
                    "the assignment sites. The registration displacements are the "
                    "only measured inputs."),
        "grid": {"pixel_size_m": PIXEL_M, "transform": list(transform)[:6]},
        "registration_displacement": displacement,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print("%s\n" % STATUS)
    header = "%-22s %-7s %-12s %-12s %-12s %-10s"
    print(header % ("fixture", "kind", "nominal_m2", "min_m2", "max_m2", "displ_m"))
    for row in rows:
        print(header % (row["fixture"], row["kind"],
                        row["area_nominal_m2"], row["area_min_m2"], row["area_max_m2"],
                        row["displacement_m"]))
    print("\nwrote", csv_path)
    print("wrote", json_path)

    print("\nlength measurements (raster fixtures) - skeleton diagnostics only")
    length_header = "%-22s %-11s %-10s %-10s %-10s %-10s %-5s %-5s"
    print(length_header % ("fixture", "nominal_m", "env_min_m", "env_max_m",
                           "eroded_m", "dilated_m", "ends", "juncs"))
    for row in rows:
        if row["kind"] != "raster":
            continue
        print(length_header % (row["fixture"], row["skeleton_length_m"],
                               row["skeleton_length_min_m"], row["skeleton_length_max_m"],
                               row["skeleton_length_eroded_m"], row["skeleton_length_dilated_m"],
                               row["endpoints"], row["junctions"]))
    print("no component is classified as linear; these are not road lengths")
    print("\nregistration: Pair A median %.3f m, Pair B median %.3f m (measured)"
          % (displacement["A"]["median_m"], displacement["B"]["median_m"]))
    print("a one-pixel band is %.1f m; the registration term adds %.2f%% to it for A, "
          "%.2f%% for B"
          % (PIXEL_M, 100 * displacement["A"]["median_m"] / PIXEL_M,
             100 * displacement["B"]["median_m"] / PIXEL_M))
    for pair in ("A", "B"):
        if not displacement[pair]["covers_whole_tile"]:
            print("WARNING: pair %s has %s unconverged window(s) %s; the tile median "
                  "is not a local estimate there"
                  % (pair, displacement[pair]["n_failed"],
                     displacement[pair]["unconverged_windows"]))
    print("this term is an operational sensitivity choice, not a calibrated bound")


if __name__ == "__main__":
    main()
