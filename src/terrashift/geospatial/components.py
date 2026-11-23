"""Component raster to GeoJSON extraction.

Extracted from ``scripts/registers/components_to_geojson.py``; geometry, CRS and property
semantics are unchanged.
"""

from __future__ import annotations





from pathlib import Path

import numpy as np

import rasterio

from rasterio import features

from rasterio.warp import transform_geom

from shapely.geometry import mapping, shape

from shapely.ops import unary_union

from terrashift.geospatial.geometry import (boundary_area_range, components, length_sensitivity,
                            skeleton_graph)


def _polygonise(component, transform):
    """Polygonise one component with the same connectivity that labelled it.

    ``rasterio.features.shapes`` defaults to 4-connectivity, while ``components``
    labels with 8. Left alone, a diagonal pair became one component of two
    pixels and two separate polygons, and the reader kept only the first, so the
    record reported 200 m2 of area against a polygon of 100 m2. Passing
    connectivity=8 and merging every returned part removes that contradiction.
    """
    parts = [shape(geometry)
             for geometry, value in features.shapes(
                 component.astype(np.uint8), mask=component, transform=transform,
                 connectivity=8)
             if int(value) == 1]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else unary_union(parts)


def extract(mask, transform, pixel_size, min_area_px=4, spur_length_px=0):
    """Vectorise a boolean mask into component records."""
    mask = np.asarray(mask, dtype=bool)
    labels, areas = components(mask)
    records = []
    for label, area in enumerate(areas, start=1):
        if area < min_area_px:
            continue
        component = labels == label
        polygon_geometry = _polygonise(component, transform)
        if polygon_geometry is None:
            continue
        area_info = boundary_area_range(component, pixel_size * pixel_size)
        length_info = skeleton_graph(component, pixel_size, spur_length_px)
        length_range = length_sensitivity(component, pixel_size,
                                          spur_length_px=spur_length_px)
        rows, cols = np.nonzero(component)
        records.append({
            "component_id": int(label),
            "area_px": int(area),
            "area_m2": area_info["nominal_m2"],
            "area_min_m2": area_info["min_m2"],
            "area_max_m2": area_info["max_m2"],
            "geometry_area_m2": round(float(polygon_geometry.area), 2),
            "skeleton_length_m": round(length_info["length_m"], 2),
            "skeleton_length_min_m": round(length_range["min_m"], 2),
            "skeleton_length_max_m": round(length_range["max_m"], 2),
            "skeleton_length_eroded_m": round(length_range["eroded_m"], 2),
            "skeleton_length_dilated_m": round(length_range["dilated_m"], 2),
            "skeleton_px": length_info["skeleton_px"],
            "skeleton_endpoints": length_info["endpoints"],
            "skeleton_junctions": length_info["junctions"],
            "geometry_class": "unclassified",
            "centroid_row": float(rows.mean()),
            "centroid_col": float(cols.mean()),
            "geometry": mapping(polygon_geometry),
        })
    return records
