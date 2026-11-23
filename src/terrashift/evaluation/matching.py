"""Spatial association between annotation records and candidate components.

Extracted from ``scripts/audit/audit_annotations_vs_candidates.py``. The bodies are
unchanged; only the location moved. This module needs shapely-typed geometries only
as arguments, so importing it touches no data, no manifest and no raster.
"""

from __future__ import annotations

__all__ = ["CANDIDATE_COVERED_CONFIDENT", "CANDIDATE_COVERED_WEAK", "overlap_area", "match_candidates"]

CANDIDATE_COVERED_CONFIDENT = 0.90   # candidate essentially inside the annotation

CANDIDATE_COVERED_WEAK = 0.10        # below this, the two barely touch


def overlap_area(left, right):
    return float(left.intersection(right).area)


def match_candidates(annotations, candidates):
    """Spatial association between annotations and candidates.

    Returns per-annotation matches and the candidate-side best annotation, so a
    mutual-best test can be applied without assuming an id convention.
    """
    cand_index = [(row["candidate_id"], row.geometry) for _, row in candidates.iterrows()]
    per_annotation = {}
    for _, row in annotations.iterrows():
        record_id = row["record_id"]
        geometry = row.geometry
        scored = []
        for candidate_id, candidate_geometry in cand_index:
            if not geometry.intersects(candidate_geometry):
                continue
            inter = overlap_area(geometry, candidate_geometry)
            if inter <= 0:
                continue
            union = geometry.area + candidate_geometry.area - inter
            scored.append({
                "candidate_id": candidate_id,
                "intersection_m2": inter,
                "iou": inter / union if union > 0 else 0.0,
                "annotation_covered": inter / geometry.area if geometry.area else 0.0,
                "candidate_covered": (inter / candidate_geometry.area
                                      if candidate_geometry.area else 0.0),
            })
        scored.sort(key=lambda item: -item["iou"])
        per_annotation[record_id] = scored
    return per_annotation
