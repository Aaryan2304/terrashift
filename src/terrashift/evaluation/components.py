"""Component-level evaluation helpers shared by the evaluation scripts.

Extracted from ``scripts/evaluation/compare_quantized_validation.py`` so the eight-connected
component recall and the audited-component identity check can be imported from the
installed package. The function bodies are unchanged; only their location moved.

``KNOWN_COMPONENT`` is duplicated here rather than imported from ``quantize``: this
module must not pull the root CLI module (and therefore torch/onnxruntime) into the
import graph. A test asserts the two copies stay equal, so drift fails the suite.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from terrashift.data.validation_windows import IGNORE_VALUE

__all__ = ["KNOWN_COMPONENT", "STRUCTURE", "component_recall", "identify_known_component"]

KNOWN_COMPONENT = {"pair": "B", "component_id": 8, "pixels": 10749}


STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def component_recall(scores, labels, target, threshold, min_size=1):
    """Recall per eight-connected change component of the reviewed mask.

    Component labelling needs the 2-D mask, while the scores are one value per
    reviewed pixel, so the components are built on the raster and then indexed with
    the same ``np.nonzero`` ordering the accumulator used. The largest Pair B
    component is the 10,749 px region.
    """
    covered = target != IGNORE_VALUE
    rows, cols = np.nonzero(covered)
    if not np.array_equal(target[covered].astype(np.int64), labels):
        raise SystemExit("score vector and 2-D mask do not share one pixel ordering")
    change = covered & (target == 1)
    components, count = ndi.label(change, structure=STRUCTURE)
    component_of_pixel = components[rows, cols]
    if count == 0:
        return {"components": 0, "rows": [], "groups": {}}
    sizes = np.bincount(components.ravel(), minlength=count + 1)[1:]
    hit = scores >= threshold
    found = []
    for identifier in range(1, count + 1):
        selected = component_of_pixel == identifier
        size = int(sizes[identifier - 1])
        if size < min_size:
            continue
        found.append({"component": identifier, "pixels": size,
                     "recall": round(float(hit[selected].mean()), 6),
                     "mean_score": round(float(scores[selected].mean()), 6),
                     "median_score": round(float(np.median(scores[selected])), 6)})
    found.sort(key=lambda row: -row["pixels"])
    for rank, row in enumerate(found, start=1):
        row["rank_by_area"] = rank
    groups = {}
    for name, predicate in (("large_ge_200px", lambda size: size >= 200),
                            ("small_lt_200px", lambda size: size < 200)):
        selected = [row for row in found if predicate(row["pixels"])]
        if selected:
            pixel_weighted = (sum(row["recall"] * row["pixels"] for row in selected)
                              / sum(row["pixels"] for row in selected))
            groups[name] = {"components": len(selected),
                            "pixels": int(sum(row["pixels"] for row in selected)),
                            "pixel_weighted_recall": round(float(pixel_weighted), 6)}
    return {"components": int(count), "rows": found, "groups": groups}


def identify_known_component(rows, pair):
    """Locate the audited 10,749 px Pair B component by area, then check its label id.

    Rank is not an identity: ``rank_by_area == 1`` only means "largest". The audit
    recorded component id 8 at 10,749 px, so the area is used to find the component and
    the id is then verified against it. If the two ever disagree, the caller must report
    the mismatch rather than silently labelling a rank as component 8.
    """
    expected = KNOWN_COMPONENT
    if pair != expected["pair"]:
        return {"status": "not_applicable",
                "reason": "no audited component identity for pair %s" % pair}
    matches = [row for row in rows if row["pixels"] == expected["pixels"]]
    if len(matches) != 1:
        return {"status": "not_identified",
                "reason": "%d components match %d px"
                          % (len(matches), expected["pixels"])}
    row = matches[0]
    if row["component"] != expected["component_id"]:
        return {"status": "id_mismatch", "found_id": row["component"],
                "expected_id": expected["component_id"], "pixels": row["pixels"],
                "recall": row["recall"]}
    return {"status": "verified", "component_id": row["component"],
            "pixels": row["pixels"], "recall": row["recall"]}
