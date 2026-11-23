"""Read-only: how far are reviewed validation pixels from training-labelled ground?

Distance is reported under both metrics because they answer different questions. Euclidean
(``distance_transform_edt``) is physical ground distance; Chebyshev (``distance_transform_cdt``) is
the metric the receptive field is measured in, so it decides whether a training-labelled pixel can
enter a validation window's support. The 51 px theoretical support radius is reported beside the
empirical 23 px L1 r90. No index is written, no model runs and test labels are never read.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.validation_windows import VALIDATION_CODE, select_windows, validation_target

RADII = (23, 50, 51)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-dir", default=os.path.join(ROOT, "outputs", "labels"))
    parser.add_argument("--partition-dir",
                        default=os.path.join(ROOT, "outputs",
                                             "partitions_v2_observation_valid"))
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    report = {"radii_px": list(RADII),
              "distance_metrics": ["euclidean", "chebyshev"],
              "note": "read-only; test labels never read, no index written"}
    for pair in ("A", "B"):
        with rasterio.open(os.path.join(args.labels_dir, "pair_%s_v2" % pair.lower(),
                                        "manual_labels_%s.tif" % pair)) as handle:
            labels = handle.read(1)
        with rasterio.open(os.path.join(args.partition_dir,
                                        "pair_%s_partition_10m.tif" % pair)) as handle:
            partition = handle.read(1)
        target = validation_target(labels, partition)
        reviewed = np.argwhere(target != 255)
        training_labelled = (labels != 255) & (partition == 1)
        if not training_labelled.any():
            report.setdefault("pairs", {})[pair] = {"note": "no training-labelled pixels"}
            continue

        euclidean = ndi.distance_transform_edt(~training_labelled)
        chebyshev = ndi.distance_transform_cdt(~training_labelled, metric="chessboard")
        d_e = euclidean[reviewed[:, 0], reviewed[:, 1]]
        d_c = chebyshev[reviewed[:, 0], reviewed[:, 1]]

        entry: dict = {"reviewed_validation_px": int(reviewed.shape[0])}
        for name, values in (("euclidean", d_e), ("chebyshev", d_c)):
            entry[name] = {
                "min_px": float(values.min()),
                "median_px": float(np.median(values)),
                "p10_px": float(np.percentile(values, 10)),
                "p90_px": float(np.percentile(values, 90)),
                "max_px": float(values.max()),
            }
            for radius in RADII:
                entry[name]["pct_within_%dpx" % radius] = round(
                    100 * float((values <= radius).mean()), 3)

        # Does any proposed validation window actually contain training ground?
        for patch in (128, 256):
            windows = select_windows(target, patch, patch // 2,
                                     tile_width=labels.shape[1])
            overlapping = 0
            for window in windows:
                r0, r1 = window["row0"], min(window["row0"] + patch, labels.shape[0])
                c0 = window["col0"]
                if training_labelled[r0:r1, c0:c0 + patch].any():
                    overlapping += 1
            entry["windows_patch_%d" % patch] = {
                "total": len(windows),
                "overlapping_training_label": overlapping,
            }
        report.setdefault("pairs", {})[pair] = entry
        print("pair %s: euclidean min %.0f median %.0f | chebyshev min %.0f median %.0f | "
              "within 51px E %.1f%% C %.1f%% | windows overlapping train %s"
              % (pair, d_e.min(), np.median(d_e), d_c.min(), np.median(d_c),
                 entry["euclidean"]["pct_within_51px"], entry["chebyshev"]["pct_within_51px"],
                 {k: v["overlapping_training_label"] for k, v in entry.items()
                  if k.startswith("windows_patch")}))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
