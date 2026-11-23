"""Split integrity check for the locked test evaluation.

Reads metadata and counts only. Establishes, before any test metric is computed:
  * checkpoint and normalisation hashes match the frozen values,
  * the partition encodes train=1 / validation=2 / test=3 with a guard band,
  * validation and test stripes are disjoint with the documented buffer,
  * how many reviewed pixels exist in each split.

Nothing here selects a threshold, a checkpoint or a model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = ROOT / "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"
NORMALISATION = ROOT / "outputs/dataset/v2/normalisation.json"
EXPECTED_CKPT = "dbce89f6811d6cdb6d56dd7d6844e58bf673f5ff65806c5d4939cced0a2554cd"
EXPECTED_NORM = "8aeea033eb2bec5012ebbceb7eca825814b6e43bb699f9e2eaee0c5bba955d2a"
CODES = {"guard_or_buffer": 0, "train": 1, "validation": 2, "test": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    report = {"hashes": {}, "partitions": {}, "label_counts": {}}

    report["hashes"]["checkpoint"] = sha256(CHECKPOINT)
    report["hashes"]["checkpoint_expected"] = EXPECTED_CKPT
    report["hashes"]["checkpoint_matches"] = report["hashes"]["checkpoint"] == EXPECTED_CKPT
    report["hashes"]["normalisation"] = sha256(NORMALISATION)
    report["hashes"]["normalisation_expected"] = EXPECTED_NORM
    report["hashes"]["normalisation_matches"] = (
        report["hashes"]["normalisation"] == EXPECTED_NORM)
    print("checkpoint   %s  match=%s" % (report["hashes"]["checkpoint"][:16],
                                         report["hashes"]["checkpoint_matches"]))
    print("normalisation %s  match=%s" % (report["hashes"]["normalisation"][:16],
                                          report["hashes"]["normalisation_matches"]))
    if not (report["hashes"]["checkpoint_matches"] and report["hashes"]["normalisation_matches"]):
        print("HASH MISMATCH — stopping")
        (ROOT / "outputs/evaluation/split_integrity_check.json").write_text(
            json.dumps(report, indent=1) + "\n")
        return 2

    for pair in ("A", "B"):
        partition_path = ROOT / ("outputs/partitions/pair_%s_partition_10m.tif" % pair)
        labels_path = ROOT / ("outputs/labels/pair_%s_v2/manual_labels_%s.tif"
                               % (pair.lower(), pair))
        with rasterio.open(partition_path) as handle:
            partition = handle.read(1)
            profile = {"shape": list(partition.shape), "crs": str(handle.crs),
                       "transform": list(handle.transform)[:6], "dtype": str(handle.dtypes[0])}
        codes, counts = np.unique(partition, return_counts=True)
        profile["code_histogram"] = {int(c): int(n) for c, n in zip(codes, counts)}

        # Column ranges per split. The partition is a stripe split on the x axis.
        column_range = {}
        for name, code in CODES.items():
            if code == 0:
                continue
            columns = np.flatnonzero((partition == code).any(axis=0))
            column_range[name] = [int(columns.min()), int(columns.max()) + 1]
        profile["column_ranges"] = column_range
        validation_cols = set(range(*column_range["validation"]))
        test_cols = set(range(*column_range["test"]))
        profile["validation_test_column_overlap"] = len(validation_cols & test_cols)
        guard = np.flatnonzero((partition == 0).any(axis=0))
        profile["guard_column_count"] = int(guard.size)
        profile["gap_validation_to_test_columns"] = (
            column_range["test"][0] - column_range["validation"][1])
        report["partitions"][pair] = profile

        with rasterio.open(labels_path) as handle:
            labels = handle.read(1)
        if labels.shape != partition.shape:
            print("PAIR %s shape mismatch labels %s vs partition %s"
                  % (pair, labels.shape, partition.shape))
            return 3
        label_values = {int(v): int(n) for v, n in
                        zip(*np.unique(labels, return_counts=True))}
        per_split = {}
        for name, code in CODES.items():
            if code == 0 and name == "guard_or_buffer":
                selected = partition == 0
            else:
                selected = partition == code
            values = labels[selected] if selected.any() else np.array([], dtype=labels.dtype)
            per_split[name] = {
                "pixels": int(selected.sum()),
                "reviewed_0": int((values == 0).sum()),
                "reviewed_1": int((values == 1).sum()),
                "uncertain_255": int((values == 255).sum()),
                "unlabelled_other": int(values.size - (values == 0).sum()
                                        - (values == 1).sum() - (values == 255).sum()),
            }
        report["label_counts"][pair] = {"label_value_histogram": label_values,
                                        "by_split": per_split}
        print("\npair %s  shape %s  codes %s" % (pair, profile["shape"], profile["code_histogram"]))
        print("  columns: val %s  test %s  overlap %d  guard cols %d"
              % (column_range["validation"], column_range["test"],
                 profile["validation_test_column_overlap"], profile["guard_column_count"]))
        print("  label histogram: %s" % label_values)
        for name in ("train", "validation", "test"):
            row = per_split[name]
            print("  %-11s px %10d | reviewed0 %8d reviewed1 %8d | 255 %8d | other %8d"
                  % (name, row["pixels"], row["reviewed_0"], row["reviewed_1"],
                     row["uncertain_255"], row["unlabelled_other"]))

    out = ROOT / "outputs/evaluation/split_integrity_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
