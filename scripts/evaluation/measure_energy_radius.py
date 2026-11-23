"""Receptive-field mass and energy, measured across phases, windows and dates.

``l1`` is magnitude mass (sum of |d output / d input|), ``l2sq`` squared-gradient energy. Both
temporal branches are measured because the head consumes |f(x1) - f(x2)|. The differentiated
quantity is the logit, with no sigmoid in the path, and the radius is Chebyshev, so each radius is
a square. Windows come from the training split only, spread across positive, negative and mixed
regions, since phases inside one window are not independent samples.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.data.manifest import load_manifest, pair_paths
from terrashift.data.safeloader import SafeProduct
from terrashift.models.siamese_unet import SiameseUNet
from terrashift.data.patch_sampling import PairData, build_pool

PATCH = 128
STRIDE = 64
BANDS = ("B02", "B03", "B04", "B08")
# Interior positions whose full support fits inside a 128 px patch: the widest left
# offset is 51 and the widest right offset is 48, so [51, 79] is safe.
SAFE_FIRST, SAFE_LAST = 51, 79
WINDOWS_PER_CATEGORY = {"positive_only": 3, "negative_only": 3, "mixed": 2}
MEASURES = ("l1_date1", "l1_date2", "l1_both", "l2sq_date1", "l2sq_date2", "l2sq_both")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_positions():
    """One interior position per (row mod 8, column mod 8): 64 combinations."""
    by_phase = {phase: [i for i in range(SAFE_FIRST, SAFE_LAST + 1) if i % 8 == phase]
                for phase in range(8)}
    if any(not positions for positions in by_phase.values()):
        raise SystemExit("interior range does not cover every phase")
    return [(row_phase, col_phase, by_phase[row_phase][0], by_phase[col_phase][0])
            for row_phase in range(8) for col_phase in range(8)]


def select_windows(pair, labels_dir, partition_dir):
    """Deterministic training windows spread across the three patch categories."""
    data = PairData(pair,
                    os.path.join(labels_dir, "pair_%s_v2" % pair.lower(),
                                 "manual_labels_%s.tif" % pair),
                    os.path.join(partition_dir, "pair_%s_partition_10m.tif" % pair))
    pool = [window for window in build_pool(data, PATCH, STRIDE)
            if window["split"] == "train"]
    chosen = []
    for category, count in WINDOWS_PER_CATEGORY.items():
        members = sorted([w for w in pool if w["category"] == category],
                         key=lambda w: (-w["reviewed"], w["row0"], w["col0"]))
        chosen.extend(members[:count])
    del data
    if not chosen:
        raise SystemExit("pair %s produced no training windows" % pair)
    return chosen


def read_pair(pair, window):
    manifest = load_manifest()
    first_path, second_path = pair_paths(manifest, pair)
    first, second = SafeProduct(first_path), SafeProduct(second_path)
    rows = (window["row0"], window["row0"] + PATCH)
    cols = (window["col0"], window["col0"] + PATCH)
    stack1 = np.stack([first.read_band(b, window=(rows, cols))[0] for b in BANDS]).astype(np.float32)
    stack2 = np.stack([second.read_band(b, window=(rows, cols))[0] for b in BANDS]).astype(np.float32)
    return np.nan_to_num(stack1), np.nan_to_num(stack2)


def channel_aggregates(grad1, grad2):
    """The six spatial measures from one pair of gradient tensors.

    ``grad1`` and ``grad2`` must be ``(1, C, H, W)``: one image, all channels, both
    dates. The contract is enforced rather than assumed, because an earlier version
    summed ``axis=0`` on a ``(1, C, H, W)`` tensor, which quietly reduced over the
    batch dimension and left a ``(C, H, W)`` volume, every radius then came out
    capped at ``C``.

    Returns ``{measure: (H, W) float array}``.
    """
    g1 = np.asarray(grad1)
    g2 = np.asarray(grad2)
    if g1.ndim != 4 or g2.ndim != 4:
        raise ValueError("gradients must be (batch, channels, H, W), got %s and %s"
                         % (g1.shape, g2.shape))
    if g1.shape != g2.shape:
        raise ValueError("the two dates must have the same gradient shape, got %s and %s"
                         % (g1.shape, g2.shape))
    if g1.shape[0] != 1:
        raise ValueError("one image per call, so batch must be 1, got %d" % g1.shape[0])
    if g1.shape[1] < 1:
        raise ValueError("gradients must have at least one channel")
    a1 = np.abs(g1[0])                       # (C, H, W), channel axis is now 0
    a2 = np.abs(g2[0])
    l1_1, l1_2 = a1.sum(axis=0), a2.sum(axis=0)
    sq1, sq2 = (a1 ** 2).sum(axis=0), (a2 ** 2).sum(axis=0)
    return {"l1_date1": l1_1, "l1_date2": l1_2, "l1_both": l1_1 + l1_2,
            "l2sq_date1": sq1, "l2sq_date2": sq2, "l2sq_both": sq1 + sq2}


def cumulative_radius(mass, centre, fraction):
    rows = np.arange(mass.shape[0]).reshape(-1, 1)
    cols = np.arange(mass.shape[1]).reshape(1, -1)
    distance = np.maximum(np.abs(rows - centre[0]), np.abs(cols - centre[1]))
    total = float(mass.sum())
    if total <= 0:
        return None
    for radius in range(mass.shape[0]):
        if float(mass[distance <= radius].sum()) / total >= fraction:
            return radius
    return mass.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-dir", default=os.path.join(ROOT, "outputs", "labels"))
    parser.add_argument("--partition-dir",
                        default=os.path.join(ROOT, "outputs", "partitions_v2_observation_valid"))
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    model = SiameseUNet(in_channels=state["enc1.0.weight"].shape[1],
                        base=state["enc1.0.weight"].shape[0])
    model.load_state_dict(state, strict=True)
    model.eval()                                  # BatchNorm uses running statistics
    mean = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(checkpoint["std"], dtype=np.float32).reshape(-1, 1, 1)

    windows = {}
    for pair in ("A", "B"):
        windows[pair] = select_windows(pair, args.labels_dir, args.partition_dir)
        counts = {c: sum(1 for w in windows[pair] if w["category"] == c)
                  for c in WINDOWS_PER_CATEGORY}
        print("pair %s: %d windows %s" % (pair, len(windows[pair]), counts))

    records = []
    for pair in ("A", "B"):
        for window in windows[pair]:
            raw1, raw2 = read_pair(pair, window)
            x1 = torch.from_numpy(((raw1 - mean) / std)[None])
            x2 = torch.from_numpy(((raw2 - mean) / std)[None])
            for row_phase, col_phase, row, col in phase_positions():
                g1 = x1.clone().requires_grad_(True)
                g2 = x2.clone().requires_grad_(True)
                model.zero_grad()
                out = model(g1, g2)               # logits, no sigmoid
                out[0, 0, row, col].backward()
                measures = channel_aggregates(g1.grad.detach().numpy(),
                                              g2.grad.detach().numpy())
                for measure, array in measures.items():
                    records.append({
                        "pair": pair, "window_row": window["row0"],
                        "window_col": window["col0"], "category": window["category"],
                        "measure": measure, "row_phase": row_phase,
                        "col_phase": col_phase,
                        "support_nonzero_px": int((array > 0).sum()),
                        "r50": cumulative_radius(array, (row, col), 0.50),
                        "r90": cumulative_radius(array, (row, col), 0.90),
                        "r99": cumulative_radius(array, (row, col), 0.99),
                    })
            print("  %s r%d c%d (%s)" % (pair, window["row0"], window["col0"],
                                         window["category"]))

    def summarise(measure, pair=None):
        subset = [r for r in records
                  if r["measure"] == measure and (pair is None or r["pair"] == pair)]
        result = {}
        for key in ("r50", "r90", "r99"):
            values = [r[key] for r in subset if r[key] is not None]
            result[key] = {"min": int(min(values)), "median": float(np.median(values)),
                           "max": int(max(values)), "n": len(values)}
        return result

    report = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256(checkpoint_path),
        "model_eval_mode": True,
        "differentiated_quantity": "pre-sigmoid logit (model output, no sigmoid applied)",
        "gradients_computed_for": ["date1", "date2"],
        "input_bands_per_date": int(state["enc1.0.weight"].shape[1]),
        "branch_combination": ("the head consumes |f(x1) - f(x2)| at four scales, so "
                              "d/dx1 and d/dx2 are separate and sign-opposed; both are "
                              "measured and their magnitudes reported combined"),
        "aggregation": {
            "l1": "sum over channels of |grad|, then integrated spatially (magnitude mass)",
            "l2sq": "sum over channels of grad^2, then integrated spatially (energy)",
            "note": "l1 is not an energy; only l2sq is",
        },
        "distance": "Chebyshev (max(|dr|,|dc|)); each radius is a square",
        "cumulative_radius": ("first r at which the mass inside max(|dr|,|dc|) <= r "
                              "reaches the fraction of the total over the window"),
        "patch_size": PATCH,
        "phase_combinations": 64,
        "windows": {pair: [{"row0": w["row0"], "col0": w["col0"],
                            "category": w["category"]} for w in windows[pair]]
                    for pair in windows},
        "summaries": {measure: {"all": summarise(measure), "A": summarise(measure, "A"),
                                "B": summarise(measure, "B")} for measure in MEASURES},
        "records": records,
        "test_data_used": False,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)

    for measure in MEASURES:
        s = report["summaries"][measure]["all"]
        print("%-10s r50 %2d/%4.1f/%2d | r90 %2d/%4.1f/%2d | r99 %2d/%4.1f/%2d | n=%d"
              % (measure, s["r50"]["min"], s["r50"]["median"], s["r50"]["max"],
                 s["r90"]["min"], s["r90"]["median"], s["r90"]["max"],
                 s["r99"]["min"], s["r99"]["median"], s["r99"]["max"], s["r90"]["n"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
