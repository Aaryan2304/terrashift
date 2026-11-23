"""Context coverage of a patch lattice: which positions see a full receptive field.

Model context and registration halo are different things. An output pixel near a patch border has
part of its support outside the patch, and training pads zeros there, so its prediction rests on
invented pixels. The halo the loader reads to warp date 2 is alignment, not context, and does not
extend the receptive field.

Reports ``full``, ``partial`` and ``none`` positions along one axis and in two dimensions, then
repeats the count with the 90 %-gradient-energy radius instead of full theoretical support.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)

from terrashift.inference.checkpoints import enumerate_support

BIG = 512
# Measured over all 64 output phases on real training windows: 19..23 px,
# median 21 (outputs/audit/context_coverage/energy_radius_phases.json). The
# conservative maximum is used here so the coverage claim does not flatter itself.
ENERGY_RADIUS_90 = 23
PATCH_SIZES = (128, 256)


def phase_offsets():
    """Left/right support offsets for each output phase modulo 8, interior only."""
    support = enumerate_support(BIG)
    offsets = {}
    for index in range(96, BIG - 96):
        low, high, _ = support[index]
        offsets.setdefault(index % 8, set()).add((low - index, high - index))
    return {phase: sorted(pairs)[0] for phase, pairs in offsets.items()}


def full_context_local(size, offsets):
    """Local positions whose whole support fits inside a patch of this size."""
    positions = []
    for index in range(size):
        low, high = offsets[index % 8]
        if index + low >= 0 and index + high <= size - 1:
            positions.append(index)
    return positions


def energy_context_local(size, radius=ENERGY_RADIUS_90):
    """Local positions at least `radius` from every border."""
    return [index for index in range(size)
            if index - radius >= 0 and index + radius <= size - 1]


def coverage_1d(size, stride, full_local, extent):
    """Classify every position along one axis by the best context it can receive."""
    best = {}
    for origin in range(0, extent - size + 1, stride):
        for local in full_local:
            best[origin + local] = "full"
    for origin in range(0, extent - size + 1, stride):
        for local in range(size):
            position = origin + local
            if position not in best:
                best[position] = "partial"
    for position in range(extent):
        best.setdefault(position, "none")
    counts = {"full": 0, "partial": 0, "none": 0}
    for position in range(extent):
        counts[best[position]] += 1
    # Longest run of non-full positions, which is the strip that never sees a
    # complete field.
    longest, run = 0, 0
    for position in range(extent):
        if best[position] == "full":
            run = 0
        else:
            run += 1
            longest = max(longest, run)
    return {"extent": extent, "counts": counts, "longest_gap_px": longest,
            "full_fraction": round(counts["full"] / extent, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extent", type=int, default=4096,
                        help="length of the stripe to tile, in pixels")
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    offsets = phase_offsets()
    report = {
        "phase_offsets": {str(phase): list(pair) for phase, pair in offsets.items()},
        "extent_min": min(high - low + 1 for low, high in offsets.values()),
        "extent_max": max(high - low + 1 for low, high in offsets.values()),
        "energy_radius_90_px": ENERGY_RADIUS_90,
        "extent_tiled_px": args.extent,
        "configurations": {},
    }

    for size in PATCH_SIZES:
        stride = size // 2
        full_local = full_context_local(size, offsets)
        energy_local = energy_context_local(size)
        full_1d = coverage_1d(size, stride, full_local, args.extent)
        energy_1d = coverage_1d(size, stride, energy_local, args.extent)
        report["configurations"]["%d_%d" % (size, stride)] = {
            "patch_size": size, "stride": stride,
            "full_context_positions_per_patch_1d": len(full_local),
            "full_context_2d_per_patch": len(full_local) ** 2,
            "patch_area": size * size,
            "full_context_fraction_of_patch": round(len(full_local) ** 2 / (size * size), 4),
            "energy_positions_per_patch_1d": len(energy_local),
            "energy_2d_per_patch": len(energy_local) ** 2,
            "energy_fraction_of_patch": round(len(energy_local) ** 2 / (size * size), 4),
            "full_context_tiling": full_1d,
            "energy_context_tiling": energy_1d,
        }
        c = report["configurations"]["%d_%d" % (size, stride)]
        print("%d/%d: full-context %d positions per axis (%d of %d px, %.1f%% of the patch)"
              % (size, stride, len(full_local), len(full_local) ** 2, size * size,
                 100 * c["full_context_fraction_of_patch"]))
        print("      tiling: %.1f%% of positions get a full field, longest gap %d px"
              % (100 * full_1d["full_fraction"], full_1d["longest_gap_px"]))
        print("      at 90%% energy (%d px): %.1f%% full, longest gap %d px"
              % (ENERGY_RADIUS_90, 100 * energy_1d["full_fraction"],
                 energy_1d["longest_gap_px"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()
