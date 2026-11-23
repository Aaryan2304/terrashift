"""Measure and correct the registration of each pair.

Reads B08 at 10 m over several land windows, estimates the date 2 to date 1 translation by ECC,
applies the median shift and measures the residual on the same windows.

Each window carries the observation-validity mask, passed as ``ref_valid``/``mov_valid``, so cloud,
saturation or a detector gap cannot pull the correlation. The residual reuses the estimating windows,
so it is a consistency check rather than independent validation; that would need separate control
features and is not implemented.

Output goes to ``--output`` (default ``outputs/audit/registration_v2_observation_valid/registration.json``),
plus one panel per pair with ``--figure-dir``. Nothing is overwritten.
"""

import argparse
import json
import os

import numpy as np


from terrashift.data.manifest import load_manifest, pair_config
from terrashift.data.safeloader import SafeProduct
from terrashift.registration import (align_to_reference, coverage_mask, estimate_shift,
                           residual_rmse, summarise)
from terrashift.data.validity import window_validity

DEFAULT_OUT = "outputs/audit/registration_v2_observation_valid/registration.json"

MANIFEST = load_manifest()

# Land windows in B08 native 10 m pixels: (row_start, row_end, col_start, col_end).
# These were chosen per tile by inspecting where land is. They are not derived
# from the manifest.
WINDOWS = {
    "A": [
        (2000, 4000, 5000, 7000),
        (4000, 6000, 6000, 8000),
        (6000, 8000, 5000, 7000),
        (7000, 9000, 4000, 6000),
        (3000, 5000, 7000, 9000),
    ],
    "B": [
        (2000, 4000, 2000, 4000),
        (4000, 6000, 4000, 6000),
        (6000, 8000, 6000, 8000),
        (3000, 5000, 7000, 9000),
        (5000, 7000, 2000, 4000),
    ],
}

BAND = "B08"
PIXEL_SIZE_M = 10.0


def read_window(product, window):
    r0, r1, c0, c1 = window
    refl, _, _ = product.read_band(BAND, window=((r0, r1), (c0, c1)))
    return refl


def window_masks(product, window):
    """Observation-validity mask for one window, aligned to that window."""
    valid, detail = window_validity(product, ((window[0], window[1]),
                                              (window[2], window[3])),
                                    band=BAND)
    return valid, detail


def run_pair(name, cfg):
    p1 = SafeProduct(cfg["path_date1"])
    p2 = SafeProduct(cfg["path_date2"])

    windows = []
    for w in cfg["windows"]:
        ref = read_window(p1, w)
        mov = read_window(p2, w)
        ref_valid, ref_detail = window_masks(p1, w)
        mov_valid, mov_detail = window_masks(p2, w)
        if ref_valid.shape != ref.shape or mov_valid.shape != mov.shape:
            raise RuntimeError("validity mask shape does not match the image window %s" % (w,))
        est = estimate_shift(ref, mov, ref_valid=ref_valid, mov_valid=mov_valid)
        est["window"] = list(w)
        est["invalid_pixels_date1"] = ref_detail["total_invalid"]
        est["invalid_pixels_date2"] = mov_detail["total_invalid"]
        est["window_pixels"] = int(ref.size)
        windows.append(est)
        print("  pair %s window %s: dx=%s dy=%s corr=%s converged=%s "
              "invalid(date1)=%d invalid(date2)=%d of %d"
              % (name, w,
                 None if est["dx"] is None else round(est["dx"], 3),
                 None if est["dy"] is None else round(est["dy"], 3),
                 None if est["correlation"] is None else round(est["correlation"], 3),
                 est["converged"], ref_detail["total_invalid"],
                 mov_detail["total_invalid"], ref.size))

    summary = summarise(windows, PIXEL_SIZE_M)
    summary["windows"] = windows
    summary["validity_masked"] = True
    summary["validity_definition"] = "nodata+msk_classi+msk_qualit, window-aligned"

    # Apply the median shift and measure the residual on the same windows.
    if "median_dx_px" in summary:
        dx, dy = summary["median_dx_px"], summary["median_dy_px"]
        residuals = []
        for w in cfg["windows"]:
            ref = read_window(p1, w)
            mov = read_window(p2, w)
            ref_valid, _ = window_masks(p1, w)
            mov_valid, _ = window_masks(p2, w)
            mov_aligned = align_to_reference(mov, dx, dy)
            # Coverage of the warp, then the joint valid footprint.
            shape = ref.shape
            covered = coverage_mask(shape, -dx, -dy)
            joint = ref_valid & mov_valid & covered
            res = estimate_shift(ref, mov_aligned, ref_valid=joint)
            res["window"] = list(w)
            res["rmse"] = residual_rmse(ref, mov_aligned, valid=joint)
            res["rmse_without_validity"] = residual_rmse(ref, mov_aligned)
            res["joint_valid_pixels"] = int(joint.sum())
            residuals.append(res)
            print("  pair %s residual %s: dx=%s dy=%s corr=%s rmse=%.4f "
                  "(unmasked %.4f)"
                  % (name, w,
                     None if res["dx"] is None else round(res["dx"], 3),
                     None if res["dy"] is None else round(res["dy"], 3),
                     None if res["correlation"] is None else round(res["correlation"], 3),
                     res["rmse"], res["rmse_without_validity"]))
        ok_res = [r for r in residuals if r["converged"]]
        if ok_res:
            rdx = np.array([r["dx"] for r in ok_res])
            rdy = np.array([r["dy"] for r in ok_res])
            rrmse = np.array([r["rmse"] for r in ok_res])
            rrmse_raw = np.array([r["rmse_without_validity"] for r in ok_res])
            displacement = np.hypot(rdx, rdy)
            summary["residual"] = {
                "n_converged": len(ok_res),
                "n_failed": len(residuals) - len(ok_res),
                "median_dx_px": float(np.median(rdx)),
                "median_dy_px": float(np.median(rdy)),
                "median_dx_m": float(np.median(rdx) * PIXEL_SIZE_M),
                "median_dy_m": float(np.median(rdy) * PIXEL_SIZE_M),
                "median_displacement_px": float(np.median(displacement)),
                "p95_displacement_px": float(np.percentile(displacement, 95)),
                "median_displacement_m": float(np.median(displacement) * PIXEL_SIZE_M),
                "p95_displacement_m": float(np.percentile(displacement, 95) * PIXEL_SIZE_M),
                "rmse_mean": float(rrmse.mean()),
                "rmse_max": float(rrmse.max()),
                "rmse_mean_without_validity": float(rrmse_raw.mean()),
                "windows": residuals,
            }
        summary["applied_shift_px"] = [dx, dy]
        summary["applied_shift_m"] = [dx * PIXEL_SIZE_M, dy * PIXEL_SIZE_M]

    return summary


def write_alignment_figure(path, ref, aligned, dx, dy, name, window):
    """Save a before / aligned-after / difference panel for one window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(ref[np.isfinite(ref)], [2, 98])
    diff = np.abs(ref - aligned)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(ref, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("Date 1 (reference)")
    axes[1].imshow(aligned, cmap="gray", vmin=lo, vmax=hi)
    axes[1].set_title("Date 2, aligned (dx=%.2f, dy=%.2f px)" % (dx, dy))
    im = axes[2].imshow(diff, cmap="magma", vmin=0, vmax=float(np.nanpercentile(diff, 99)))
    axes[2].set_title("|date 1 - aligned date 2|")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Registration check, pair %s, window %s" % (name, window))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUT,
                        help="JSON path for this run; an existing file is never replaced")
    parser.add_argument("--figure-dir", default=None,
                        help="optional directory for one alignment panel per pair")
    parser.add_argument("--pairs", nargs="*", choices=["A", "B"], default=None)
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite existing registration output: %s\n"
                         "choose a new --output." % args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    result = {}
    for name in (args.pairs or sorted(MANIFEST["pairs"])):
        cfg = pair_config(MANIFEST, name)
        cfg["windows"] = WINDOWS[name]
        print("=" * 70)
        print("PAIR", name, cfg["tile"])
        result[name] = {"tile": cfg["tile"], "crs": cfg["crs"],
                        "date1": cfg["date1"], "date2": cfg["date2"],
                        "path_date1": cfg["path_date1"],
                        "path_date2": cfg["path_date2"],
                        "band": BAND, "pixel_size_m": PIXEL_SIZE_M}
        result[name]["initial"] = run_pair(name, cfg)

        if args.figure_dir:
            summary = result[name]["initial"]
            if "applied_shift_px" in summary:
                p1 = SafeProduct(cfg["path_date1"])
                p2 = SafeProduct(cfg["path_date2"])
                w = cfg["windows"][1]
                ref = read_window(p1, w)
                mov = read_window(p2, w)
                dx, dy = summary["applied_shift_px"]
                aligned = align_to_reference(mov, dx, dy)
                path = os.path.join(args.figure_dir,
                                    "registration_check_%s.png" % name)
                write_alignment_figure(path, ref, aligned, dx, dy, name, w)
                print("  wrote", path)

    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=1)
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()
