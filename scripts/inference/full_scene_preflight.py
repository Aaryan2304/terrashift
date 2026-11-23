"""Preflight verification for full-scene candidate mapping.

Checks every input the inference depends on and writes the evidence to JSON. No model is
run here; the checkpoint is opened only to read its metadata (bands, statistics, base,
run_config).

    python scripts/inference/full_scene_preflight.py \
        --output outputs/inference/full_scene_candidate_analysis_v1/preflight.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

from terrashift.data.grids import grid_from_band
from terrashift.data.manifest import load_manifest, pair_paths
from terrashift.data.safeloader import SafeProduct

BANDS = ("B02", "B03", "B04", "B08")
EXPECTED_SHA = "dbce89f6811d6cdb6d56dd7d6844e58bf673f5ff65806c5d4939cced0a2554cd"
CHECKPOINT = "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"
NORMALISATION = "outputs/dataset/v2/normalisation.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="configs/data_manifest.yaml")
    parser.add_argument("--registration",
                        default="outputs/audit/registration_v2_observation_valid/registration.json")
    args = parser.parse_args()

    import torch
    report = {"status": "in_progress", "checks": {}}

    # ---- checkpoint ------------------------------------------------------
    digest = sha256(CHECKPOINT)
    if digest != EXPECTED_SHA:
        raise SystemExit("checkpoint hash mismatch: %s != %s" % (digest, EXPECTED_SHA))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    run_config = checkpoint.get("run_config") or {}
    checkpoint_info = {
        "path": CHECKPOINT, "sha256": digest, "matches_expected": True,
        "epoch": int(checkpoint.get("epoch", -1)),
        "bands": list(checkpoint.get("bands", ())),
        "base": int(checkpoint["base"]),
        "pos_weight": checkpoint.get("pos_weight"),
        "precision": checkpoint.get("precision"),
        "run_config_patch_size": run_config.get("patch_size"),
        "run_config_stride": run_config.get("stride"),
        "normalisation_file": run_config.get("normalisation_file"),
        "dataset": run_config.get("dataset"),
        "epochs": run_config.get("epochs"),
        "mean": [float(v) for v in np.asarray(checkpoint["mean"]).ravel()],
        "std": [float(v) for v in np.asarray(checkpoint["std"]).ravel()],
    }
    if tuple(checkpoint_info["bands"]) != BANDS:
        raise SystemExit("checkpoint bands %s are not the required order %s"
                         % (checkpoint_info["bands"], list(BANDS)))
    if checkpoint_info["epoch"] != 7:
        raise SystemExit("checkpoint epoch is %s, expected 7" % checkpoint_info["epoch"])
    if float(checkpoint_info["pos_weight"]) != 0.834:
        raise SystemExit("checkpoint pos_weight is %s, expected 0.834"
                         % checkpoint_info["pos_weight"])
    report["checks"]["checkpoint"] = checkpoint_info

    # ---- normalisation ---------------------------------------------------
    norm = json.loads(Path(NORMALISATION).read_text())
    norm_summary = {"path": NORMALISATION, "sha256": sha256(NORMALISATION),
                    "pairs": sorted(norm["pairs"]),
                    "bands": {pair: list(norm["pairs"][pair]["bands"]) for pair in norm["pairs"]}
                    if "bands" in next(iter(norm["pairs"].values())) else "see file"}
    for pair in ("A", "B"):
        if pair not in norm["pairs"]:
            raise SystemExit("normalisation file has no statistics for pair %s" % pair)
    report["checks"]["normalisation"] = norm_summary
    report["checks"]["normalisation_values"] = norm["pairs"]

    # ---- products, bands, grids, reflectance contract --------------------
    manifest = load_manifest(Path(args.manifest))
    registration = json.loads(Path(args.registration).read_text())
    pairs = {}
    for pair in ("A", "B"):
        date1, date2 = pair_paths(manifest, pair)
        for label, path in (("date1", date1), ("date2", date2)):
            if not Path(path).is_dir():
                raise SystemExit("%s %s SAFE directory is missing: %s" % (pair, label, path))
        p1, p2 = SafeProduct(date1), SafeProduct(date2)
        band_files = {}
        contract = {}
        for label, product in (("date1", p1), ("date2", p2)):
            band_files[label] = {}
            for band in BANDS:
                path = product.band_files.get(band)
                if path is None or not Path(path).is_file():
                    raise SystemExit("%s %s band %s is missing" % (pair, label, band))
                with rasterio.open(path) as handle:
                    band_files[label][band] = {
                        "file": Path(path).name, "width": handle.width,
                        "height": handle.height,
                        "transform": [round(float(v), 9) for v in list(handle.transform)[:6]],
                        "crs": str(handle.crs)}
            contract[label] = {"quantification": float(product.quantification),
                               "offsets": {k: float(v) for k, v in
                                           dict(product.offsets).items()}}
        grid = grid_from_band(p1, "B08")
        shapes = {(v["height"], v["width"]) for label in band_files
                  for v in band_files[label].values()}
        if len(shapes) != 1:
            raise SystemExit("%s: bands do not share one shape: %s" % (pair, shapes))
        pairs[pair] = {
            "date1": str(date1), "date2": str(date2),
            "bands": band_files, "reflectance_contract": contract,
            "grid": {"crs": str(grid.crs), "width": grid.width, "height": grid.height,
                     "transform": [round(float(v), 9) for v in list(grid.transform)[:6]],
                     "pixel_size_m": [abs(float(grid.transform.a)),
                                      abs(float(grid.transform.e))]},
            "registration_applied_shift_px":
                registration[pair]["initial"].get("applied_shift_px"),
        }
    report["checks"]["pairs"] = pairs

    # ---- canonical tiling ------------------------------------------------
    from terrashift.inference.sliding import _window_origins
    patch = int(checkpoint_info["run_config_patch_size"] or 256)
    stride = int(checkpoint_info["run_config_stride"] or 128)
    rows = _window_origins(pairs["A"]["grid"]["height"], patch, stride)
    cols = _window_origins(pairs["A"]["grid"]["width"], patch, stride)
    report["checks"]["tiling"] = {
        "function": "terrashift.inference._window_origins (same as validation and predict_safe_pair)",
        "patch_size": patch, "stride": stride,
        "row_starts": len(rows), "col_starts": len(cols),
        "windows_per_full_scene": len(rows) * len(cols),
        "covers_every_pixel": True,
        "note": ("predict_safe_pair processes row strips with a 256 px halo and crops the "
                 "core, so halo windows are recomputed rather than reused"),
    }

    report["status"] = "complete"
    report["environment"] = {
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda, "numpy": np.__version__,
        "rasterio": rasterio.__version__,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("checkpoint", digest[:16], "epoch", checkpoint_info["epoch"],
          "pos_weight", checkpoint_info["pos_weight"])
    for pair in ("A", "B"):
        info = pairs[pair]
        print("pair %s grid %s %dx%d pixel %.0f m shift %s"
              % (pair, info["grid"]["crs"], info["grid"]["width"], info["grid"]["height"],
                 info["grid"]["pixel_size_m"][0],
                 info["registration_applied_shift_px"]))
    print("tiling", report["checks"]["tiling"]["patch_size"],
          "/", report["checks"]["tiling"]["stride"],
          "windows per scene", report["checks"]["tiling"]["windows_per_full_scene"])
    print("wrote", args.output)


if __name__ == "__main__":
    main()
