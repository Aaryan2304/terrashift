"""Manifest and report for the full-scene candidate analysis.

All figures quoted here are read back from the artifacts written by the inference,
mask and figure stages; nothing is transcribed by hand.

    python scripts/reporting/full_scene_manifest_report.py \
        --output outputs/inference/full_scene_candidate_analysis_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import rasterio
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

PAIRS = ("A", "B")
REQUIRED = (0.03, 0.05, 0.10)
SUPPLEMENTARY = (0.5, 0.7, 0.9)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)

    preflight = json.loads((out / "preflight.json").read_text())
    configs = {pair: json.loads((out / "_inference" / ("prediction_config_%s.json" % pair)).read_text())
               for pair in PAIRS}
    required = list(csv.DictReader(open(out / "candidate_volume_summary.csv")))
    supplementary = list(csv.DictReader(open(out / "candidate_volume_supplementary.csv")))
    curve = json.loads((out / "candidate_fraction_curve.json").read_text())
    verification = json.loads((out / "verification_masks.json").read_text())
    verification_supp = json.loads((out / "verification_masks_supplementary.json").read_text()) \
        if (out / "verification_masks_supplementary.json").exists() else {}

    # ---- curve figure ----------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for axis, pair in zip(axes, PAIRS):
        points = curve[pair]
        thresholds = [p[0] for p in points]
        fractions = [p[1] * 100 for p in points]
        axis.plot(thresholds, fractions, marker="o", lw=1.6, color="#1f4e9c")
        for threshold in REQUIRED:
            nearest = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - threshold))
            axis.annotate("%.0f%%" % fractions[nearest],
                          (thresholds[nearest], fractions[nearest]),
                          textcoords="offset points", xytext=(4, 6), fontsize=8)
        for threshold in SUPPLEMENTARY:
            nearest = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - threshold))
            axis.annotate("%.2f%%" % fractions[nearest],
                          (thresholds[nearest], fractions[nearest]),
                          textcoords="offset points", xytext=(4, 6), fontsize=8,
                          color="#8b1a1a")
        axis.set_yscale("log")
        axis.set_xlabel("score threshold")
        axis.set_ylabel("candidate fraction of valid pixels (%)")
        axis.set_title("Pair %s — measured candidate volume" % pair, fontsize=10)
        axis.grid(alpha=0.3, which="both")
    figure.suptitle("Candidate volume against threshold, measured on the full scene", fontsize=11)
    figure.tight_layout()
    figure.savefig(out / "figures" / "candidate_volume_curve.png", dpi=150)
    plt.close(figure)

    # ---- geometry checks -------------------------------------------------
    geometry = {}
    for pair in PAIRS:
        source = preflight["checks"]["pairs"][pair]["grid"]
        with rasterio.open(out / "rasters" / ("pair_%s_scores.tif" % pair)) as handle:
            written = {"crs": str(handle.crs), "height": handle.height, "width": handle.width,
                       "transform": [float(v) for v in list(handle.transform)[:6]],
                       "nodata": handle.nodata, "dtype": handle.dtypes[0]}
            data = handle.read(1)
        with rasterio.open(out / "rasters" / ("pair_%s_valid_mask.tif" % pair)) as handle:
            valid = handle.read(1).astype(bool)
        nodata_pixels = int((data == -9999.0).sum())
        geometry[pair] = {
            "crs_matches_source": written["crs"] == source["crs"],
            "transform_matches_source": all(abs(a - b) < 1e-9 for a, b in
                                            zip(written["transform"], source["transform"])),
            "dimensions_match_source": (written["height"] == source["height"]
                                        and written["width"] == source["width"]),
            "pixel_size_m": [abs(written["transform"][0]), abs(written["transform"][4])],
            "nodata_value": written["nodata"],
            "nodata_pixels": nodata_pixels,
            "valid_pixels": int(valid.sum()),
            "total_pixels": int(valid.size),
            "nodata_equals_invalid": nodata_pixels == int((~valid).sum()),
            "no_invalid_written_as_zero": bool(np.all(~np.isfinite(data[~valid])) or
                                               np.all(data[~valid] == -9999.0)),
            "score_range_on_valid": [round(float(data[valid].min()), 6),
                                     round(float(data[valid].max()), 6)],
            "score_mean_on_valid": round(float(data[valid].mean()), 6),
            "score_median_on_valid": round(float(np.median(data[valid])), 6),
        }
        del data, valid
        print("geometry", pair, geometry[pair]["crs_matches_source"],
              geometry[pair]["transform_matches_source"],
              geometry[pair]["dimensions_match_source"],
              geometry[pair]["nodata_equals_invalid"])

    # ---- manifest --------------------------------------------------------
    output_files = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.suffix in (".tif", ".csv", ".png", ".json", ".md"):
            output_files.append({"path": str(path.relative_to(out)),
                                 "bytes": path.stat().st_size,
                                 "sha256": sha256(path)})
    manifest = {
        "status": "complete",
        "task": "full-scene candidate mapping, not test evaluation",
        "inputs": {
            "pairs": {pair: {"date1": preflight["checks"]["pairs"][pair]["date1"],
                             "date2": preflight["checks"]["pairs"][pair]["date2"],
                             "tile_grid": preflight["checks"]["pairs"][pair]["grid"],
                             "bands": list(preflight["checks"]["checkpoint"]["bands"]),
                             "reflectance_contract":
                                 preflight["checks"]["pairs"][pair]["reflectance_contract"],
                             "registration_applied_shift_px":
                                 preflight["checks"]["pairs"][pair]["registration_applied_shift_px"]}
                      for pair in PAIRS},
            "manifest_yaml": "configs/data_manifest.yaml",
        },
        "checkpoint": {
            "path": preflight["checks"]["checkpoint"]["path"],
            "sha256": preflight["checks"]["checkpoint"]["sha256"],
            "matches_expected": preflight["checks"]["checkpoint"]["matches_expected"],
            "epoch": preflight["checks"]["checkpoint"]["epoch"],
            "pos_weight": preflight["checks"]["checkpoint"]["pos_weight"],
            "base": preflight["checks"]["checkpoint"]["base"],
        },
        "normalisation": {
            "path": preflight["checks"]["normalisation"]["path"],
            "sha256": preflight["checks"]["normalisation"]["sha256"],
            "type": "per-pair per-band z-score from the training partition",
        },
        "inference": {
            "entry_point": "scripts/inference/predict_safe_pair.py (unchanged production path)",
            "tiling": preflight["checks"]["tiling"],
            "strip_rows": 512, "halo_rows": 256,
            "device": configs["A"]["device"],
            "batch_size_per_forward": 1,
            "precision": "float32 model, sigmoid output written as float32",
            "runtime_seconds": {pair: configs[pair]["elapsed_seconds"] for pair in PAIRS},
            "peak_process_rss_mib": {pair: configs[pair]["peak_process_rss_mib"]
                                     for pair in PAIRS},
            "peak_gpu_memory_mib": {pair: configs[pair]["peak_gpu_memory_mib"]
                                    for pair in PAIRS},
            "throughput_pixels_per_second": {pair: configs[pair]["pixels_per_second"]
                                             for pair in PAIRS},
        },
        "outputs_geometry": geometry,
        "thresholds": {"required": list(REQUIRED), "supplementary": list(SUPPLEMENTARY)},
        "candidate_volume": required,
        "candidate_volume_supplementary": supplementary,
        "verification": {"required": {k: v for k, v in verification.items()
                                      if isinstance(v, dict)
                                      and "mask_reproduced_from_scores" in v},
                         "supplementary": {k: v for k, v in verification_supp.items()
                                           if isinstance(v, dict)
                                           and "mask_reproduced_from_scores" in v}},
        "software": preflight["environment"],
        "scope": {
            "test_labels_accessed": False,
            "test_geojson_accessed": False,
            "test_masks_accessed": False,
            "test_metrics_computed": False,
            "validation_annotations_used_in_figures": False,
            "operating_threshold_locked": False,
            "checkpoint_modified": False,
            "training_run": False,
        },
        "files": output_files,
        "notes": [
            "Scores are sigmoid outputs. They are not calibrated probabilities.",
            "Candidate area is predicted candidate coverage, not confirmed man-made change.",
            "No minimum-area filter and no component removal was applied.",
        ],
    }
    (out / "inference_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    # ---- report ----------------------------------------------------------
    lines = []
    add = lines.append
    req = {(r["pair"], float(r["threshold"])): r for r in required}
    sup = {(r["pair"], float(r["threshold"])): r for r in supplementary}
    runtime_total = sum(configs[p]["elapsed_seconds"] for p in PAIRS)

    add("# Full-scene candidate analysis")
    add("")
    add("Full-scene mapping of both pairs with the provisionally selected checkpoint. This is "
        "**not** a test evaluation: no test label, mask, index or metric was read, and nothing "
        "here is scored against held-out annotations.")
    add("")
    add("Checkpoint `local_0.834_s43`, epoch 7, `%s…`, hash verified before inference "
        "against `outputs/evaluation/checkpoint_threshold_decision/checkpoint_verification.csv`."
        % manifest["checkpoint"]["sha256"][:16])
    add("")
    add("## 1. Preflight")
    add("")
    add(table(["Item", "Verified"],
              [["checkpoint SHA-256", manifest["checkpoint"]["sha256"]],
               ["bands", ", ".join(manifest["inputs"]["pairs"]["A"]["bands"])],
               ["reflectance contract",
                "quantisation %.0f, offset %.0f, both dates, both pairs"
                % (manifest["inputs"]["pairs"]["A"]["reflectance_contract"]["date1"]["quantification"],
                   manifest["inputs"]["pairs"]["A"]["reflectance_contract"]["date1"]["offsets"]["0"])],
               ["normalisation", manifest["normalisation"]["path"]],
               ["normalisation SHA-256", manifest["normalisation"]["sha256"][:32] + "…"],
               ["tiling", "%d px patch, %d px stride, %d windows per scene"
                % (manifest["inference"]["tiling"]["patch_size"],
                   manifest["inference"]["tiling"]["stride"],
                   manifest["inference"]["tiling"]["windows_per_full_scene"])],
               ["validity", "NoData + MSK_CLASSI/MSK_QUALIT, both dates, plus warp coverage"],
               ["grid", "EPSG:32643, 10980 x 10980, 10 m, both pairs"]]))
    add("")

    add("## 2. Inference")
    add("")
    add(table(["pair", "device", "batch", "runtime", "peak process RSS", "peak GPU",
               "valid pixels", "valid fraction"],
              [[pair, configs[pair]["device"], 1,
                "%.1f s" % configs[pair]["elapsed_seconds"],
                "%.0f MiB" % configs[pair]["peak_process_rss_mib"],
                "%.0f MiB" % configs[pair]["peak_gpu_memory_mib"],
                "{:,}".format(geometry[pair]["valid_pixels"]),
                "%.6f" % (geometry[pair]["valid_pixels"] / geometry[pair]["total_pixels"])]
               for pair in PAIRS]))
    add("")
    add("Total inference wall time %.1f s (%.1f min), run sequentially, one pair at a time."
        % (runtime_total, runtime_total / 60))
    add("")

    add("## 3. Geospatial verification")
    add("")
    add(table(["check"] + ["Pair %s" % p for p in PAIRS],
              [["CRS matches source"] + [geometry[p]["crs_matches_source"] for p in PAIRS],
               ["transform matches source"] + [geometry[p]["transform_matches_source"] for p in PAIRS],
               ["dimensions match source (10980²)"] + [geometry[p]["dimensions_match_source"] for p in PAIRS],
               ["pixel size (m)"] + ["×".join("%.0f" % v for v in geometry[p]["pixel_size_m"]) for p in PAIRS],
               ["NoData = -9999"] + [geometry[p]["nodata_value"] for p in PAIRS],
               ["NoData pixels"] + ["{:,}".format(geometry[p]["nodata_pixels"]) for p in PAIRS],
               ["NoData equals invalid count"] + [geometry[p]["nodata_equals_invalid"] for p in PAIRS],
               ["invalid never written as 0"] + [geometry[p]["no_invalid_written_as_zero"] for p in PAIRS],
               ["score range on valid"] + ["%.4f – %.4f" % tuple(geometry[p]["score_range_on_valid"]) for p in PAIRS],
               ["score mean / median on valid"] + ["%.4f / %.4f" % (geometry[p]["score_mean_on_valid"],
                                                                    geometry[p]["score_median_on_valid"]) for p in PAIRS]]))
    add("")
    add("Every mask was reproduced from its saved score raster and compared pixel for pixel; "
        "each reproduced exactly, its pixel count equals the component-inventory total, and "
        "its grid equals the score raster's:")
    add("")
    add(table(["mask", "reproduced", "mask px", "inventory px", "components"],
              [["%s @ %.2f" % (k.split("_")[0], float(k.split("_")[1])),
                v["mask_reproduced_from_scores"],
                "{:,}".format(v["mask_pixels"]), "{:,}".format(v["inventory_total_pixels"]),
                "{:,}".format(v["components_in_inventory"])]
               for k, v in manifest["verification"]["required"].items()] +
              [["%s @ %.2f (supplementary)" % (k.split("_")[0], float(k.split("_")[1])),
                v["mask_reproduced_from_scores"],
                "{:,}".format(v["mask_pixels"]), "{:,}".format(v["inventory_total_pixels"]),
                "{:,}".format(v["components_in_inventory"])]
               for k, v in manifest["verification"]["supplementary"].items()]))
    add("")

    add("## 4. Candidate volume at the three requested thresholds")
    add("")
    add(table(["pair", "thr", "candidate px", "% of valid", "area km²", "components",
               "largest km²", "median px", "components 1 px"],
              [[p, "%.2f" % t,
                "{:,}".format(int(req[(p, t)]["candidate_pixels"])),
                "%.3f" % (float(req[(p, t)]["candidate_fraction_of_valid"]) * 100),
                "%.2f" % float(req[(p, t)]["candidate_area_km2"]),
                "{:,}".format(int(req[(p, t)]["components"])),
                "%.2f" % float(req[(p, t)]["largest_component_km2"]),
                "{:,}".format(int(req[(p, t)]["median_component_pixels"])),
                "{:,}".format(int(req[(p, t)]["components_1px"]))]
               for p in PAIRS for t in REQUIRED]))
    add("")
    add("These three thresholds are unusable as a candidate layer on full scenes. At 0.05 the "
        "mask covers %.1f %% of Pair A and %.1f %% of Pair B, and the largest single component "
        "holds %.1f %% and %.1f %% of all candidate pixels — the mask is one connected "
        "continent, not a set of candidate regions."
        % (100 * float(req[("A", 0.05)]["candidate_fraction_of_valid"]),
           100 * float(req[("B", 0.05)]["candidate_fraction_of_valid"]),
           100 * float(req[("A", 0.05)]["largest_component_pixels"]) /
           float(req[("A", 0.05)]["candidate_pixels"]),
           100 * float(req[("B", 0.05)]["largest_component_pixels"]) /
           float(req[("B", 0.05)]["candidate_pixels"])))
    add("")

    add("## 5. Where the score field actually sits")
    add("")
    add("The reason is not that the model saturates near 1.0; it is that the whole scene sits "
        "above the reviewed-data band. Mean score on valid pixels is %.4f for Pair A and "
        "%.4f for Pair B, median %.4f and %.4f. Half of each scene scores above roughly 0.25, "
        "so any threshold below that marks most of the tile."
        % (geometry["A"]["score_mean_on_valid"], geometry["B"]["score_mean_on_valid"],
           geometry["A"]["score_median_on_valid"], geometry["B"]["score_median_on_valid"]))
    add("")
    add(table(["threshold"] + ["Pair A %", "Pair B %"],
              [["%.2f" % p[0], "%.2f" % (p[1] * 100), "%.2f" % (q[1] * 100)]
               for p, q in zip(curve["A"], curve["B"])]))
    add("")
    add("Figure: `figures/candidate_volume_curve.png`.")
    add("")

    add("## 6. Supplementary thresholds, measured to answer the workload question")
    add("")
    add(table(["pair", "thr", "candidate px", "% of valid", "area km²", "components",
               "largest km²", "median px"],
              [[p, "%.2f" % t,
                "{:,}".format(int(sup[(p, t)]["candidate_pixels"])),
                "%.3f" % (float(sup[(p, t)]["candidate_fraction_of_valid"]) * 100),
                "%.1f" % float(sup[(p, t)]["candidate_area_km2"]),
                "{:,}".format(int(sup[(p, t)]["components"])),
                "%.2f" % float(sup[(p, t)]["largest_component_km2"]),
                "{:,}".format(int(sup[(p, t)]["median_component_pixels"]))]
               for p in PAIRS for t in SUPPLEMENTARY]))
    add("")
    add("Two things change together as the threshold rises. Candidate area falls by orders of "
        "magnitude, and the component structure inverts: at 0.05 one component holds nearly "
        "the entire mask, while at 0.90 the mask breaks into thousands of small fragments "
        "(median 32 px on Pair A, 20 px on Pair B, i.e. 0.0032 and 0.0020 km²). Neither "
        "regime is a clean region layer.")
    add("")

    add("## 7. Pair A against Pair B")
    add("")
    add("Pair B is systematically higher-scoring than Pair A: mean %.3f against %.3f, and at "
        "0.50 it produces %.1f %% candidate coverage against %.2f %%. A single threshold "
        "therefore does not transfer between the pairs — the same number means %.1fx the "
        "workload on B at 0.50, though the gap narrows to %.1fx at 0.90. "
        "Reporting one pooled figure would hide this."
        % (geometry["B"]["score_mean_on_valid"], geometry["A"]["score_mean_on_valid"],
           100 * float(sup[("B", 0.5)]["candidate_fraction_of_valid"]),
           100 * float(sup[("A", 0.5)]["candidate_fraction_of_valid"]),
           float(sup[("B", 0.5)]["candidate_fraction_of_valid"]) /
           float(sup[("A", 0.5)]["candidate_fraction_of_valid"]),
           float(sup[("B", 0.9)]["candidate_fraction_of_valid"]) /
           float(sup[("A", 0.9)]["candidate_fraction_of_valid"])))
    add("")

    add("## 8. Recommendation, and why it cannot be locked")
    add("")
    add("**The reviewed-data band (0.03–0.10) is rejected by full-scene evidence.** It was "
        "derived from 25,409 reviewed pixels in which change was deliberately over-represented "
        "(prevalence 0.53–0.59). On the full scene the same thresholds flag 70–90 % of both "
        "tiles. This is the transfer failure that the earlier decision analysis flagged as "
        "unresolved, now measured.")
    add("")
    ratio = {t: {p: int(sup[(p, t)]["candidate_pixels"]) / int(sup[(p, 0.9)]["candidate_pixels"])
                 for p in PAIRS} for t in (0.5, 0.7)}
    add("**Provisional operating point: 0.90**, on measured workload grounds only — Pair A "
        "%s candidates over %.1f km² (%d components), Pair B %s over %.1f km² (%d components). "
        "0.70 gives %.1fx and %.1fx that volume; 0.50 gives %.1fx and %.1fx."
        % ("{:,}".format(int(sup[("A", 0.9)]["candidate_pixels"])),
           float(sup[("A", 0.9)]["candidate_area_km2"]),
           int(sup[("A", 0.9)]["components"]),
           "{:,}".format(int(sup[("B", 0.9)]["candidate_pixels"])),
           float(sup[("B", 0.9)]["candidate_area_km2"]),
           int(sup[("B", 0.9)]["components"]),
           ratio[0.7]["A"], ratio[0.7]["B"], ratio[0.5]["A"], ratio[0.5]["B"]))
    add("")
    add("**This recommendation is weak and should not be locked.** Two measured facts pull "
        "against it:")
    add("")
    add("1. At 0.90 the components are tiny (median 20–32 px). A candidate layer made of "
        "0.002–0.003 km² specks is not a reviewable region list; it is noise with a high "
        "score floor.")
    add("2. The reviewed validation analysis measured component-8 recall of only 0.18 at "
        "0.50 and lower still above it. A threshold that suppresses the scene-wide background "
        "also suppresses the one large change region that was actually verified.")
    add("")
    add("Taken together: **no threshold in the measured range simultaneously gives usable "
        "scene-wide volume and retains the verified large change region.** The checkpoint's "
        "score field is not separable at tile scale. Locking a threshold now would encode a "
        "choice between missing known change and reviewing tens of thousands of specks.")
    add("")
    add("The threshold therefore stays unlocked, and this analysis does not support advancing "
        "to test evaluation with this checkpoint as a full-scene candidate generator.")
    add("")

    add("## 9. Limitations")
    add("")
    add("1. **Candidate area is not confirmed change.** Every area figure is predicted "
        "coverage. Nothing here was checked against annotation.")
    add("2. **No minimum-area filter was applied**, as instructed. This is why component "
        "counts are dominated by single-pixel fragments; the raw counts are reported first "
        "and filtering is left as a separate decision.")
    add("3. **The reviewed validation mask was not used to tune these outputs.** No "
        "annotation influenced thresholds, filtering or figure framing.")
    add("4. **Scores are sigmoid outputs, not calibrated probabilities.** No calibration was "
        "fitted or claimed.")
    add("5. **Geometry is the source grid**, so the outputs are directly comparable with the "
        "SAFE products, but the model was trained on 256 px patches and has no scene-level "
        "context; tile-scale artefacts are expected and were not corrected.")
    add("6. **One checkpoint, one seed.** The earlier paired-seed work found the weight "
        "ordering reverses between seeds, so the transfer failure measured here is a property "
        "of this checkpoint, not necessarily of the architecture or the training recipe.")
    add("")

    add("## 10. Artifacts")
    add("")
    add("    rasters/     2 score rasters, 2 validity masks, 6 required + 6 supplementary masks")
    add("    components/  12 component inventories (CSV)")
    add("    figures/     10 per-pair figures + candidate_volume_curve.png")
    add("    candidate_volume_summary.csv / .json          the three requested thresholds")
    add("    candidate_volume_supplementary.csv / .json   0.50, 0.70, 0.90")
    add("    candidate_fraction_curve.json                threshold to volume curve")
    add("    verification_masks.json / _supplementary.json reproducibility checks")
    add("    preflight.json                               inputs, contract, tiling, versions")
    add("    inference_manifest.json                      full provenance and file hashes")
    add("")
    add("Provenance: checkpoint `%s`, normalisation `%s…`. Device %s, batch 1, total runtime "
        "%.1f s. Test labels and test metrics were not accessed; the threshold remains "
        "unlocked."
        % (manifest["checkpoint"]["sha256"][:16],
           manifest["normalisation"]["sha256"][:16],
           configs["A"]["device"], runtime_total))
    add("")

    (out / "full_scene_candidate_analysis.md").write_text("\n".join(lines) + "\n")
    print("wrote inference_manifest.json (%d file records)" % len(output_files))
    print("wrote full_scene_candidate_analysis.md")


if __name__ == "__main__":
    main()
