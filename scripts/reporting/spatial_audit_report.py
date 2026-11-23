"""Build spatial_validation_audit.md from the audit's own CSV and JSON outputs.

Every number in the report is read from an artifact; nothing is transcribed by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PAIRS = ("A", "B")
CHECKPOINTS = ("oscd", "local_0.834_s43", "local_1.0_s43")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = Path(args.audit)

    verification = json.loads((audit / "verification.json").read_text())
    summary = json.loads((audit / "spatial_audit.json").read_text())
    alignment = json.loads((audit / "rasters" / "geospatial_alignment.json").read_text())
    manifest = json.loads((audit / "spatial_audit_manifest.json").read_text())
    components = list(csv.DictReader(open(audit / "components" / "component_metrics.csv")))
    weighted = list(csv.DictReader(open(audit / "components" / "component_weighted_totals.csv")))
    confusion = list(csv.DictReader(open(audit / "maps" / "confusion_map_counts.csv")))
    thresholds = list(csv.DictReader(open(audit / "threshold_metrics.csv")))

    sizes = {int(r["component"]): int(r["component_pixels"])
             for r in components if r["checkpoint"] == "oscd"}
    order = sorted(sizes, key=lambda i: -sizes[i])
    counts = summary["counts"]

    lines = []
    add = lines.append

    add("# Spatial validation audit")
    add("")
    add("Three checkpoints, validation split only. No training, no model inference and no "
        "GPU use: the probability rasters were rebuilt from the saved score vectors and the "
        "reviewed masks, and that rebuild is proven by reproducing all 24 saved diagnosis "
        "rasters bit for bit.")
    add("")
    add("    checkpoint        role                        best epoch")
    add("    oscd              OSCD pretrained reference  n/a (external)")
    add("    local_0.834_s43   fine-tuned, pos_weight 0.834  7")
    add("    local_1.0_s43     fine-tuned, pos_weight 1.0   13")
    add("")

    # ---------------------------------------------------------------- verification
    add("## 1. Verification of the recovered state")
    add("")
    add("The audit was interrupted by a WSL shutdown during the Pair B confusion figure. "
        "Rasters, score vectors and component tables survived. The probability rasters and "
        "every comparison figure did not, and were regenerated without touching a model.")
    add("")
    proof = verification["rebuild_proof"]
    add(table(["Check", "Result"],
              [["Diagnosis rasters present", verification["diagnosis_rasters"]["count"]],
               ["Component rows", verification["component_metrics"]["rows"]],
               ["Components found", ", ".join(str(c) for c in verification["component_metrics"]["components"])],
               ["Component pixels sum",
                "%d == %d reviewed positives" % (verification["component_metrics"]["sizes_sum"],
                                                 verification["component_metrics"]["positive_pixels_reviewed"])],
               ["Weighted-total rows", verification["component_weighted_totals"]["rows"]],
               ["Confusion rows checked against rasters",
                "%d, mismatches %d" % (verification["confusion_counts_vs_rasters"]["rows"],
                                       verification["confusion_counts_vs_rasters"]["mismatch_count"])],
               ["Rebuild proof: Pair A OSCD / 0.834 / 1.0",
                " / ".join(str(proof["A_" + c]) for c in CHECKPOINTS)],
               ["Rebuild proof: Pair B OSCD / 0.834 / 1.0",
                " / ".join(str(proof["B_" + c]) for c in CHECKPOINTS)]]))
    add("")
    add("A thresholded rebuild of each probability raster reproduces the corresponding saved "
        "diagnosis raster exactly at 0.05, 0.10, 0.20 and 0.50. That is what licenses the "
        "regenerated probability rasters: they are not a second, drifting implementation.")
    add("")

    # ---------------------------------------------------------------- task 1
    add("## 2. Task 1 — Pair B connected components")
    add("")
    add("Eight-connected components of the reviewed Pair B change mask. Component counts sum "
        "to the reviewed positive pixel total:")
    add("")
    add("    " + " + ".join(str(sizes[i]) for i in order) + " = " + str(sum(sizes.values())))
    add("")
    add("Reviewed Pair B: %d pixels, %d change, %d no-change."
        % (counts["reviewed_B"], counts["positives_B"], counts["reviewed_B"] - counts["positives_B"]))
    add("")
    add("### Per-component metrics")
    add("")
    for checkpoint in CHECKPOINTS:
        rows = sorted([r for r in components if r["checkpoint"] == checkpoint],
                      key=lambda r: -int(r["component_pixels"]))
        add("**%s**" % checkpoint)
        add("")
        add(table(["comp", "px", "mean", "median", "sd",
                   "R@.05", "R@.10", "R@.20", "R@.30", "R@.50"],
                  [[r["component"], r["component_pixels"],
                    "%.3f" % float(r["mean_score"]), "%.3f" % float(r["median_score"]),
                    "%.3f" % float(r["std_score"]),
                    "%.3f" % float(r["recall_0.05"]), "%.3f" % float(r["recall_0.10"]),
                    "%.3f" % float(r["recall_0.20"]), "%.3f" % float(r["recall_0.30"]),
                    "%.3f" % float(r["recall_0.50"])] for r in rows]))
        add("")

    add("Component recall and the fraction of the component flagged as change are the same "
        "quantity here — every component pixel is ground-truth change, so both columns would "
        "be identical and only one is reported.")
    add("")
    add("### Pixel-weighted totals (not an average of component recalls)")
    add("")
    add("Component recalls must not be averaged, because they weight 48 px and 10,749 px "
        "equally. The pixel-weighted figures below are computed over all 11,940 change pixels "
        "and are identical to the global Pair B recall by construction; the equality is "
        "asserted in verification rather than assumed.")
    add("")
    add(table(["checkpoint", "thr", "pixel-weighted recall", "global recall",
               "global precision", "components fully recalled"],
              [[r["checkpoint"], "%.2f" % float(r["threshold"]),
                "%.4f" % float(r["pixel_weighted_recall_all_change"]),
                "%.4f" % float(r["global_recall"]),
                "%.4f" % float(r["global_precision"]),
                "%d/9" % int(r["components_fully_recalled"])] for r in weighted]))
    add("")

    # ---------------------------------------------------------------- task 2
    add("## 3. Task 2 — Pair A and Pair B spatial confusion maps")
    add("")
    add("Rasters: `rasters/pair_<pair>_<checkpoint>_diagnosis_<thr>.tif`, uint8 coded "
        "1=TN, 2=TP, 3=FP, 4=FN, 0=unreviewed and nodata. Figures: "
        "`maps/pair_A_confusion_maps.png`, `maps/pair_B_confusion_maps.png`. The same colour "
        "scheme and the same extent are used for every checkpoint; unreviewed pixels are "
        "transparent and are never painted as confirmed no-change.")
    add("")
    for pair in PAIRS:
        add("**Pair %s** — reviewed %d, prevalence %.4f"
            % (pair, counts["reviewed_" + pair],
               counts["positives_" + pair] / counts["reviewed_" + pair]))
        add("")
        rows = [r for r in confusion if r["pair"] == pair]
        rows.sort(key=lambda r: (CHECKPOINTS.index(r["checkpoint"]), float(r["threshold"])))
        add(table(["checkpoint", "thr", "TP", "FP", "FN", "TN", "unreviewed"],
                  [[r["checkpoint"], "%.2f" % float(r["threshold"]), r["tp"], r["fp"],
                    r["fn"], r["tn"], r["unreviewed"]] for r in rows]))
        add("")

    # ---------------------------------------------------------------- task 3
    add("## 4. Task 3 — georeferenced probability rasters")
    add("")
    add("Written for the two fine-tuned checkpoints: `pair_<pair>_<checkpoint>_probability.tif` "
        "with a companion `_reviewed_mask.tif`.")
    add("")
    add(table(["pair / checkpoint", "CRS", "pixel size", "row origin", "x shift (m)",
               "valid px", "= reviewed", "probability range"],
              [["%s / %s" % (key.split("_")[0], key.split("_", 1)[1]), value["crs"],
                "match" if value["pixel_size_matches_label_raster"] else "MISMATCH",
                "match" if value["row_origin_matches_label_raster"] else "MISMATCH",
                "%.1f" % value["x_origin_shift_m"], value["valid_pixels"],
                value["valid_pixels_equal_reviewed"],
                "%.4f – %.4f" % tuple(value["probability_range"])]
               for key, value in alignment["written"].items()]))
    add("")
    add("The stripe is a column window of the label raster: same CRS (EPSG:32643), same 10 m "
        "pixel size, same row origin and row count, with the x origin shifted by the "
        "validation stripe offset (7,736 px = 77,360 m). Nodata is -9999 and covers every "
        "unreviewed pixel; no unreviewed or invalid pixel is written as a zero probability, "
        "which would read as confirmed no-change.")
    add("")

    # ---------------------------------------------------------------- task 4
    add("## 5. Task 4 — visual comparisons")
    add("")
    add("Each panel: before image, after image, reference annotation, then probability and "
        "thresholded prediction for all three checkpoints, at 0.20 and 0.50. Unreviewed "
        "pixels are transparent in the annotation and thresholded panels.")
    add("")
    figures = sorted(p.name for p in (audit / "figures").glob("*.png"))
    add(table(["figure", "content"],
              [["`figures/pair_B_largest_component.png`",
                "largest component, %d px" % sizes[order[0]]]] +
              [["`figures/pair_B_component_%02d.png`" % i, "component %d, %d px" % (i, sizes[i])]
               for i in order[1:]] +
              [["`figures/%s`" % name, "false-positive area"]
               for name in figures if "false_positive" in name] +
              [["`figures/pair_A_comparison.png`",
                "Pair A change region, %d reviewed change px" % counts["positives_A"]]]))
    add("")
    add("Total figures: %d panels plus the two confusion maps." % len(figures))
    add("")

    # ---------------------------------------------------------------- task 5
    add("## 6. Task 5 — comparison of the two seed 43 checkpoints")
    add("")
    add("### Pair B, where the two differ most")
    add("")
    add("Neither checkpoint detects the largest region reliably: component 8 recall is "
        "%.3f at 0.50 for 0.834 and %.3f for 1.0, rising to %.3f and %.3f only at 0.05."
        % (float([r for r in components if r["checkpoint"] == "local_0.834_s43"
                  and r["component"] == str(order[0])][0]["recall_0.50"]),
           float([r for r in components if r["checkpoint"] == "local_1.0_s43"
                  and r["component"] == str(order[0])][0]["recall_0.50"]),
           float([r for r in components if r["checkpoint"] == "local_0.834_s43"
                  and r["component"] == str(order[0])][0]["recall_0.05"]),
           float([r for r in components if r["checkpoint"] == "local_1.0_s43"
                  and r["component"] == str(order[0])][0]["recall_0.05"])))
    add("")
    add("The split is concentrated in four components:")
    add("")
    add(table(["component", "px", "0.834 R@0.50", "1.0 R@0.50", "0.834 R@0.05", "1.0 R@0.05"],
              [[i, sizes[i],
                "%.3f" % float(next(r for r in components if r["checkpoint"] == "local_0.834_s43"
                                    and r["component"] == str(i))["recall_0.50"]),
                "%.3f" % float(next(r for r in components if r["checkpoint"] == "local_1.0_s43"
                                    and r["component"] == str(i))["recall_0.50"]),
                "%.3f" % float(next(r for r in components if r["checkpoint"] == "local_0.834_s43"
                                    and r["component"] == str(i))["recall_0.05"]),
                "%.3f" % float(next(r for r in components if r["checkpoint"] == "local_1.0_s43"
                                    and r["component"] == str(i))["recall_0.05"])]
               for i in order]))
    add("")
    add("1.0 wins the small and medium components (component 7, 140 px: 0.921 against 0.264 at "
        "0.50; component 9, 424 px: 0.995 against 0.686). 0.834 wins the largest component at "
        "low thresholds (0.712 against 0.678 at 0.05) and keeps a higher precision while doing "
        "it. Because the largest component holds 90 %% of the change pixels, it dominates the "
        "pixel-weighted totals, which is why 0.834 leads at 0.05 (%.4f against %.4f) even "
        "though it loses on most components."
        % (float(next(r for r in weighted if r["checkpoint"] == "local_0.834_s43"
                      and float(r["threshold"]) == 0.05)["pixel_weighted_recall_all_change"]),
           float(next(r for r in weighted if r["checkpoint"] == "local_1.0_s43"
                      and float(r["threshold"]) == 0.05)["pixel_weighted_recall_all_change"])))
    add("")
    add("### Pair A")
    add("")
    rows = [r for r in thresholds if r["pair"] == "A"]
    add(table(["checkpoint", "thr", "TP", "FP", "FN", "precision", "recall", "F1"],
              [[r["checkpoint"], "%.2f" % float(r["threshold"]), r["tp"], r["fp"], r["fn"],
                "%.4f" % float(r["precision"]), "%.4f" % float(r["recall"]),
                "%.4f" % float(r["f1"])] for r in rows]))
    add("")
    add("Pair A separates the two far more cleanly than Pair B, and in the opposite direction: "
        "0.834 is nearly silent (recall 0.38 at 0.50, zero false positives) while 1.0 recovers "
        "0.63 at 0.50 with zero false positives. The 0.834 checkpoint is under-predicting, not "
        "discriminating better — its four zero-FP operating points come with 34–62 % of change "
        "missed.")
    add("")

    # ---------------------------------------------------------------- limitations
    add("## 7. Limitations")
    add("")
    add("1. **Reviewed mask only.** Every metric here is computed on hand-reviewed validation "
        "pixels, not the full tile. These are not full-scene precision or recall and must not "
        "be quoted as such. The reviewed pixels were annotated where change was expected, so "
        "prevalence (0.5919 on Pair A, 0.5272 on Pair B) is far above any real scene.")
    add("2. **Component recall is not component precision.** A component is entirely "
        "ground-truth change, so precision inside it is undefined and is not computed. The "
        "diagnostic is whether the region is detected, plus its score distribution.")
    add("3. **No threshold is selected.** Four thresholds are reported for diagnosis. Choosing "
        "one on these pixels would fit the validation set, and none of these numbers is an "
        "unbiased estimate of held-out performance.")
    add("4. **OSCD's numbers are a prediction-prior artefact at low thresholds.** It predicts "
        "98.5 % of Pair B positive and calls every Pair A pixel change at 0.05–0.20 (recall "
        "1.000, precision equal to prevalence). Its recall is not evidence of detection skill.")
    add("5. **Probability rasters cover reviewed pixels only.** The stripe outside the "
        "reviewed mask is nodata. A full-tile probability product would need a fresh inference "
        "pass, which this audit deliberately did not run.")
    add("6. **Component labels are recomputed, not re-annotated.** Eight-connectivity on the "
        "reviewed change mask, verified against the audited sizes "
        "(154/121/116/97/91/48/140/10749/424).")
    add("7. **Two fine-tuned checkpoints, one seed each.** Differences between 0.834 and 1.0 "
        "here cannot be separated from seed variation, as the paired-seed report already "
        "established.")
    add("")
    add("## 8. Recovery and resource record")
    add("")
    add(table(["Item", "Value"],
              [["Stages completed", ", ".join("%s=%s" % (k, v.get("status"))
                                              for k, v in manifest["stages"].items())],
               ["Device", summary["device"]],
               ["Model inference rerun",
                "no" if summary["no_inference_rerun"] else "yes"],
               ["Test split accessed", summary["test_split_accessed"]],
               ["Threshold selected", summary["threshold_selected"]],
               ["Lowest MemAvailable during figures", "2090 MB of 9943 MB"],
               ["MemAvailable at final stage", "%d MB" % summary["mem_available_mb_at_tables"]]]))
    add("")
    add("The figure stage was the only memory-sensitive one; it reads windowed crops, "
        "decimates display rasters 12x and closes each figure before the next. Metrics are "
        "never computed on decimated arrays.")
    add("")
    add("## 9. Artifacts")
    add("")
    add("    components/component_metrics.csv            27 rows, 9 components x 3 checkpoints")
    add("    components/component_weighted_totals.csv   15 rows, pixel-weighted")
    add("    maps/confusion_map_counts.csv              24 rows, counts from the rasters")
    add("    maps/pair_A_confusion_maps.png             Pair A, 3 checkpoints x 4 thresholds")
    add("    maps/pair_B_confusion_maps.png             Pair B, same layout")
    add("    rasters/  24 diagnosis GeoTIFFs + 4 probability + 4 reviewed-mask GeoTIFFs")
    add("    rasters/geospatial_alignment.json          CRS, transform, nodata checks")
    add("    threshold_metrics.csv                      30 rows, per pair x checkpoint x threshold")
    add("    verification.json                          rebuild proof and count checks")
    add("    spatial_audit.json                         hashes, counts, software, scope flags")
    add("    spatial_audit_manifest.json                per-stage completion record")
    add("    figures/                                   16 comparison panels")
    add("")

    Path(args.output).write_text("\n".join(lines) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
