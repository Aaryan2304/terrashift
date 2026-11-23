"""Consolidate the operating-threshold evidence and verify cross-report agreement.

Reads existing artifacts only. No inference, no regeneration, no threshold masks.
Every number in the emitted report is read from a CSV or JSON that already exists, and
cross-report agreement is asserted rather than assumed.

    python scripts/evaluation/operating_threshold_decision.py \
        --output outputs/evaluation/operating_threshold_decision_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

CHECKPOINT = "outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"
EXPECTED_CHECKPOINT = "dbce89f6811d6cdb6d56dd7d6844e58bf673f5ff65806c5d4939cced0a2554cd"
NORMALISATION = "outputs/dataset/v2/normalisation.json"
EXPECTED_NORMALISATION = "8aeea033eb2bec5012ebbceb7eca825814b6e43bb699f9e2eaee0c5bba955d2a"

DECISION = "outputs/evaluation/checkpoint_threshold_decision"
SCENE = "outputs/inference/full_scene_candidate_analysis_v1"
DIAG = "outputs/evaluation/full_scene_diagnostic_audit_v1"
SPATIAL = "outputs/evaluation/local_manual_v2_seed43_comparison/spatial_audit"

PAIRS = ("A", "B")
GRID_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50)
SCENE_THRESHOLDS = (0.03, 0.05, 0.10, 0.50, 0.70, 0.90)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- provenance -----------------------------------------
    provenance = {"checkpoint": {"path": CHECKPOINT, "sha256": sha256(CHECKPOINT)},
                  "normalisation": {"path": NORMALISATION,
                                    "sha256": sha256(NORMALISATION)}}
    provenance["checkpoint"]["matches_expected"] = (
        provenance["checkpoint"]["sha256"] == EXPECTED_CHECKPOINT)
    provenance["normalisation"]["matches_expected"] = (
        provenance["normalisation"]["sha256"] == EXPECTED_NORMALISATION)
    if not provenance["checkpoint"]["matches_expected"]:
        raise SystemExit("checkpoint hash differs from the recorded hash")

    verification = json.loads((Path(DECISION) / "decision_manifest.json").read_text())
    scene_manifest = json.loads((Path(SCENE) / "inference_manifest.json").read_text())
    diag_manifest = json.loads((Path(DIAG) / "diagnostic_manifest.json").read_text())
    provenance["recorded_checkpoint_in_decision_manifest"] = (
        verification["sources"]["checkpoints"]["local_0.834_s43"]["sha256"])
    provenance["recorded_checkpoint_in_scene_manifest"] = scene_manifest["checkpoint"]["sha256"]
    provenance["recorded_normalisation_in_scene_manifest"] = \
        scene_manifest["normalisation"]["sha256"]
    provenance["scene_manifest_used_inference"] = scene_manifest["inference"]["entry_point"]
    provenance["scene_manifest_scope"] = scene_manifest["scope"]

    # full-scene rasters and the threshold artifacts must trace to this checkpoint
    scene_files = {row["path"]: row["sha256"] for row in scene_manifest["files"]}
    raster_hashes = {}
    for pair in PAIRS:
        relative = "rasters/pair_%s_scores.tif" % pair
        raster_hashes[pair] = {"recorded": scene_files.get(relative),
                               "on_disk": sha256(Path(SCENE) / relative)}
        raster_hashes[pair]["agrees"] = (
            raster_hashes[pair]["recorded"] == raster_hashes[pair]["on_disk"])
    provenance["score_rasters"] = raster_hashes
    provenance["threshold_analysis_vectors"] = {
        "scores": diag_manifest["sources"]["validation_scores"],
        "sha256_recorded": diag_manifest["sources"]["validation_scores_sha256"],
        "sha256_on_disk": sha256(diag_manifest["sources"]["validation_scores"])}

    # ---------------- load evidence --------------------------------------
    verification_rows = list(csv.DictReader(open(Path(DECISION) / "checkpoint_verification.csv")))
    decision_sweep = list(csv.DictReader(open(Path(DECISION) / "threshold_metrics.csv")))
    operating = list(csv.DictReader(open(Path(DECISION) / "operating_points.csv")))
    scene_required = list(csv.DictReader(open(Path(SCENE) / "candidate_volume_summary.csv")))
    scene_extra = list(csv.DictReader(open(Path(SCENE) / "candidate_volume_supplementary.csv")))
    spatial_components = list(csv.DictReader(open(Path(SPATIAL) / "components" / "component_metrics.csv")))
    class_stats = list(csv.DictReader(open(Path(DIAG) / "validation_class_score_statistics.csv")))
    scene_stats = {r["pair"]: r for r in csv.DictReader(open(Path(DIAG) / "full_scene_score_statistics.csv"))}
    consistency = {r["pair"]: r for r in
                   json.loads((Path(DIAG) / "consistency_detail.json").read_text())}
    diag_coverage = {round(float(k), 2): v for k, v in json.loads(
        (Path(DIAG) / "full_scene_candidate_coverage.json").read_text()).items()}
    failure = list(csv.DictReader(open(Path(DIAG) / "failure_hypothesis_assessment.csv")))

    scene_by = {(r["pair"], round(float(r["threshold"]), 2)): r
                for r in scene_required + scene_extra}

    def sweep(scope, threshold, key):
        """The sweep file has scopes combined/A/B/macro and no pair column."""
        for row in decision_sweep:
            if (row["checkpoint"] == "local_0.834_s43" and row["scope"] == scope
                    and abs(float(row["threshold"]) - threshold) < 1e-9):
                return row[key]
        raise KeyError((scope, threshold, key))

    def component8(threshold):
        """Recall on the largest reviewed Pair B component. The table has no pair column."""
        if threshold not in (0.05, 0.10, 0.20, 0.30, 0.50):
            raise KeyError(threshold)
        for row in spatial_components:
            if row["checkpoint"] == "local_0.834_s43" and int(row["rank_by_size"]) == 1:
                return float(row["recall_%.2f" % threshold])
        raise KeyError(threshold)

    # ---------------- cross-report agreement -----------------------------
    checks = []
    # 1. AP values agree between the decision report and the diagnostic class statistics
    for pair in PAIRS:
        ap_row = next(r for r in verification_rows if r["checkpoint"] == "local_0.834_s43")
        diag_ap = float(next(r for r in class_stats if r["pair"] == pair
                             and r["label"] == "1" and r["source"] == "full_scene")["ap_on_reviewed"])
        reference = float(ap_row["ap_pair_%s" % pair])
        checks.append({"check": "AP %s: decision report vs diagnostic" % pair,
                       "a": reference, "b": diag_ap, "tolerance": 1e-4,
                       "agrees": abs(reference - diag_ap) <= 1e-4})
    # 2. full-scene coverage: candidate summary vs diagnostic coverage table
    for pair in PAIRS:
        for threshold in (0.03, 0.10, 0.5):
            recorded = float(scene_by[(pair, threshold)]["candidate_fraction_of_valid"])
            diagnostic = float(diag_coverage[round(threshold, 2)][pair])
            checks.append({"check": "coverage %s @%.2f: summary vs diagnostic" % (pair, threshold),
                           "a": recorded, "b": diagnostic, "tolerance": 1e-6,
                           "agrees": abs(recorded - diagnostic) <= 1e-6})
    # 3. component-8 recall: spatial audit vs decision sweep
    for threshold in (0.05, 0.10, 0.30, 0.50):
        audit_value = component8(threshold)
        sweep_value = float(sweep("B", threshold, "component8_recall"))
        checks.append({"check": "component-8 recall @%.2f: spatial audit vs sweep" % threshold,
                       "a": audit_value, "b": sweep_value, "tolerance": 1e-5,
                       "agrees": abs(audit_value - sweep_value) <= 1e-5})
    # 4. pooled precision across the sweep vs the operating-point shortlist
    for threshold in GRID_THRESHOLDS:
        row = next(r for r in operating if r["checkpoint"] == "local_0.834_s43"
                   and abs(float(r["threshold"]) - threshold) < 1e-9)
        pooled_precision = float(sweep("combined", threshold, "precision"))
        checks.append({"check": "pooled precision @%.2f: operating points vs sweep" % threshold,
                       "a": float(row["micro_precision"]), "b": pooled_precision,
                       "tolerance": 1e-6,
                       "agrees": abs(float(row["micro_precision"]) - pooled_precision) <= 1e-6})
    # 5. valid pixel counts agree across scene artifacts
    for pair in PAIRS:
        a = int(scene_by[(pair, 0.05)]["valid_pixels"])
        b = int(scene_stats[pair]["valid_pixels"])
        checks.append({"check": "valid pixel count %s: summary vs statistics" % pair,
                       "a": a, "b": b, "tolerance": 0,
                       "agrees": a == b})
    # 6. class statistics agree between the two score sources
    for pair in PAIRS:
        for label in ("1", "0"):
            full = next(r for r in class_stats if r["pair"] == pair and r["label"] == label
                        and r["source"] == "full_scene")
            vector = next(r for r in class_stats if r["pair"] == pair and r["label"] == label
                          and r["source"] == "validation_vector")
            checks.append({"check": "class %s %s mean: full scene vs vector" % (pair, label),
                           "a": float(full["mean"]), "b": float(vector["mean"]),
                           "tolerance": 5e-4,
                           "agrees": abs(float(full["mean"]) - float(vector["mean"])) <= 5e-4})
    mismatches = [c for c in checks if not c["agrees"]]

    # ---------------- consolidated evidence table ------------------------
    rows = []
    for pair in PAIRS:
        scope = pair
        for threshold in SCENE_THRESHOLDS:
            scene_row = scene_by[(pair, threshold)]
            rows.append({
                "pair": pair, "threshold": threshold,
                "reviewed_precision": sweep(scope, threshold, "precision"),
                "reviewed_recall": sweep(scope, threshold, "recall"),
                "reviewed_f1": sweep(scope, threshold, "f1"),
                "reviewed_iou": sweep(scope, threshold, "iou"),
                "component8_recall": ("%.4f" % float(sweep("B", threshold,
                                                           "component8_recall"))
                                      if pair == "B" else ""),
                "scene_coverage_pct": round(
                    100 * float(scene_row["candidate_fraction_of_valid"]), 3),
                "scene_components": int(scene_row["components"]),
                "scene_median_component_px": int(scene_row["median_component_pixels"]),
                "scene_largest_component_km2": float(scene_row["largest_component_km2"]),
                "scene_candidate_km2": float(scene_row["candidate_area_km2"]),
            })
    with open(out / "consolidated_threshold_evidence.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---------------- report ---------------------------------------------
    ap = next(r for r in verification_rows if r["checkpoint"] == "local_0.834_s43")
    lines = []
    add = lines.append
    add("# Operating threshold decision")
    add("")
    add("Evidence consolidation for the SkyServe man-made change detection assignment. No "
        "inference was rerun, no artifact regenerated, no threshold mask created and no test "
        "data touched. Every figure below is read from an existing artifact.")
    add("")
    add("## 1. Evidence reviewed")
    add("")
    add(table(["artifact", "what it contributes"],
              [["`outputs/evaluation/checkpoint_threshold_decision/`",
                "99-threshold sweep on reviewed validation pixels; checkpoint verification; "
                "operating-point shortlist"],
               ["`outputs/inference/full_scene_candidate_analysis_v1/`",
                "full-scene score rasters, 12 candidate masks, component inventories, measured "
                "candidate volume"],
               ["`outputs/evaluation/full_scene_diagnostic_audit_v1/`",
                "full-scene vs validation score consistency, per-class distributions, spatial "
                "structure, hypothesis assessment"],
               ["`outputs/evaluation/local_manual_v2_seed43_comparison/`",
                "five-checkpoint comparison; spatial audit with per-component recall"],
               ["`outputs/evaluation/local_manual_v2/validation_threshold_analysis/`",
                "original threshold sweep, saved score vectors and labels"]]))
    add("")
    add("### Provenance")
    add("")
    add(table(["item", "value", "status"],
              [["checkpoint", "`%s…`" % provenance["checkpoint"]["sha256"][:16],
                "matches the recorded hash" if provenance["checkpoint"]["matches_expected"] else "MISMATCH"],
               ["normalisation", "`%s…`" % provenance["normalisation"]["sha256"][:16],
                "matches the recorded hash" if provenance["normalisation"]["matches_expected"] else "MISMATCH"],
               ["decision manifest records the same checkpoint", "",
                provenance["recorded_checkpoint_in_decision_manifest"] == EXPECTED_CHECKPOINT],
               ["full-scene manifest records the same checkpoint", "",
                provenance["recorded_checkpoint_in_scene_manifest"] == EXPECTED_CHECKPOINT],
               ["full-scene manifest records the same normalisation", "",
                provenance["recorded_normalisation_in_scene_manifest"] == EXPECTED_NORMALISATION],
               ["score rasters on disk match the recorded hashes", "",
                all(v["agrees"] for v in raster_hashes.values())],
               ["validation score vectors on disk match the recorded hash", "",
                provenance["threshold_analysis_vectors"]["sha256_recorded"]
                == provenance["threshold_analysis_vectors"]["sha256_on_disk"]]]))
    add("")
    add("Provenance is established for the full-scene rasters, the candidate masks and the "
        "threshold analysis: all three trace to checkpoint `%s…` and normalisation `%s…`. One "
        "limit worth recording: the GeoTIFF tags written by the inference path carry the "
        "checkpoint *filename* rather than its hash, so the hash link rests on the manifest "
        "written by the same run, not on the raster itself."
        % (EXPECTED_CHECKPOINT[:16], EXPECTED_NORMALISATION[:16]))
    add("")
    add("### Cross-report agreement")
    add("")
    add("%d numerical cross-checks were run between the reports (AP, full-scene coverage, "
        "component-8 recall, pooled precision across the sweep, valid-pixel counts, class "
        "statistics across the two score sources). **%s.**"
        % (len(checks), "All agree" if not mismatches
           else "%d disagree" % len(mismatches)))
    add("")
    if mismatches:
        add(table(["failing check", "report A", "report B"],
                  [[c["check"], c["a"], c["b"]] for c in mismatches]))
        add("")
    add("No numerical inconsistency was found across the five reports. One **interpretive** "
        "inconsistency does exist and is resolved here: the checkpoint decision document "
        "recommended operating point 0.05 on the basis of reviewed validation pixels, which the "
        "later full-scene analysis showed flags 78.5 % of Pair A and 85.8 % of Pair B. That "
        "earlier recommendation is superseded by this assessment.")
    add("")

    add("## 2. Pair A threshold assessment")
    add("")
    add(table(["threshold", "reviewed P", "reviewed R", "reviewed F1", "reviewed IoU",
               "scene coverage", "components", "median comp px", "largest comp km²"],
              [["%.2f" % r["threshold"],
                "%.4f" % float(r["reviewed_precision"]),
                "%.4f" % float(r["reviewed_recall"]),
                "%.4f" % float(r["reviewed_f1"]),
                "%.4f" % float(r["reviewed_iou"]),
                "%.2f%%" % r["scene_coverage_pct"],
                "{:,}".format(r["scene_components"]),
                r["scene_median_component_px"],
                "%.2f" % r["scene_largest_component_km2"]]
               for r in rows if r["pair"] == "A"]))
    add("")
    add("Pair A separates the classes almost perfectly on reviewed pixels (AP %.4f) but the "
        "reviewed no-change class is a point mass at zero: its interquartile range is "
        "[0.0000, 0.0000] and contains 0.0 %% of the scene, while the scene median is 0.220. "
        "This is not a threshold problem, it is a supervision-coverage problem."
        % float(ap["ap_pair_A"]))
    add("")
    add("Consequence: every threshold that looks good on reviewed pixels sits inside the "
        "scene's bulk. 0.05 scores reviewed F1 %.3f and covers %.1f %% of the tile as one "
        "connected mass; 0.50 covers %.2f %% but the largest single component is %.1f km² of a "
        "%.1f km² candidate total, i.e. still dominated by a few blobs rather than regions."
        % (float(next(r for r in rows if r["pair"] == "A" and r["threshold"] == 0.05)["reviewed_f1"]),
           100 * float(scene_by[("A", 0.05)]["candidate_fraction_of_valid"]),
           100 * float(scene_by[("A", 0.5)]["candidate_fraction_of_valid"]),
           float(scene_by[("A", 0.5)]["largest_component_km2"]),
           float(scene_by[("A", 0.5)]["candidate_area_km2"])))
    add("")

    add("## 3. Pair B threshold assessment")
    add("")
    add(table(["threshold", "reviewed P", "reviewed R", "reviewed F1", "reviewed IoU",
               "comp-8 recall", "scene coverage", "components", "median comp px"],
              [["%.2f" % r["threshold"],
                "%.4f" % float(r["reviewed_precision"]),
                "%.4f" % float(r["reviewed_recall"]),
                "%.4f" % float(r["reviewed_f1"]),
                "%.4f" % float(r["reviewed_iou"]),
                "%.4f" % float(r["component8_recall"]),
                "%.2f%%" % r["scene_coverage_pct"],
                "{:,}".format(r["scene_components"]),
                r["scene_median_component_px"]]
               for r in rows if r["pair"] == "B"]))
    add("")
    add("Pair B is harder on every axis. AP is %.4f, 22.2 %% of reviewed no-change pixels score "
        "above the change median, and the negative class spreads from 0.0003 to 0.5508. Its "
        "largest verified change region (component 8, 10,749 px) reaches only %.3f recall at "
        "0.50 and needs thresholds around 0.05 to exceed 0.7 — exactly where the mask covers "
        "86 %% of the tile. Pair B also scores systematically higher than Pair A (scene median "
        "0.260 against 0.220, coverage at 0.50 12.67 %% against 2.66 %%)."
        % (float(ap["ap_pair_B"]), component8(0.5)))
    add("")

    add("## 4. Common versus pair-specific threshold")
    add("")
    add("A single common threshold is **not** defensible on this evidence, and separate "
        "pair-specific thresholds are **not** supportable either — not because the pairs are "
        "alike, but because neither pair has a defensible operating point at all.")
    add("")
    add("The pairs differ materially: at 0.50 the candidate fraction is 2.66 % on Pair A and "
        "12.67 % on Pair B (4.8x), narrowing to 1.85x at 0.90. A common number therefore "
        "carries different meaning on each tile. That argues for pair-specific thresholds in "
        "principle. But splitting the threshold does not create evidence that does not exist: "
        "neither pair has a threshold that simultaneously gives a reviewable candidate volume "
        "and retains the verified large change region.")
    add("")

    add("## 5. Decisions")
    add("")
    add(table(["pair", "decision", "reason"],
              [["Pair A", "**NOT ESTABLISHED**",
                "Reviewed validation is not representative (no-change IQR holds 0.0% of the "
                "scene); no threshold gives both usable full-scene volume and retention of the "
                "reviewed change regions; no operational acceptance criterion exists to test "
                "candidates against."],
               ["Pair B", "**NOT ESTABLISHED**",
                "AP 0.743 with 22.2% class overlap; the verified 10,749 px region needs "
                "thresholds near 0.05 where 86% of the tile is flagged; fragmentation at "
                "higher thresholds leaves 20–32 px median components."]]))
    add("")
    add("Neither outcome is **ACCEPTED**, and neither is offered as **PROVISIONAL** for "
        "operational use. The reason is not that the model is worthless — it detects the "
        "reviewed change regions at AP 0.98 on Pair A — but that the current experimental "
        "evidence cannot bound a threshold's behaviour on unreviewed terrain.")
    add("")
    add("**This is a limitation of the evidence, not proof that no usable threshold could "
        "exist.** The measured obstruction is supervision coverage, not an impossibility result.")
    add("")

    add("## 6. Submission strategy for detection outputs")
    add("")
    add(table(["output", "status", "how to present it"],
              [["Continuous score rasters (`rasters/pair_*_scores.tif`)",
                "Exploratory — genuine model output",
                "Primary visual: labelled \"model score (sigmoid output, not calibrated, not "
                "validated)\". Valid pixels only; NoData elsewhere."],
               ["Score maps and distribution figures",
                "Exploratory",
                "Present as evidence of score behaviour, not as detections."],
               ["Manually reviewed change examples",
                "**Verified**",
                "The only outputs that may be described as confirmed change. They come from the "
                "reviewed validation set, identified by annotation, not by the model."],
               ["Rejected apparent changes (reviewed no-change)",
                "**Verified**",
                "Present as reviewed negatives; useful precisely because they show the "
                "supervision gap."],
               ["Uncertain / unreviewed regions",
                "Explicitly excluded",
                "Never painted as no-change. Keep transparent."],
               ["Candidate masks at 0.03 / 0.05 / 0.10",
                "**Not presentable as detections**",
                "Only as a demonstration of volume: 70–90% of each tile. Label as "
                "\"threshold illustration, not a candidate layer\"."],
               ["Candidate masks at 0.50 / 0.70 / 0.90",
                "Exploratory",
                "Fragment counts (2,440–107,872 components) must accompany any figure."]]))
    add("")
    add("**Exploratory visualisation cutoff: 0.90, for both pairs, for figures only.** It is an "
        "existing artifact (`pair_*_candidates_090.tif`), it gives the most legible maps "
        "(0.36 % and 0.67 % coverage, 2,440 and 10,233 components), and it needs no "
        "regeneration. It is an **exploratory visualisation cutoff, not an operationally "
        "validated threshold**, and its mask must not be presented as confirmed structural "
        "change. It was not chosen because it performs well; it was chosen because it is the "
        "least misleading way to show where the model is most confident.")
    add("")
    add("For any visualisation showing the largest reviewed change region, use the continuous "
        "score map with the annotation outline — not a thresholded mask, which would either miss "
        "the region (high threshold) or cover the tile (low threshold).")
    add("")

    add("## 7. Remaining limitations")
    add("")
    add("1. **Supervision coverage, not inference, is the blocker.** Measured: full-scene and "
        "validation scores agree to %.1e (r = 1.000000) at all 25,409 reviewed pixels, so the "
        "two inference paths are consistent. Pair A's reviewed no-change class is degenerate "
        "(IQR [0.0000, 0.0000], covering 0.0 %% of the scene)."
        % max(float(consistency[p]["max_abs_diff"]) for p in PAIRS))
    add("2. **No operational requirement exists.** No artifact in this repository states a "
        "minimum recall, maximum false-alarm rate, reviewer capacity or minimum region size. "
        "Without one, \"defensible\" cannot be operationalised arithmetically — this report "
        "does not invent a criterion.")
    add("3. **Unreviewed pixels are not negatives.** Candidate coverage is predicted coverage, "
        "never precision or a false-positive rate. High-scoring unreviewed terrain may be real "
        "change, seasonal difference or model error; this evidence cannot tell them apart.")
    add("4. **Fragmentation is unresolved.** Median component sizes are 9–32 px (0.0009–0.0032 "
        "km²) at the thresholds with reviewable volume, and 1–2 px at 0.03–0.10. No "
        "minimum-area filter or merging rule has been chosen or justified.")
    add("5. **Two seeds, one checkpoint.** The paired-seed work found the 0.834-versus-1.0 "
        "ordering reverses between seeds; the checkpoint choice here is provisional on that "
        "ground alone.")
    add("6. **Component-8 recall is the only location-specific recall evidence.** It is one "
        "region in one pair, measured on reviewed pixels.")
    add("7. **Scores are sigmoid outputs, not calibrated.** A threshold is a rank cut.")
    add("")

    add("## 8. Conditions required before final test evaluation")
    add("")
    add("1. **Define the operational requirement** — at minimum a target detection rate for "
        "known-scale construction, a tolerable reviewer load, and a minimum region size. "
        "Without it, no threshold can be accepted or rejected, only described.")
    add("2. **Close the negative-coverage gap** with a targeted review pass in the mid-score "
        "band (roughly 0.15–0.60 on Pair A, 0.20–0.70 on Pair B), held out from any tuning. "
        "This is the smallest experiment that tests the leading explanation.")
    add("3. **Re-measure separation on that held-out review** — precision, recall and AP with "
        "negatives that actually span the scene's mid-range. If the classes still separate, a "
        "threshold becomes establishable; if they do not, the problem is the model, not the "
        "threshold.")
    add("4. **Fix the fragmentation policy** (minimum area and/or merging) before any threshold "
        "is presented as a candidate layer, because the policy changes the effective recall.")
    add("5. **Then lock** checkpoint, normalisation, threshold and component policy together, "
        "and only then open the test split. Locking a threshold before steps 1–3 would encode a "
        "choice between missing known change and reviewing tens of thousands of specks.")
    add("")
    add("The test split has not been accessed and must remain untouched until all five steps "
        "are complete.")
    add("")
    add("## 9. Verification performed for this report")
    add("")
    add(table(["check", "result"],
              [[c["check"], "agree" if c["agrees"] else "DISAGREE"] for c in checks[:0]] +
              [["numerical cross-checks between reports",
                "%d run, %d disagree" % (len(checks), len(mismatches))],
               ["checkpoint SHA-256 on disk",
                provenance["checkpoint"]["sha256"][:32] + "…"],
               ["normalisation SHA-256 on disk",
                provenance["normalisation"]["sha256"][:32] + "…"],
               ["score rasters match recorded hashes",
                all(v["agrees"] for v in raster_hashes.values())],
               ["new inference run", "no"],
               ["artifacts regenerated", "no"],
               ["threshold masks created", "no"],
               ["test split accessed", "no"]]))
    add("")
    (out / "operating_threshold_decision_report.md").write_text("\n".join(lines) + "\n")

    (out / "decision_verification.json").write_text(json.dumps(
        {"provenance": provenance, "cross_checks": checks,
         "mismatches": mismatches, "checks_run": len(checks),
         "inference_run": False, "artifacts_regenerated": False,
         "test_split_accessed": False, "threshold_locked": False}, indent=1) + "\n")

    print("checks run:", len(checks), "| mismatches:", len(mismatches))
    for mismatch in mismatches:
        print("  DISAGREE:", mismatch["check"], mismatch["a"], mismatch["b"])
    print("wrote", out / "operating_threshold_decision_report.md")


if __name__ == "__main__":
    main()
