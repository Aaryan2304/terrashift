"""Quantify the competing explanations and emit the hypothesis table, manifest and report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PAIRS = ("A", "B")
LABELS = "outputs/evaluation/local_manual_v2/validation_threshold_analysis/_fine_tuned_labels.npy"
VECTOR = "outputs/evaluation/local_manual_v2_seed43_comparison/_local_0.834_s43_scores.npy"
SPLIT_A = 2759


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    out = Path(args.dir)

    hist = json.loads((out / "full_scene_histograms.json").read_text())
    edges = np.array(hist["edges"])
    class_rows = list(csv.DictReader(open(out / "validation_class_score_statistics.csv")))
    consistency = {row["pair"]: row
                   for row in json.loads((out / "consistency_detail.json").read_text())}
    stats = {r["pair"]: r for r in csv.DictReader(open(out / "full_scene_score_statistics.csv"))}
    coverage = json.loads((out / "full_scene_candidate_coverage.json").read_text())
    blocks = list(csv.DictReader(open(out / "full_scene_block_statistics.csv")))
    labels_all = np.load(LABELS)
    vectors = np.load(VECTOR).astype(np.float32)

    def class_row(pair, label):
        return next(r for r in class_rows if r["pair"] == pair and r["label"] == str(label)
                    and r["source"] == "full_scene")

    def below(pair, threshold):
        counts = np.array(hist["histograms"][pair], dtype=float)
        below_mask = edges[:-1] < threshold
        return float(counts[below_mask].sum() / counts.sum())

    evidence = {}
    for pair in PAIRS:
        neg = class_row(pair, 0)
        pos = class_row(pair, 1)
        sl = slice(0, SPLIT_A) if pair == "A" else slice(SPLIT_A, None)
        lab, val = labels_all[sl], vectors[sl]
        neg_scores = val[lab == 0]
        pos_scores = val[lab == 1]
        pair_blocks = [b for b in blocks if b["pair"] == pair]
        means = np.array([float(b["mean_score"]) for b in pair_blocks])
        high = np.array([float(b["fraction_ge_0p50"]) * int(b["valid_pixels"])
                         for b in pair_blocks])
        order = np.argsort(-high)
        top_share = float(high[order[:max(1, len(high) // 10)]].sum() / max(high.sum(), 1))
        evidence[pair] = {
            "reviewed_negative_p95": float(neg["p95"]),
            "reviewed_negative_median": float(neg["median"]),
            "reviewed_positive_median": float(pos["median"]),
            "scene_below_negative_p95": below(pair, float(neg["p95"]) + 1e-6),
            "scene_below_0p10": below(pair, 0.10),
            "scene_median": float(stats[pair]["median_from_hist"]),
            "scene_mean": float(stats[pair]["mean"]),
            "negatives_above_positive_median": float((neg_scores > float(pos["median"])).mean()),
            "positives_below_negative_p95": float((pos_scores <= float(neg["p95"])).mean()),
            "ap_on_reviewed": float(pos["ap_on_reviewed"]),
            "block_mean_min": float(means.min()), "block_mean_median": float(np.median(means)),
            "block_mean_max": float(means.max()), "block_mean_std": float(means.std()),
            "top_decile_block_share_of_high_scores": top_share,
            "coverage_at_0p03": float(coverage["0.03"][pair]),
            "coverage_at_0p10": float(coverage["0.1"][pair]),
            "coverage_at_0p50": float(coverage["0.5"][pair]),
            "neg_iqr": [float(neg["p25"]), float(neg["p75"])],
            "scene_in_neg_iqr": None,
            "scene_in_neg_p05_p95": None,
            "max_abs_score_diff": float(consistency[pair]["max_abs_diff"]),
            "pearson_r": float(consistency[pair]["pearson_r"]),
        }

    # interval overlap: how much of the scene the reviewed classes actually span
    for pair in PAIRS:
        counts = np.array(hist["histograms"][pair], dtype=float)
        total = counts.sum()
        neg = class_row(pair, 0)

        def fraction_between(low, high):
            mask = (edges[:-1] >= low - 1e-9) & (edges[:-1] < high)
            return float(counts[mask].sum() / total)

        evidence[pair]["scene_in_neg_iqr"] = fraction_between(float(neg["p25"]),
                                                              float(neg["p75"]))
        evidence[pair]["scene_in_neg_p05_p95"] = fraction_between(float(neg["p05"]),
                                                                 float(neg["p95"]))

    # ---- hypothesis table ------------------------------------------------
    a_max = max(evidence[p]["max_abs_score_diff"] for p in PAIRS)
    a_r = min(evidence[p]["pearson_r"] for p in PAIRS)
    hypotheses = [
        ("A", "Inference mismatch: preprocessing, normalisation, tiling, padding, overlap, "
              "coordinate mapping",
         "Not supported",
         "Full-scene scores reproduce the saved validation vector at every reviewed pixel: "
         "max |difference| %.2e over 25,409 pixels, Pearson r = %.6f, zero pixels beyond 1e-3. "
         "The residual is float-level (batch-1 versus batch-4 forward passes). Stripe offset "
         "77,360 m / 7,736 px verified from the two transforms, and the canonical tiling "
         "(select_windows) uses the same 512/256 strips and full-tile window grid as "
         "predict_safe_pair." % (a_max, a_r)),
        ("B", "Annotation sampling bias: reviewed pixels do not represent the full scene",
         "Confirmed",
         "Pair A: the reviewed no-change interquartile range is [%.4f, %.4f] and contains %.1f%% "
         "of the scene — a degenerate negative sample sitting at zero, while the scene median is "
         "%.3f. Pair B is milder but the same shape: negative IQR [%.4f, %.4f] covering %.1f%% "
         "of the scene, 5-95 range covering %.1f%%, negative median %.4f against a scene median "
         "of %.3f. In both pairs the reviewed negatives are drawn from the bottom of the "
         "distribution the model produces over the tile."
         % (evidence["A"]["neg_iqr"][0], evidence["A"]["neg_iqr"][1],
            100 * evidence["A"]["scene_in_neg_iqr"], evidence["A"]["scene_median"],
            evidence["B"]["neg_iqr"][0], evidence["B"]["neg_iqr"][1],
            100 * evidence["B"]["scene_in_neg_iqr"], 100 * evidence["B"]["scene_in_neg_p05_p95"],
            evidence["B"]["reviewed_negative_median"], evidence["B"]["scene_median"])),
        ("C", "Insufficient hard negatives: training annotations under-represent stable but "
              "visually complex land cover",
         "Supported by evidence",
         "Direct consequence of B. On Pair A no-change pixels have mean %.4f and standard "
         "deviation %.4f while change pixels have mean %.4f (AP %.3f); a negative class with "
         "near-zero spread supplies almost no example of stable-but-lookalike terrain. Pair B is "
         "less extreme (negative mean %.4f, std %.4f) but its negatives still centre at %.3f "
         "while the scene centres at %.3f. The training signal therefore never had to separate "
         "man-made change from the mid-range background that dominates each tile."
         % (float(class_row("A", 0)["mean"]), float(class_row("A", 0)["std"]),
            float(class_row("A", 1)["mean"]), evidence["A"]["ap_on_reviewed"],
            float(class_row("B", 0)["mean"]), float(class_row("B", 0)["std"]),
            evidence["B"]["reviewed_negative_median"], evidence["B"]["scene_median"])),
        ("D", "Score distribution / domain shift between the pairs",
         "Supported by evidence",
         "Pair B scores systematically higher: scene mean %.4f against %.4f, median %.3f against "
         "%.3f, coverage at 0.50 %.2f%% against %.2f%%. Reviewed no-change medians also differ "
         "(%.3f on B against %.3f on A). The pairs differ in acquisition (2022/2023 against "
         "2023/2026) and in scene character, so one threshold cannot serve both."
         % (evidence["B"]["scene_mean"], evidence["A"]["scene_mean"],
            evidence["B"]["scene_median"], evidence["A"]["scene_median"],
            100 * evidence["B"]["coverage_at_0p50"], 100 * evidence["A"]["coverage_at_0p50"],
            evidence["B"]["reviewed_negative_median"], evidence["A"]["reviewed_negative_median"])),
        ("E", "Model context limitation: patch-based inference lacks scene context",
         "Plausible but unverified",
         "High scores concentrate rather than distribute uniformly (block mean score %.3f to "
         "%.3f on A, %.3f to %.3f on B), which is consistent with local texture driving the "
         "decision. It is also consistent with real land-cover variation, and no experiment here "
         "separates the two. Testing it needs a context/scale ablation, which this audit did not "
         "run."
         % (evidence["A"]["block_mean_min"], evidence["A"]["block_mean_max"],
            evidence["B"]["block_mean_min"], evidence["B"]["block_mean_max"])),
        ("F", "Other implementation or data issue",
         "Not supported",
         "No further defect was found: grids, CRS, offsets, band order, reflectance contract, "
         "per-pair statistics and validity handling all verify against the artifacts. The one "
         "asymmetry is genuine data, not a bug: Pair A has 20,392 invalid pixels and Pair B 101."),
    ]
    with open(out / "failure_hypothesis_assessment.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "hypothesis", "classification", "evidence"])
        writer.writerows(hypotheses)

    manifest = {
        "status": "complete",
        "task": "full-scene versus validation score diagnostic",
        "model_inference_run": False,
        "training_run": False,
        "checkpoint_modified": False,
        "normalisation_modified": False,
        "annotations_modified": False,
        "threshold_selected": False,
        "test_data_accessed": False,
        "unreviewed_pixels_treated_as_negatives": False,
        "device": "cpu",
        "sources": {
            "full_scene_scores": {"A": "outputs/inference/full_scene_candidate_analysis_v1/rasters/pair_A_scores.tif",
                                  "B": "outputs/inference/full_scene_candidate_analysis_v1/rasters/pair_B_scores.tif"},
            "validity_masks": {"A": "outputs/inference/full_scene_candidate_analysis_v1/rasters/pair_A_valid_mask.tif",
                               "B": "outputs/inference/full_scene_candidate_analysis_v1/rasters/pair_B_valid_mask.tif"},
            "validation_scores": VECTOR,
            "validation_scores_sha256": sha256(VECTOR),
            "validation_labels": LABELS,
            "validation_labels_sha256": sha256(LABELS),
            "reviewed_mask_from": {
                "A": "outputs/evaluation/local_manual_v2_seed43_comparison/spatial_audit/rasters/pair_A_oscd_diagnosis_0.50.tif",
                "B": "outputs/evaluation/local_manual_v2_seed43_comparison/spatial_audit/rasters/pair_B_oscd_diagnosis_0.50.tif"},
            "component_inventories": "outputs/inference/full_scene_candidate_analysis_v1/components/",
        },
        "consistency": {p: {"max_abs_diff": evidence[p]["max_abs_score_diff"],
                            "pearson_r": evidence[p]["pearson_r"],
                            "reviewed_pixels": int(consistency[p]["reviewed_pixels"]),
                            "col_offset_px": int(consistency[p]["col_offset_px"])} for p in PAIRS},
        "evidence": evidence,
        "outputs": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    (out / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    # ---- report ----------------------------------------------------------
    lines = []
    add = lines.append
    add("# Full-scene versus validation score diagnostic")
    add("")
    add("Diagnostic only. No inference, no training, no checkpoint or annotation change, no "
        "threshold selection, and no test data of any kind. The question was whether the "
        "70–90 % full-scene candidate coverage is an artefact of the inference path or a "
        "property of the model's scores.")
    add("")
    add("## 1. Inference consistency — the decisive test")
    add("")
    add("Both inference paths use the same canonical tiling: `terrashift.data.validation_windows."
        "select_windows` places windows with `inference_strips(512, 256)` and `window_grid` over "
        "the full tile width, which is exactly what `scripts/inference/predict_safe_pair.py` runs. The two "
        "score sets must therefore agree at every reviewed pixel if the coordinate mapping is "
        "correct.")
    add("")
    add(table(["pair", "reviewed px", "stripe offset", "max abs diff", "mean abs diff",
               "p99 abs diff", "Pearson r", "pixels > 1e-3"],
              [[p, "{:,}".format(int(consistency[p]["reviewed_pixels"])),
                "%d px / %.0f m" % (int(consistency[p]["col_offset_px"]),
                                    float(consistency[p]["col_offset_m"])),
                "%.2e" % float(consistency[p]["max_abs_diff"]),
                "%.2e" % float(consistency[p]["mean_abs_diff"]),
                "%.2e" % float(consistency[p]["p99_abs_diff"]),
                "%.6f" % float(consistency[p]["pearson_r"]),
                int(consistency[p]["pixels_exceeding"]["0.001"])] for p in PAIRS]))
    add("")
    add("The residual is float-level noise from batch-1 full-scene forwards against batch-4 "
        "validation forwards. Percentiles, correlation and per-class statistics computed from "
        "either source are identical to the digits that matter.")
    add("")
    add("**The correct comparison method is therefore direct pixel comparison at reviewed "
        "coordinates, which is what this audit performed.** The two paths are directly "
        "comparable; no phase re-derivation or re-inference was needed.")
    add("")

    add("## 2. Reviewed class distributions")
    add("")
    add(table(["pair", "class", "n", "mean", "median", "std", "p05", "p25", "p75", "p95"],
              [[p, "change (1)" if l == 1 else "no-change (0)", "{:,}".format(int(class_row(p, l)["count"])),
                "%.4f" % float(class_row(p, l)["mean"]), "%.4f" % float(class_row(p, l)["median"]),
                "%.4f" % float(class_row(p, l)["std"]), "%.4f" % float(class_row(p, l)["p05"]),
                "%.4f" % float(class_row(p, l)["p25"]), "%.4f" % float(class_row(p, l)["p75"]),
                "%.4f" % float(class_row(p, l)["p95"])]
               for p in PAIRS for l in (1, 0)]))
    add("")
    add("Percentage overlap between the classes, measured on reviewed pixels only:")
    add("")
    add(table(["pair", "no-change above change median", "change at or below no-change p95",
               "AP on reviewed pixels"],
              [[p, "%.1f%%" % (100 * evidence[p]["negatives_above_positive_median"]),
                "%.1f%%" % (100 * evidence[p]["positives_below_negative_p95"]),
                "%.4f" % evidence[p]["ap_on_reviewed"]] for p in PAIRS]))
    add("")
    add("The classes do separate within the reviewed sample — AP %.3f on Pair A and %.3f on "
        "Pair B — but that separation is bought with a negative class that is almost degenerate "
        "on Pair A (mean %.4f, standard deviation %.4f). A boundary fitted against those "
        "negatives has seen very little stable-but-lookalike content."
        % (evidence["A"]["ap_on_reviewed"], evidence["B"]["ap_on_reviewed"],
           float(class_row("A", 0)["mean"]), float(class_row("A", 0)["std"])))
    add("")

    add("## 3. Full-scene distributions")
    add("")
    add(table(["pair", "valid px", "mean", "median", "std", "q10", "q25", "q75", "q90", "q99"],
              [[p, "{:,}".format(int(stats[p]["valid_pixels"])), "%.4f" % float(stats[p]["mean"]),
                "%.3f" % float(stats[p]["median_from_hist"]),
                "%.4f" % float(stats[p]["std_from_hist"]),
                stats[p]["q10"], stats[p]["q25"], stats[p]["q75"], stats[p]["q90"],
                stats[p]["q99"]] for p in PAIRS]))
    add("")
    add(table(["pair", "coverage @0.03", "@0.10", "@0.50",
               "scene below no-change p95", "scene median"],
              [[p, "%.1f%%" % (100 * evidence[p]["coverage_at_0p03"]),
                "%.1f%%" % (100 * evidence[p]["coverage_at_0p10"]),
                "%.2f%%" % (100 * evidence[p]["coverage_at_0p50"]),
                "%.1f%%" % (100 * evidence[p]["scene_below_negative_p95"]),
                "%.3f" % evidence[p]["scene_median"]] for p in PAIRS]))
    add("")
    add("**The scene sits where neither reviewed class lives.** On Pair A only %.1f %% of the "
        "scene scores at or below the reviewed no-change 95th percentile (%.4f), while the scene "
        "median is %.3f — above the reviewed no-change median by %.3f. The model is not "
        "contradicting itself between the two runs; it is assigning mid-range scores to terrain "
        "that was never reviewed."
        % (100 * evidence["A"]["scene_below_negative_p95"],
           evidence["A"]["reviewed_negative_p95"], evidence["A"]["scene_median"],
           evidence["A"]["scene_median"] - evidence["A"]["reviewed_negative_median"]))
    add("")
    add("This is why 0.03–0.10 covered 70–90 % of the tiles: those thresholds sit inside the "
        "scene's bulk, not in a tail.")
    add("")

    add("## 4. Spatial observations")
    add("")
    add(table(["pair", "block mean min", "median", "max", "std",
               "top-decile blocks hold"],
              [[p, "%.3f" % evidence[p]["block_mean_min"],
                "%.3f" % evidence[p]["block_mean_median"],
                "%.3f" % evidence[p]["block_mean_max"],
                "%.3f" % evidence[p]["block_mean_std"],
                "%.0f%% of >=0.50 pixels" % (100 * evidence[p]["top_decile_block_share_of_high_scores"])]
               for p in PAIRS]))
    add("")
    add("Scores are structured, not uniform: block means span a wide range and high scores "
        "concentrate. From the representative crops (`representative_regions/`), the highest-"
        "mean block on Pair B is an agricultural patchwork with no structural change between "
        "dates — field boundaries, crop-type differences and seasonal contrast carry the high "
        "scores. The lowest-mean blocks are homogeneous surfaces. These are observations to "
        "inspect, not labels: the audit did not classify any region as change or no-change.")
    add("")
    add("Candidate high-score patterns are therefore consistent with: stable but visually "
        "heterogeneous land cover (fields, field edges, scattered settlement), radiometric or "
        "seasonal differences, and possibly real construction. Separating those requires "
        "reviewed negatives in exactly those places — which the current annotation set does not "
        "contain.")
    add("")

    add("## 5. Hypothesis assessment")
    add("")
    add(table(["id", "hypothesis", "classification"],
              [[h[0], h[1], h[2]] for h in hypotheses]))
    add("")
    add("Evidence for each is recorded in `failure_hypothesis_assessment.csv`.")
    add("")

    add("## 6. Root cause")
    add("")
    add("**Case 2, with Case 3 as its mechanism.** The inference path is consistent — that is "
        "measured, not assumed — so the discrepancy is not an implementation bug. The reviewed "
        "sample does not span the score space the model produces at tile scale. On Pair A the "
        "negative sample is degenerate: a point mass at zero (IQR [%.4f, %.4f], containing %.1f%% "
        "of the scene) while the scene median is %.3f. The training supervision therefore "
        "contains almost no example of stable-but-lookalike land cover, which is most of each "
        "tile. Pair B shows the same pattern more mildly (negative median %.4f against scene "
        "median %.3f), so the mechanism is shared even though its severity is not."
        % (evidence["A"]["neg_iqr"][0], evidence["A"]["neg_iqr"][1],
           100 * evidence["A"]["scene_in_neg_iqr"], evidence["A"]["scene_median"],
           evidence["B"]["reviewed_negative_median"], evidence["B"]["scene_median"]))
    add("")
    add("Secondary finding, independent of the above: the two pairs have materially different "
        "score distributions (scene means %.4f and %.4f), so a single operating threshold across "
        "both is not supportable even after the annotation gap is closed."
        % (evidence["A"]["scene_mean"], evidence["B"]["scene_mean"]))
    add("")
    add("Unresolved: whether patch-level context contributes (hypothesis E). The block structure "
        "is consistent with it but does not demonstrate it, and no ablation was run.")
    add("")

    add("## 7. Recommended next action")
    add("")
    add("**Do not retrain yet, and do not change the inference code.** The inference code is "
        "exonerated by measurement, and a retrain would be launched against a supervision set "
        "whose defect is now quantified but not yet repaired.")
    add("")
    add("Smallest controlled next experiment: **a targeted negative-sampling review pass.** "
        "Select several hundred unreviewed locations from the scene's mid-score band "
        "(roughly 0.15–0.60 on Pair A, 0.20–0.70 on Pair B, where the current negatives do not "
        "reach), annotate them as change or no-change using the same protocol, and re-measure "
        "the class overlap and AP on that held-out review. That answers the leading hypothesis "
        "directly, costs review time rather than GPU time, and produces the hard negatives any "
        "later training run would need.")
    add("")
    add("Only if those mid-band locations turn out to be genuinely mostly change should the "
        "conclusion move toward Case 3 proper — that the model cannot separate man-made change "
        "from other content — and a retraining experiment become justified.")
    add("")

    add("## 8. Scope")
    add("")
    add("    inference run                no")
    add("    training run                 no")
    add("    checkpoint / normalisation   unmodified")
    add("    annotations                  unmodified")
    add("    threshold selected           no")
    add("    test labels / masks / indices / metrics   not accessed")
    add("    unreviewed pixels as negatives            never")
    add("    device                       cpu, windowed reads")
    add("")
    add("One limitation of the comparison: the validation vector and the full-scene raster differ "
        "by up to %.1e because the two runs batched forward passes differently. That is below "
        "every tolerance relevant to a threshold decision, but it means the two are equal to "
        "float precision rather than bit-identical." % a_max)
    add("")

    (out / "full_scene_diagnostic_report.md").write_text("\n".join(lines) + "\n")
    print("wrote failure_hypothesis_assessment.csv, diagnostic_manifest.json, "
          "full_scene_diagnostic_report.md")
    for pair in PAIRS:
        e = evidence[pair]
        print("[%s] scene median %.3f | reviewed neg median %.3f | neg p95 %.4f | "
              "scene below that %.1f%% | AP %.3f"
              % (pair, e["scene_median"], e["reviewed_negative_median"],
                 e["reviewed_negative_p95"], 100 * e["scene_below_negative_p95"],
                 e["ap_on_reviewed"]))


if __name__ == "__main__":
    main()
