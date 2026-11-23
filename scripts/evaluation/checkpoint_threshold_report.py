"""Build checkpoint_threshold_decision.md from the decision analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    root = Path(args.dir)

    verification = list(csv.DictReader(open(root / "checkpoint_verification.csv")))
    operating = list(csv.DictReader(open(root / "operating_points.csv")))
    sweep = list(csv.DictReader(open(root / "threshold_metrics.csv")))
    manifest = json.loads((root / "decision_manifest.json").read_text())
    matched = manifest["matched_recall"]

    by_name = {row["checkpoint"]: row for row in verification}
    fine = [row for row in operating if row["rationale"] == "conventional threshold grid"]
    info = [row for row in operating if row["rationale"] != "conventional threshold grid"]

    def row_at(name, threshold):
        for row in operating:
            if row["checkpoint"] == name and abs(float(row["threshold"]) - threshold) < 1e-9:
                return row
        raise KeyError((name, threshold))

    def fmt(threshold):
        return "%.2f" % float(threshold)

    sweeps = {(r["checkpoint"], r["scope"], round(float(r["threshold"]), 2)): r for r in sweep}

    def s_at(name, scope, threshold):
        return sweeps[(name, scope, round(threshold, 2))]

    lines = []
    add = lines.append

    add("# Checkpoint and operating-threshold decision")
    add("")
    add("Validation data only. No training, no model inference, no CUDA, no test-split "
        "access. Every number below is read from the saved validation score vectors and the "
        "reviewed validation labels; nothing is regenerated from a model.")
    add("")
    add("This is a decision analysis, not an automatic selection. The checkpoint "
        "recommendation and the threshold band are both provisional and both reversible.")
    add("")

    # ---------------------------------------------------------------- 1
    add("## 1. Verified inputs")
    add("")
    add(table(["checkpoint", "best epoch", "SHA-256", "pooled AP recorded matches"],
              [[row["checkpoint"], row["best_epoch"], row["sha256"][:16] + "…",
                row["pooled_ap_matches_recorded"]] for row in verification]))
    add("")
    add("Score vectors: 25,409 float32 values per checkpoint (2,759 Pair A + 22,650 Pair B). "
        "Labels: %d reviewed pixels, %d Pair A positives, %d Pair B positives. Pair B "
        "components recovered from an existing diagnosis raster: %s components, sizes sum "
        "%d." % (manifest["counts"]["reviewed_total"], manifest["counts"]["positives_A"],
                 manifest["counts"]["positives_B"], manifest["counts"]["components"],
                 sum(manifest["counts"]["component_sizes"].values())))
    add("")

    # ---------------------------------------------------------------- 2
    add("## 2. Verified checkpoint comparison")
    add("")
    add(table(["metric", "OSCD (reference)", "0.834 s43", "1.0 s43"],
              [["best epoch", "n/a", by_name["local_0.834_s43"]["best_epoch"],
                by_name["local_1.0_s43"]["best_epoch"]],
               ["**pooled AP (micro)**",
                by_name["oscd"]["ap_pooled_micro"],
                by_name["local_0.834_s43"]["ap_pooled_micro"],
                by_name["local_1.0_s43"]["ap_pooled_micro"]],
               ["Pair A AP", by_name["oscd"]["ap_pair_A"],
                by_name["local_0.834_s43"]["ap_pair_A"],
                by_name["local_1.0_s43"]["ap_pair_A"]],
               ["Pair B AP", by_name["oscd"]["ap_pair_B"],
                by_name["local_0.834_s43"]["ap_pair_B"],
                by_name["local_1.0_s43"]["ap_pair_B"]],
               ["**macro AP (mean of pairs)**",
                by_name["oscd"]["ap_macro_mean_of_pairs"],
                by_name["local_0.834_s43"]["ap_macro_mean_of_pairs"],
                by_name["local_1.0_s43"]["ap_macro_mean_of_pairs"]],
               ["pooled AUC", by_name["oscd"]["auc_pooled_micro"],
                by_name["local_0.834_s43"]["auc_pooled_micro"],
                by_name["local_1.0_s43"]["auc_pooled_micro"]],
               ["Pair A AUC", by_name["oscd"]["auc_pair_A"],
                by_name["local_0.834_s43"]["auc_pair_A"],
                by_name["local_1.0_s43"]["auc_pair_A"]],
               ["Pair B AUC", by_name["oscd"]["auc_pair_B"],
                by_name["local_0.834_s43"]["auc_pair_B"],
                by_name["local_1.0_s43"]["auc_pair_B"]]]))
    add("")
    add("### Micro and macro disagree, and that matters for the selection rule")
    add("")
    add("For both fine-tuned checkpoints the pooled AP is *lower* than the mean of the two "
        "pair APs: micro − macro is %s for 0.834 and %s for 1.0. Pooling interleaves two "
        "score distributions whose scales differ, so a ranking that is strong within each "
        "pair loses ground when the pairs are ranked together. Combined AP therefore "
        "understates both checkpoints, and it is a scale-sensitive quantity rather than a "
        "pure summary of detection quality."
        % (by_name["local_0.834_s43"]["micro_minus_macro"],
           by_name["local_1.0_s43"]["micro_minus_macro"]))
    add("")
    add("### Is 0.834 s43 the better checkpoint?")
    add("")
    add("It is better on pooled AP (%s vs %s), Pair B AP (%s vs %s), Pair B AUC (%s vs %s) "
        "and macro AP (%s vs %s). It is **worse on Pair A**: AP %s vs %s and AUC %s vs %s."
        % (by_name["local_0.834_s43"]["ap_pooled_micro"], by_name["local_1.0_s43"]["ap_pooled_micro"],
           by_name["local_0.834_s43"]["ap_pair_B"], by_name["local_1.0_s43"]["ap_pair_B"],
           by_name["local_0.834_s43"]["auc_pair_B"], by_name["local_1.0_s43"]["auc_pair_B"],
           by_name["local_0.834_s43"]["ap_macro_mean_of_pairs"],
           by_name["local_1.0_s43"]["ap_macro_mean_of_pairs"],
           by_name["local_0.834_s43"]["ap_pair_A"], by_name["local_1.0_s43"]["ap_pair_A"],
           by_name["local_0.834_s43"]["auc_pair_A"], by_name["local_1.0_s43"]["auc_pair_A"]))
    add("")
    add("Pair B holds 89 % of the reviewed pixels, so a pooled figure is mostly a Pair B "
        "statement. **\"Better\" is therefore conditional, not universal.** The provisional "
        "candidate survives verification on Pair B terms; it does not survive as a claim that "
        "it detects Pair A better, because it does not.")
    add("")

    # ---------------------------------------------------------------- 3
    add("## 3. Threshold analysis")
    add("")
    add("Full sweep at 99 thresholds from 0.01 to 0.99 for four scopes per checkpoint "
        "(pooled, Pair A, Pair B, macro) is in `threshold_metrics.csv`. The table below is the "
        "conventional grid, identical for both checkpoints so the columns are comparable.")
    add("")
    for name in ("local_0.834_s43", "local_1.0_s43"):
        rows = [r for r in fine if r["checkpoint"] == name]
        add("**%s**" % name)
        add("")
        add(table(["thr", "TP", "FP", "FN", "precision", "recall", "F1", "IoU",
                   "pred+", "Pair A R", "Pair B P", "Pair B R", "comp-8 R"],
                  [[fmt(r["threshold"]), int(float(r["micro_tp"])),
                    int(float(r["micro_fp"])), int(float(r["micro_fn"])),
                    r["micro_precision"], r["micro_recall"], r["micro_f1"], r["micro_iou"],
                    r["predicted_positive_fraction"], r["pair_A_recall"],
                    r["pair_B_precision"], r["pair_B_recall"], r["component8_recall"]]
                   for r in rows]))
        add("")

    add("### Do not select on F1")
    add("")
    f1_points = {row["checkpoint"]: row for row in info
                 if row["rationale"].startswith("maximum pooled F1")}
    add("Maximum pooled F1 sits at threshold %s for 0.834 and %s for 1.0 — the very bottom "
        "of the score range. At those points a candidate layer flags %s and %s of the reviewed "
        "pixels and still only reaches precision %s and %s. Because this reviewed set is "
        "roughly half change, F1 is maximised by predicting almost everything, so F1 ranks "
        "the two checkpoints in a way that has nothing to do with which one separates change "
        "from no-change."
        % (f1_points["local_0.834_s43"]["threshold"], f1_points["local_1.0_s43"]["threshold"],
           f1_points["local_0.834_s43"]["predicted_positive_fraction"],
           f1_points["local_1.0_s43"]["predicted_positive_fraction"],
           f1_points["local_0.834_s43"]["micro_precision"],
           f1_points["local_1.0_s43"]["micro_precision"]))
    add("")
    add("### Is 0.50 appropriate?")
    add("")
    at50 = {name: row_at(name, 0.50) for name in ("local_0.834_s43", "local_1.0_s43")}
    add("No, for this use. At 0.50 the fine-tuned checkpoints reach pooled precision %s "
        "(0.834) and %s (1.0) but recall only %s and %s, with Pair B component-8 recall %s and "
        "%s. A candidate generator for human review that misses three quarters of the largest "
        "change region is not fit for purpose, however attractive its precision looks on an "
        "enriched reviewed subset."
        % (at50["local_0.834_s43"]["micro_precision"],
           at50["local_1.0_s43"]["micro_precision"],
           at50["local_0.834_s43"]["micro_recall"],
           at50["local_1.0_s43"]["micro_recall"],
           at50["local_0.834_s43"]["component8_recall"],
           at50["local_1.0_s43"]["component8_recall"]))
    add("")

    # ---------------------------------------------------------------- 4
    add("## 4. Matched-recall comparison")
    add("")
    add("If the two checkpoints are asked to achieve the same Pair B recall, 0.834 s43 needs a "
        "higher threshold and returns more precision at every level tested:")
    add("")
    add(table(["Pair B recall", "0.834 s43 threshold", "0.834 precision", "0.834 FP",
               "1.0 s43 threshold", "1.0 precision", "1.0 FP"],
              [["%.2f" % row["pair_B_recall"], row["local_0.834_s43_threshold"],
                row["local_0.834_s43_precision"], row["local_0.834_s43_fp"],
                row["local_1.0_s43_threshold"], row["local_1.0_s43_precision"],
                row["local_1.0_s43_fp"]] for row in matched]))
    add("")
    reductions = [(1 - float(r["local_0.834_s43_fp"]) / float(r["local_1.0_s43_fp"])) * 100
                  for r in matched]
    precisions = [float(r["local_0.834_s43_precision"]) - float(r["local_1.0_s43_precision"])
                  for r in matched]
    add("The gap never reverses across the tested range: precision advantage %.3f to %.3f, "
        "and **%.0f%% to %.0f%% fewer Pair B false positives at equal recall**. This is the "
        "strongest single argument for carrying 0.834 forward."
        % (min(precisions), max(precisions), min(reductions), max(reductions)))
    add("")

    # ---------------------------------------------------------------- 5
    add("## 5. Pair A and the cost of the choice")
    add("")
    add("Pair A behaves in the opposite direction. At 0.05, 0.834 recalls %s of Pair A change "
        "and 1.0 recalls %s; at 0.50, 0.834 recalls %s against %s. 1.0 is the stronger Pair A "
        "detector at every threshold, and by a wide margin at the conventional 0.50."
        % (row_at("local_0.834_s43", 0.05)["pair_A_recall"],
           row_at("local_1.0_s43", 0.05)["pair_A_recall"],
           at50["local_0.834_s43"]["pair_A_recall"],
           at50["local_1.0_s43"]["pair_A_recall"]))
    add("")
    add("So the checkpoint choice embeds a priority: 0.834 favours Pair B precision, 1.0 "
        "favours Pair A recall. Which is correct depends on which pair the downstream workflow "
        "cares about, and that is a question about the business, not about the metrics.")
    add("")

    # ---------------------------------------------------------------- 6
    add("## 6. Largest Pair B component (component 8, 10,749 px)")
    add("")
    add(table(["thr"] + ["0.834 R", "1.0 R"],
              [[fmt(r["threshold"]), r["component8_recall"],
                row_at("local_1.0_s43", float(r["threshold"]))["component8_recall"]]
               for r in fine if r["checkpoint"] == "local_0.834_s43"]))
    add("")
    c8_tbl = [(float(r["threshold"]), float(r["component8_recall"]),
               float(row_at("local_1.0_s43", float(r["threshold"]))["component8_recall"]))
              for r in fine if r["checkpoint"] == "local_0.834_s43"]
    higher = [t for t, a, b in c8_tbl if a > b]
    add("The region is only recoverable at low thresholds. **0.834 leads on component 8 only "
        "at the lowest grid point (0.712 vs 0.678 at 0.05); from 0.10 upward 1.0 recovers "
        "more of it** — %s. This is the one place where the Pair B aggregate and the largest "
        "region disagree, and it is worth being explicit: 0.834's advantage is in precision "
        "across the pair, not in reaching the biggest site."
        % ", ".join("%.2f: %.3f vs %.3f" % (t, a, b) for t, a, b in c8_tbl[1:]))
    add("")
    add("Neither checkpoint detects the region at 0.50 (0.180 and 0.207). Any operating point "
        "at or above 0.50 accepts losing roughly four fifths of the largest construction site "
        "in Pair B.")
    add("")

    # ---------------------------------------------------------------- 7
    add("## 7. Shortlisted operating points")
    add("")
    add("Four candidates per checkpoint, taken from the measured curve, plus the landmarks "
        "that were considered and rejected.")
    add("")
    short_rows = []
    for name in ("local_0.834_s43", "local_1.0_s43"):
        for threshold in (0.03, 0.05, 0.10, 0.15):
            pooled = s_at(name, "combined", threshold)
            pairb = s_at(name, "B", threshold)
            short_rows.append(
                [name, "%.2f" % threshold, pooled["precision"], pooled["recall"],
                 pairb["precision"], pairb["recall"], pairb["component8_recall"],
                 pooled["predicted_positive_fraction"],
                 round(float(pooled["fp"]) / 25.409 / 1000 * 1000, 1)])
    add(table(["checkpoint", "thr", "pooled P", "pooled R", "Pair B P", "Pair B R",
               "comp-8 R", "pred+", "FP per 1000 reviewed"], short_rows))
    add("")
    add("Rejected and why:")
    add("")
    add("- **0.20 and 0.30** — component-8 recall falls to 0.43 and 0.32 for 0.834. The "
        "largest site is lost before the false-positive load becomes acceptable.")
    add("- **0.50** — precision looks best and is the weakest option for review, as shown "
        "above.")
    add("- **0.01 (max F1)** — flags 74–76 % of reviewed pixels; F1 is not a usable criterion "
        "on a set that is half positive.")
    add("")

    # ---------------------------------------------------------------- 8
    add("## 8. Recommendation (provisional, both items unlocked)")
    add("")
    add("**Carry forward: `local_0.834_s43` (best epoch 7, `dbce89f6…0a2554cd`).**")
    add("")
    add("Reasons, in order of strength: at matched Pair B recall it holds %.3f to %.3f higher "
        "precision and %.0f%% to %.0f%% fewer false positives; it has the higher Pair B AP, "
        "Pair B AUC, macro AP and pooled AP; and Pair B carries 89%% of the reviewed evidence."
        % (min(precisions), max(precisions), min(reductions), max(reductions)))
    add("")
    add("**Carry it with a stated reservation:** it is the weaker Pair A detector, the pooled "
        "AP gap over 1.0 (0.0089) is smaller than the seed-to-seed spread already measured for "
        "this weight on seed 42 versus 43 (≈0.069), and the earlier paired-seed analysis found "
        "the 0.834-versus-1.0 ordering reverses across seeds. If Pair A detection turns out to "
        "matter more than Pair B precision, switch to `local_1.0_s43` and re-read this "
        "document's Pair A column.")
    add("")
    p05, p03, p10, p20, p01 = (s_at("local_0.834_s43", "combined", t)
                                for t in (0.05, 0.03, 0.10, 0.20, 0.01))
    b05, b03, b10, b20 = (s_at("local_0.834_s43", "B", t) for t in (0.05, 0.03, 0.10, 0.20))
    b05_alt = s_at("local_1.0_s43", "B", 0.05)
    add("**Provisional operating point: 0.05, with 0.03–0.10 as the defensible band.**")
    add("")
    add("At 0.05, `local_0.834_s43` gives pooled precision %.3f, pooled recall %.3f, Pair B "
        "precision %.3f, Pair B recall %.3f and component-8 recall %.3f, flagging %.3f of "
        "reviewed pixels. Roughly seven of every ten Pair B candidates are genuine, and it "
        "reaches nearly three quarters of the largest change region."
        % (float(p05["precision"]), float(p05["recall"]), float(b05["precision"]),
           float(b05["recall"]), float(b05["component8_recall"]),
           float(p05["predicted_positive_fraction"])))
    add("")
    add("**Why this side of the band, and why it is a band.** Recall is far more elastic than "
        "precision here. Moving 0.10 → 0.05 buys %+.3f Pair B recall for %+.3f precision; "
        "0.05 → 0.03 buys a further %+.3f recall for %+.3f precision. Both trades favour "
        "recall, which is the purpose of a candidate layer. The other direction is expensive: "
        "0.05 → 0.20 costs %.3f Pair B recall and returns only %.3f precision, while "
        "component-8 recall falls from %.3f to %.3f. There is no sharp knee — pooled precision "
        "drifts only from "
        "%.3f to %.3f between 0.01 and 0.10 while recall falls from %.3f to %.3f — so the data "
        "does not single out a threshold. The exact value inside the band should follow review "
        "capacity, not a metric."
        % (float(b05["recall"]) - float(b10["recall"]),
           float(b05["precision"]) - float(b10["precision"]),
           float(b03["recall"]) - float(b05["recall"]),
           float(b03["precision"]) - float(b05["precision"]),
           abs(float(b20["recall"]) - float(b05["recall"])),
           float(b20["precision"]) - float(b05["precision"]),
           float(b05["component8_recall"]), float(b20["component8_recall"]),
           float(p01["precision"]), float(p10["precision"]),
           float(p01["recall"]), float(p10["recall"])))
    add("")
    add("**Component 8 at the recommended point.** At 0.05, 0.834 recovers %.3f of the largest "
        "site against %.3f for 1.0 at the same threshold, so this recommendation does not trade "
        "away the biggest region — 0.05 is the one grid point where 0.834 leads on it. From "
        "0.10 upward 1.0 recovers more of that region, a further reason to keep the operating "
        "point low."
        % (float(b05["component8_recall"]), float(b05_alt["component8_recall"])))
    add("")
    add("**The threshold is not locked.** No numeric recall target was assumed at any point. "
        "The band was read off the measured curve and then bounded by reviewer load, which is "
        "the constraint a candidate layer actually faces. Setting the final value needs "
        "tile-scale candidate counts, which this audit cannot produce without full-scene "
        "inference — deliberately not run.")
    add("")

    # ---------------------------------------------------------------- 9
    add("## 9. Limitations and unresolved decisions")
    add("")
    add("1. **The reviewed set is not a scene.** Prevalence is 0.527 on Pair B and 0.592 on "
        "Pair A because review concentrated on candidate change. Precision, predicted-positive "
        "fraction and every FP count here are properties of that enriched subset and will not "
        "transfer to full tiles. Threshold setting on tile-scale data will differ, probably "
        "toward higher thresholds.")
    add("2. **Two seeds, one checkpoint each.** The 0.834-versus-1.0 comparison at seed 43 is "
        "a single paired sample. The earlier report found the sign reverses at seed 42, so the "
        "checkpoint recommendation is not established fact.")
    add("3. **Component 8 dominates everything pixel-weighted.** It is 90 % of Pair B change, "
        "so pooled and Pair B figures are largely its score distribution. Component-level "
        "columns are given separately for that reason.")
    add("4. **No calibration was assessed.** Scores are compared as rankings; nothing here "
        "shows they are probabilities in any calibrated sense. A threshold is a rank cut, not "
        "a calibrated confidence.")
    add("5. **Unresolved: which pair the workflow prioritises.** This document cannot decide "
        "it. It changes the checkpoint recommendation, and it is the one input the evidence "
        "cannot supply.")
    add("6. **Unresolved: acceptable candidate volume.** If review capacity is the binding "
        "constraint, the operating point should move toward 0.15 or higher; if missed sites "
        "are the binding cost, toward 0.05.")
    add("7. **Unresolved: tile-scale false-positive rate.** Needs full-scene inference or a "
        "stratified unenriched sample. Neither exists in the current artifacts.")
    add("")

    # ---------------------------------------------------------------- 10
    add("## 10. Artifacts")
    add("")
    add("    checkpoint_verification.csv   path, SHA-256, best epoch, pooled/pair AP and AUC,")
    add("                                  micro-macro split, match against recorded AP")
    add("    threshold_metrics.csv         99 thresholds x 4 scopes x 2 checkpoints, full counts")
    add("    operating_points.csv          shortlist plus rejected landmarks, both checkpoints")
    add("    precision_recall_curves.png   pooled, Pair A, Pair B curves for all three checkpoints")
    add("    decision_manifest.json        source paths and hashes, scope flags, matched recall")
    add("    checkpoint_threshold_decision.md  this document")
    add("")
    add("Provenance: labels `%s…`, score vectors and checkpoints hashed in the manifest. "
        "Validation only; the test split was not accessed and no model was run."
        % manifest["sources"]["labels"]["sha256"][:16])
    add("")

    (root / "checkpoint_threshold_decision.md").write_text("\n".join(lines) + "\n")
    print("wrote", root / "checkpoint_threshold_decision.md")


if __name__ == "__main__":
    main()
