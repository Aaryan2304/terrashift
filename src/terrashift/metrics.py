"""Binary change-detection metrics.

TP, FP, FN and TN describe a binary result completely, and everything here is arithmetic on those
counts or on the score ranking.

Undefined results are NaN, never 0.0: precision means nothing when a detector never fires, and 0.0
would report a measurement that did not happen. ``accuracy`` is absent on purpose. At a 4.85 % change
rate, predicting nothing scores 95.2 % with zero recall, while ``iou`` and ``f1`` exclude TN.

Ranking metrics follow scikit-learn: AP is ``-sum(diff(recall) * precision[:-1])`` over distinct
scores with ``(1, 0)`` appended; ROC AUC is the trapezoidal area with ``fpr = fp / n_neg`` and
``tpr = tp / n_pos``, with ``(0, 0)`` prepended. scikit-learn is not installed here, so ``__main__``
checks both against its worked examples.
"""

import numpy as np


def metrics_from_counts(tp, fp, fn, tn):
    """Precision, recall, F1 and IoU from four counts.

    Returns ``NaN`` for a metric whose denominator is zero. ``tp``, ``fp``,
    ``fn``, ``tn`` may be integers or floats.
    """
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    tn = float(tn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1_denominator = 2.0 * tp + fp + fn
    f1 = 2.0 * tp / f1_denominator if f1_denominator > 0 else float("nan")
    iou_denominator = tp + fp + fn
    iou = tp / iou_denominator if iou_denominator > 0 else float("nan")

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def counts_at_distinct_scores(y_true, y_score):
    """True-positive and false-positive counts at each distinct score value.

    This is the reduction scikit-learn performs: the samples are sorted by
    descending score, then only the last index of each group of tied scores is
    kept. Returns ``(tps, fps)``, both increasing, and the number of positives.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.size != s.size:
        raise ValueError("y_true and y_score must have the same length, "
                         "got %d and %d" % (y.size, s.size))
    if y.size == 0:
        raise ValueError("y_true and y_score must not be empty")

    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    s_sorted = s[order]

    # Indices where the score changes, plus the final index.
    distinct = np.nonzero(np.diff(s_sorted))[0]
    threshold_idxs = np.concatenate([distinct, [y_sorted.size - 1]])

    running_tp = np.cumsum(y_sorted == 1)
    running_fp = np.cumsum(y_sorted == 0)
    tps = running_tp[threshold_idxs].astype(np.float64)
    fps = running_fp[threshold_idxs].astype(np.float64)
    return tps, fps, float((y == 1).sum())


def precision_recall_arrays(y_true, y_score):
    """Precision and recall arrays as scikit-learn returns them.

    Recall is decreasing and the arrays end with ``(precision=1, recall=0)``.
    Returns ``NaN`` for both when the truth contains no positive sample.
    """
    tps, fps, n_pos = counts_at_distinct_scores(y_true, y_score)
    if n_pos == 0:
        # scikit-learn warns and sets recall to 1 for all thresholds; there is
        # no meaningful curve, so report undefined.
        return None, None

    ps = tps + fps
    precision = np.where(ps != 0, tps / ps, 0.0)
    recall = tps / tps[-1]
    precision = np.concatenate([precision[::-1], [1.0]])
    recall = np.concatenate([recall[::-1], [0.0]])
    return precision, recall


def average_precision(y_true, y_score):
    """Average precision, matching ``sklearn.metrics.average_precision_score``.

    AP summarizes the precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold as the weight. Not interpolated.
    """
    precision, recall = precision_recall_arrays(y_true, y_score)
    if precision is None:
        return float("nan")
    return float(max(0.0, -np.sum(np.diff(recall) * precision[:-1])))


def roc_auc(y_true, y_score):
    """Area under the ROC curve, matching ``sklearn.metrics.roc_auc_score``.

    Trapezoidal area under ``tpr`` versus ``fpr``, with ``(0, 0)`` prepended.
    Returns ``NaN`` when the truth contains only one class.
    """
    y = np.asarray(y_true).ravel()
    tps, fps, n_pos = counts_at_distinct_scores(y_true, y_score)
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    fpr = np.concatenate([[0.0], fps / n_neg])
    tpr = np.concatenate([[0.0], tps / n_pos])
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def changed_area_stats(pred_mask):
    """Changed and unchanged pixel counts and percentages for one mask."""
    mask = np.asarray(pred_mask)
    total = int(mask.size)
    changed = int(np.count_nonzero(mask))
    if total == 0:
        return {"changed_pixels": 0, "total_pixels": 0,
                "changed_pct": float("nan"), "unchanged_pct": float("nan")}
    changed_pct = 100.0 * changed / total
    return {"changed_pixels": changed, "total_pixels": total,
            "changed_pct": changed_pct, "unchanged_pct": 100.0 - changed_pct}


if __name__ == "__main__":
    print("=" * 72)
    print("metrics.py self-check")
    print("=" * 72)

    # --- counts, and the degenerate cases that must be NaN
    cm = metrics_from_counts(2, 2, 2, 10)
    assert cm["precision"] == 0.5 and cm["recall"] == 0.5 and cm["f1"] == 0.5
    assert abs(cm["iou"] - 2 / 6) < 1e-12
    assert abs(cm["iou"] - cm["f1"] / (2 - cm["f1"])) < 1e-12
    print("hand-made 4x4: P=R=F1=0.5, IoU=2/6, IoU == F1/(2-F1)      OK")

    empty = metrics_from_counts(0, 0, 0, 0)
    assert all(np.isnan(empty[k]) for k in ("precision", "recall", "f1", "iou"))
    print("no prediction and no truth -> all four metrics NaN          OK")

    nothing_fired = metrics_from_counts(0, 0, 5, 95)
    assert np.isnan(nothing_fired["precision"]), "undefined precision must be NaN"
    assert nothing_fired["recall"] == 0.0
    print("never fired -> precision NaN, recall 0.0                    OK")

    # --- the accuracy trap, at this project's real change rate
    lazy = metrics_from_counts(0, 0, 485, 9515)
    accuracy = (lazy["tp"] + lazy["tn"]) / (lazy["tp"] + lazy["fp"] + lazy["fn"] + lazy["tn"])
    print("do-nothing detector at 4.85%% change: accuracy %.4f, recall %.1f, IoU %.1f"
          % (accuracy, lazy["recall"], lazy["iou"]))
    assert accuracy > 0.95 and lazy["recall"] == 0.0 and lazy["iou"] == 0.0
    print("accuracy looks great while the detector found nothing      OK")

    # --- AP against the worked example printed in the sklearn docs
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    ap = average_precision(y_true, y_score)
    print("AP for the sklearn documented example: %.4f (docs say 0.83)" % ap)
    assert abs(ap - 0.83) < 0.005, ap

    # --- AUC against the worked example printed in the sklearn docs
    y_true = np.array([1, 1, 2, 2])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    # sklearn: roc_curve(y_true, y_score, pos_label=2); auc(fpr, tpr) == 0.75
    pos = y_true == 2
    auc = roc_auc(pos.astype(int), y_score)
    print("AUC for the sklearn documented example: %.4f (docs say 0.75)" % auc)
    assert abs(auc - 0.75) < 1e-12, auc

    # --- a perfect and a worthless ranking
    perfect = roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    random_like = roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    print("perfect ranking AUC %.4f, all-tied ranking AUC %.4f" % (perfect, random_like))
    assert abs(perfect - 1.0) < 1e-12 and abs(random_like - 0.5) < 1e-12
    assert abs(average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12
    print("perfect ranking -> AUC 1.0 and AP 1.0; ties -> AUC 0.5    OK")

    # --- undefined ranking metrics
    assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.5, 0.9])), "one-class truth -> AUC NaN"
    assert np.isnan(average_precision([0, 0, 0], [0.1, 0.5, 0.9])), "no positives -> AP NaN"
    print("one-class truth -> AUC NaN; no positives -> AP NaN         OK")

    # --- changed area
    stats = changed_area_stats(np.array([[0, 1], [0, 0]]))
    assert stats["changed_pixels"] == 1 and stats["total_pixels"] == 4
    assert stats["changed_pct"] == 25.0 and stats["unchanged_pct"] == 75.0
    print("changed_area_stats: 1 of 4 -> 25.0% / 75.0%                OK")

    print("\n" + "=" * 72)
    print("all self-checks passed")
    print("=" * 72)
