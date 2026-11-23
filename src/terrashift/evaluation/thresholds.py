"""Threshold-dependent metric reporting."""

from terrashift.metrics import average_precision, metrics_from_counts, roc_auc


def metrics_at_thresholds(scores, labels, thresholds):
    """Detection metrics at each fixed threshold, plus the ranking metrics."""
    result = {"ap": round(float(average_precision(labels, scores)), 6),
              "auc": round(float(roc_auc(labels, scores)), 6),
              "pixels": int(labels.size),
              "positives": int((labels == 1).sum()),
              "thresholds": {}}
    for threshold in thresholds:
        predicted = scores >= threshold
        truth = labels == 1
        counts = metrics_from_counts(int((predicted & truth).sum()),
                                     int((predicted & ~truth).sum()),
                                     int((~predicted & truth).sum()),
                                     int((~predicted & ~truth).sum()))
        result["thresholds"]["%.2f" % threshold] = {
            key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in counts.items()}
        result["thresholds"]["%.2f" % threshold]["predicted_positive_fraction"] = \
            round(float(predicted.mean()), 6)
    return result
