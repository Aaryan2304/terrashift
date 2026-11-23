"""Sliding-window inference and change-area summaries.

A change-detection model trained on 256 x 256 patches cannot be fed a
10980 x 10980 Sentinel-2 tile in one call. This module tiles a large pair,
runs the model on each tile, and averages the overlapping predictions.

Averaging the overlaps is not just a convenience: an interior pixel is covered
by several patches while a corner pixel is covered by one, so summing without
averaging would weight interior pixels more heavily. Dividing by the coverage
count makes every pixel count once. The averaging is also a mild ensemble.
"""

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from terrashift.metrics import changed_area_stats


def _window_origins(length, patch_size, stride):
    """Start indices of the windows that cover ``length`` pixels.

    The last window is shifted back so that it ends exactly at the edge, which
    guarantees full coverage with no pixel left out.
    """
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def sliding_window_predict(model, x1, x2, patch_size, stride, device,
                           batch_size=1, threshold=None):
    """Predict a change probability map for a pair of arbitrary size.

    Parameters
    ----------
    model : torch.nn.Module
        Takes ``(B, C, H, W)`` pairs and returns ``(B, 1, H, W)`` logits.
    x1, x2 : torch.Tensor
        ``(C, H, W)`` tensors for the two dates.
    patch_size : int
        The size of patch the model was trained on.
    stride : int
        The step between windows. Smaller means more overlap and more compute.
    device : torch.device
    batch_size : int
        Patches per forward call. Keep small on a 4 GB GPU.
    threshold : float, optional
        When given, also return a binary mask at this probability.

    Returns
    -------
    prob : numpy.ndarray, float64, shape (H, W)
        The averaged change probability.
    mask : numpy.ndarray, bool, shape (H, W), or None
    """
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape, got %s and %s"
                         % (tuple(x1.shape), tuple(x2.shape)))
    if x1.ndim != 3:
        raise ValueError("x1 and x2 must have shape (C, H, W)")
    if patch_size < 1 or stride < 1:
        raise ValueError("patch_size and stride must be positive")
    if stride > patch_size:
        raise ValueError("stride must not exceed patch_size; larger strides leave gaps")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    _, height, width = x1.shape
    if height < 1 or width < 1:
        raise ValueError("input height and width must be positive")
    # float32 probabilities and uint32 counts bound full-scene accumulator RAM.
    prob_map = np.zeros((height, width), dtype=np.float32)
    # uint32 avoids overflow even at high overlap (for example stride=1).
    count_map = np.zeros((height, width), dtype=np.uint32)

    ys = _window_origins(height, patch_size, stride)
    xs = _window_origins(width, patch_size, stride)

    model.eval()
    batch_a, batch_b, places = [], [], []
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + patch_size, height)
            x1_stop = min(x0 + patch_size, width)
            a = x1[:, y0:y1, x0:x1_stop]
            b = x2[:, y0:y1, x0:x1_stop]
            pad_h = patch_size - a.shape[1]
            pad_w = patch_size - a.shape[2]
            if pad_h or pad_w:
                a = F.pad(a, (0, pad_w, 0, pad_h))
                b = F.pad(b, (0, pad_w, 0, pad_h))
            batch_a.append(a)
            batch_b.append(b)
            places.append((y0, y1, x0, x1_stop))

            if len(batch_a) == batch_size:
                _accumulate(model, batch_a, batch_b, places, prob_map, count_map,
                            device, patch_size)
                batch_a, batch_b, places = [], [], []

    if batch_a:
        _accumulate(model, batch_a, batch_b, places, prob_map, count_map,
                    device, patch_size)

    if np.any(count_map == 0):
        raise RuntimeError("sliding windows did not cover every output pixel")
    prob = prob_map / count_map
    mask = None if threshold is None else (prob >= threshold)
    return prob, mask


def _accumulate(model, batch_a, batch_b, places, prob_map, count_map, device,
                patch_size):
    """Run one batch of patches and add the results to the coverage maps."""
    a = torch.stack(batch_a).to(device)
    b = torch.stack(batch_b).to(device)
    logits = model(a, b)
    expected = (len(batch_a), 1, patch_size, patch_size)
    if tuple(logits.shape) != expected:
        raise ValueError("model output shape %s does not match expected %s"
                         % (tuple(logits.shape), expected))
    probs = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
    for prob, (y0, y1, x0, x1) in zip(probs, places):
        prob_map[y0:y1, x0:x1] += prob[:y1 - y0, :x1 - x0]
        count_map[y0:y1, x0:x1] += np.uint32(1)


def summarize_pair(name, mask):
    """One summary row for a change mask."""
    stats = changed_area_stats(mask)
    return {"name": name, "changed_pixels": stats["changed_pixels"],
            "total_pixels": stats["total_pixels"],
            "changed_pct": stats["changed_pct"],
            "unchanged_pct": stats["unchanged_pct"]}


def write_folder_summary(rows, path):
    """Write the per-pair summary CSV and print the folder mean."""
    fields = ["name", "changed_pixels", "total_pixels", "changed_pct",
              "unchanged_pct"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        mean_changed = float(np.mean([r["changed_pct"] for r in rows]))
        print("processed %d pairs" % len(rows))
        print("mean changed area   : %.2f%%" % mean_changed)
        print("mean unchanged area : %.2f%%" % (100.0 - mean_changed))
    print("wrote", path)


def find_matching_pairs(before_dir, after_dir, extensions=(".npy",)):
    """Match supported array files by name between two folders."""
    before = {f for f in os.listdir(before_dir)
              if f.lower().endswith(extensions)}
    after = {f for f in os.listdir(after_dir)
             if f.lower().endswith(extensions)}
    common = sorted(before & after)
    for missing in sorted(before - after)[:5]:
        print("[warn] %s has no match in %s" % (missing, after_dir))
    for missing in sorted(after - before)[:5]:
        print("[warn] %s has no match in %s" % (missing, before_dir))
    return [(name, os.path.join(before_dir, name),
             os.path.join(after_dir, name)) for name in common]


if __name__ == "__main__":
    print("=" * 72)
    print("sliding_window_predict self-check on a toy model")
    print("=" * 72)

    class Toy(torch.nn.Module):
        """A separable toy scorer: logit = 5 * (mean(B) - mean(A)) - 2.

        Background pixels give logit -2 (sigmoid 0.119); the changed block
        gives logit 3 (sigmoid 0.953). Both are checkable by hand.
        """

        def forward(self, a, b):
            return 5.0 * (b.mean(dim=1, keepdim=True)
                          - a.mean(dim=1, keepdim=True)) - 2.0

    model = Toy()
    device = torch.device("cpu")
    h, w = 300, 260
    x1 = torch.zeros(1, h, w)
    x2 = torch.zeros(1, h, w)
    x2[:, 10:20, 10:20] = 1.0

    prob, mask = sliding_window_predict(model, x1, x2, patch_size=64, stride=32,
                                        device=device, threshold=0.5)
    print("input %dx%d, patch 64, stride 32 -> prob %s, mask %s"
          % (h, w, prob.shape, None if mask is None else mask.shape))
    assert prob.shape == (h, w)
    assert mask is not None and mask.shape == (h, w)
    assert prob[15, 15] > 0.9, prob[15, 15]
    assert prob[0, 0] < 0.2, prob[0, 0]
    assert mask[15, 15] and not mask[0, 0]
    print("changed block detected, background quiet                      OK")

    # A stride larger than the patch must be rejected instead of leaving gaps.
    try:
        sliding_window_predict(model, x1, x2, patch_size=64, stride=200,
                               device=device)
    except ValueError:
        print("stride larger than patch rejected to prevent coverage gaps OK")
    else:
        raise AssertionError("a stride larger than the patch was accepted")

    # A single patch, and a pair smaller than the patch.
    small = torch.zeros(1, 30, 30)
    p3, _ = sliding_window_predict(model, small, small.clone(), patch_size=64,
                                   stride=32, device=device)
    assert p3.shape == (30, 30)
    print("input smaller than the patch is padded and cropped back       OK")

    class ExactThresholdToy(torch.nn.Module):
        def forward(self, a, b):
            return torch.zeros((a.shape[0], 1, a.shape[2], a.shape[3]), device=a.device)

    tie_input = torch.zeros(1, 8, 8)
    tie_prob, tie_mask = sliding_window_predict(
        ExactThresholdToy(), tie_input, tie_input.clone(), patch_size=8,
        stride=8, device=device, threshold=0.5)
    assert np.all(tie_prob == 0.5) and tie_mask is not None and tie_mask.all()
    print("probability equal to threshold follows the inclusive training rule OK")

    assert mask is not None
    stats = changed_area_stats(mask.astype(np.uint8))
    print("changed_area_stats: %d of %d = %.2f%%"
          % (stats["changed_pixels"], stats["total_pixels"],
             stats["changed_pct"]))
    print("\nall self-checks passed")
