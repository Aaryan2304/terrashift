"""Run a trained model over paired four-band CHW NumPy arrays.

Files are matched by name between before and after folders. The arrays must
contain four channels in the same units as the training inputs, for example::

    before/tile_001.npy  <->  after/tile_001.npy

This script does not read SAFE products or preserve geospatial metadata. It is
an array-folder inference utility, not a georeferenced map-production tool.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese_unet import SiameseUNet
from utils.inference import (find_matching_pairs, sliding_window_predict,
                             summarize_pair, write_folder_summary)


def load(path):
    """Load one CHW four-band NumPy array as float32."""
    if not path.lower().endswith(".npy"):
        raise ValueError("folder inference accepts .npy arrays only")
    array = np.load(path).astype(np.float32)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError("expected a 2-D or 3-D array, got %s" % (array.shape,))
    return array


def load_pair(path_a, path_b, mean, std):
    """Load a pair of arrays and normalise with the training statistics."""
    a = load(path_a)
    b = load(path_b)
    if a.shape != b.shape:
        raise ValueError("shape mismatch: %s is %s, %s is %s"
                         % (path_a, a.shape, path_b, b.shape))
    if a.shape[0] != len(mean):
        raise ValueError("the array has %d channels but the checkpoint has %d"
                         % (a.shape[0], len(mean)))

    tensor_a = torch.from_numpy(a)
    tensor_b = torch.from_numpy(b)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    return (tensor_a - mean_t) / std_t, (tensor_b - mean_t) / std_t


def load_checkpoint(path, device):
    # The checkpoint holds tensors plus plain dicts, lists, floats and strings,
    # all of which weights_only=True permits. Restricting the unpickler prevents
    # arbitrary code execution from an untrusted checkpoint file.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    bands = checkpoint["bands"]
    model = SiameseUNet(in_channels=len(bands),
                        base=checkpoint.get("base", 16)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    return model, bands, mean, std, checkpoint


def comparison_figure(a, b, mask, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(a, cmap="gray")
    axes[0].set_title("Before")
    axes[1].imshow(b, cmap="gray")
    axes[1].set_title("After")
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted change")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/predictions")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.patch_size < 8 or args.patch_size % 8 != 0:
        parser.error("patch-size must be a positive multiple of 8 for this U-Net")
    if args.stride < 1 or args.stride > args.patch_size:
        parser.error("stride must be between 1 and patch-size")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    model, bands, mean, std, checkpoint = load_checkpoint(args.checkpoint, device)
    print("checkpoint epoch %s | bands %s | patch %d | stride %d | threshold %.2f"
          % (checkpoint.get("epoch"), bands, args.patch_size, args.stride,
             args.threshold))

    pairs = find_matching_pairs(args.before, args.after)
    if not pairs:
        print("no matching before/after pairs found")
        return
    print("%d matching pairs" % len(pairs))

    rows = []
    for index, (name, path_a, path_b) in enumerate(pairs, start=1):
        stem = os.path.splitext(name)[0]
        try:
            tensor_a, tensor_b = load_pair(path_a, path_b, mean, std)
        except (ValueError, OSError) as exc:
            print("[%d/%d] %s skipped: %s" % (index, len(pairs), name, exc))
            continue

        prob, mask = sliding_window_predict(
            model, tensor_a, tensor_b, args.patch_size, args.stride, device,
            batch_size=args.batch_size, threshold=args.threshold)
        assert mask is not None, "a threshold was given, so a mask is required"
        row = summarize_pair(name, mask.astype(np.uint8))
        rows.append(row)
        print("[%d/%d] %s: changed %.2f%% | unchanged %.2f%%"
              % (index, len(pairs), name, row["changed_pct"],
                 row["unchanged_pct"]))

        import matplotlib.image as mpimg
        mask_path = os.path.join(args.out, stem + "_mask.png")
        mpimg.imsave(mask_path, (mask.astype(np.uint8) * 255), cmap="gray",
                     vmin=0, vmax=255)
        try:
            comparison_figure(tensor_a[0].numpy(), tensor_b[0].numpy(), mask,
                              os.path.join(args.out, stem + "_figure.png"),
                              "%s | changed %.2f%%" % (name, row["changed_pct"]))
        except Exception as exc:      # a figure is not a result; do not abort
            print("    figure skipped:", exc)

    write_folder_summary(rows, os.path.join(args.out, "summary.csv"))


if __name__ == "__main__":
    main()
