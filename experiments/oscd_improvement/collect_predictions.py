"""Collect stitched validation predictions for offline analysis.

One forward pass over a checkpoint's validation split, written as per-city
probability and truth rasters so threshold sweeps, post-processing sweeps and
scene-level object metrics can be computed without touching the GPU again.

Validation only, by construction: the split is an argument but the default and
the only supported value used here is ``validation``. The official test regions
stay locked.

    python experiments/oscd_improvement/collect_predictions.py \
        --checkpoint outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt \
        --output experiments/oscd_improvement/predictions/baseline_val.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data.oscd_original import DEFAULT_BANDS, OSCDOriginalPatchDataset
from models.siamese_unet import SiameseUNet


def build_model(checkpoint, device):
    """Rebuild the architecture the checkpoint recorded, not an assumed one."""
    base = checkpoint.get("base", 16)
    bands = tuple(checkpoint.get("bands", DEFAULT_BANDS))
    run_config = checkpoint.get("run_config", {}) or {}
    arch = checkpoint.get("architecture") or run_config.get("architecture") or "siamese_unet"
    if arch == "siamese_resnet18":
        from models.siamese_resnet18 import SiameseResNet18
        model = SiameseResNet18(in_channels=len(bands), pretrained=False)
    else:
        model = SiameseUNet(in_channels=len(bands), base=base)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, bands, base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=["validation"])
    parser.add_argument("--index", default=None,
                        help="patch index; defaults to config data.oscd_patch_index")
    parser.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    cfg = yaml.safe_load(open(args.config))
    index_path = args.index or cfg["data"]["oscd_patch_index"]
    # weights_only=True: these checkpoints hold tensors plus plain dict/list/str
    # values (run_config, val_metrics), never custom classes, so unpickling
    # arbitrary objects is not needed.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, bands, base = build_model(checkpoint, args.device)

    dataset = OSCDOriginalPatchDataset(index_path, args.split, bands,
                                       checkpoint["mean"], checkpoint["std"])
    rows = dataset.rows
    print("checkpoint", os.path.basename(os.path.dirname(args.checkpoint)),
          "| arch", (checkpoint.get("run_config") or {}).get("architecture", "siamese_unet"),
          "| base", base, "| bands", bands,
          "| windows", len(dataset), "| device", args.device)

    # Canvas size per city from the patch origins.
    dims = {}
    for row in rows:
        city = row["city"]
        height = int(row["row_off"]) + int(row["height"])
        width = int(row["col_off"]) + int(row["width"])
        current = dims.get(city, (0, 0))
        dims[city] = (max(current[0], height), max(current[1], width))

    prob = {city: np.zeros(shape, dtype=np.float32) for city, shape in dims.items()}
    truth = {city: np.zeros(shape, dtype=np.uint8) for city, shape in dims.items()}
    covered = {city: np.zeros(shape, dtype=bool) for city, shape in dims.items()}

    for start in range(0, len(dataset), args.batch_size):
        stop = min(start + args.batch_size, len(dataset))
        samples = [dataset[i] for i in range(start, stop)]
        x1 = torch.stack([s["image1"] for s in samples]).to(args.device)
        x2 = torch.stack([s["image2"] for s in samples]).to(args.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(x1, x2))[:, 0].float().cpu().numpy()
        for offset, index in enumerate(range(start, stop)):
            row = rows[index]
            city = row["city"]
            r0, c0 = int(row["row_off"]), int(row["col_off"])
            h, w = int(row["height"]), int(row["width"])
            prob[city][r0:r0 + h, c0:c0 + w] = probabilities[offset]
            truth[city][r0:r0 + h, c0:c0 + w] = samples[offset]["mask"].numpy().astype(np.uint8)
            covered[city][r0:r0 + h, c0:c0 + w] = True
        print("  %d/%d" % (stop, len(dataset)), end="\r")
    print()

    for city in sorted(dims):
        missing = int((~covered[city]).sum())
        print("%-12s %s  uncovered pixels %d" % (city, dims[city], missing))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    payload = {"bands": np.array(bands)}
    for city in sorted(dims):
        payload["prob_" + city] = prob[city]
        payload["truth_" + city] = truth[city]
        payload["covered_" + city] = covered[city]
    np.savez_compressed(args.output, **payload)
    print("wrote", args.output, os.path.getsize(args.output), "bytes")


if __name__ == "__main__":
    main()
