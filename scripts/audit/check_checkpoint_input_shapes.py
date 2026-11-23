"""Shape and loss smoke tests for the pretrained OSCD checkpoint.

Loads the checkpoint, rebuilds the architecture it recorded and runs forward passes at both candidate
patch sizes. Nothing is trained: no optimiser step, no gradients, no dataset.

Three questions. Does it load, and what input does it expect? Does a 256 x 256 tensor come out the
same size, which reduces to the downsampling divisibility this measures rather than assumes? And what
field of view does one output pixel see? A larger patch buys more independent views, not a wider one,
and the receptive field is measured by backpropagating a single output pixel.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

from terrashift.models.siamese_unet import DiceBCELoss, SiameseUNet
from terrashift.inference.checkpoints import enumerate_support

STATUS = "forward_pass_smoke_test_no_training"


def load_model(path):
    """Rebuild the model the checkpoint describes and load its weights strictly.

    ``weights_only=True`` is used deliberately: this checkpoint holds only
    tensors and plain metadata (ints, floats, lists, dicts), which the restricted
    loader accepts, so there is no reason to permit arbitrary unpickling.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise SystemExit("checkpoint has no model_state_dict")
    in_channels = int(state["enc1.0.weight"].shape[1])
    base = int(state["enc1.0.weight"].shape[0])
    model = SiameseUNet(in_channels=in_channels, base=base)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint, in_channels, base


def forward_smoke(model, size, pos_weight=1.0, batch=2, seed=0):
    """One forward pass at a given square size, plus a loss value."""
    generator = torch.Generator().manual_seed(seed)
    x1 = torch.randn(batch, model.enc1[0].in_channels, size, size, generator=generator)
    x2 = torch.randn(batch, model.enc1[0].in_channels, size, size, generator=generator)
    with torch.no_grad():
        logits = model(x1, x2)
        probabilities = torch.sigmoid(logits)
        target = (torch.rand(batch, 1, size, size, generator=generator) > 0.9).float()
        # Unweighted, then with the checkpoint's own pos_weight, so the loss is
        # exercised with both the plain and the training-time weighting.
        loss = DiceBCELoss(pos_weight=1.0)(logits, target)
        weighted_loss = DiceBCELoss(pos_weight=pos_weight)
        real_loss = weighted_loss(logits, target)
    result = {
        "input_shape": [batch, int(model.enc1[0].in_channels), size, size],
        "output_shape": list(logits.shape),
        "shape_preserved": tuple(logits.shape[2:]) == (size, size),
        "finite_output": bool(torch.isfinite(logits).all()),
        "output_min": float(logits.min()),
        "output_max": float(logits.max()),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "loss_finite": bool(torch.isfinite(loss)),
        "loss": float(loss),
    }
    result["loss_with_checkpoint_pos_weight"] = float(real_loss)
    result["loss_with_pos_weight_finite"] = bool(torch.isfinite(real_loss))
    return result


def accepted_sizes(model, candidates):
    """Which input sizes run, and which fail on the skip-connection concat."""
    results = {}
    for size in candidates:
        try:
            with torch.no_grad():
                x = torch.zeros(1, model.enc1[0].in_channels, size, size)
                out = model(x, x)
            results[str(size)] = {"accepted": True, "output": list(out.shape[2:])}
        except Exception as error:                     # noqa: BLE001
            results[str(size)] = {"accepted": False, "error": type(error).__name__,
                                  "detail": str(error).splitlines()[0][:160]}
    return results


def receptive_field(model, size, seeds=(0, 1, 2)):
    """Extent in input pixels that one central output pixel depends on.

    Random input is required: with a constant input every convolution output is
    constant, and the ReLU derivative of a constant non-positive value is zero
    everywhere, so the gradient vanishes and the measured extent is 0. Several
    seeds are used and the largest extent kept, because a single seed leaves some
    units in their dead zone and understates the field.
    """
    channels = model.enc1[0].in_channels
    best = {"extent": 0}
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        x1 = torch.randn(1, channels, size, size, generator=generator).requires_grad_(True)
        x2 = torch.randn(1, channels, size, size, generator=generator).requires_grad_(True)
        model.zero_grad()
        out = model(x1, x2)
        centre = size // 2
        out[0, 0, centre, centre].backward()
        gradient = x1.grad.abs().sum(dim=(0, 1))
        rows = torch.nonzero(gradient.sum(dim=1) > 0).flatten()
        cols = torch.nonzero(gradient.sum(dim=0) > 0).flatten()
        if rows.numel() == 0:
            continue
        extent = int(max(rows.numel(), cols.numel()))
        if extent > best["extent"]:
            best = {
                "first_row": int(rows.min()), "last_row": int(rows.max()),
                "first_col": int(cols.min()), "last_col": int(cols.max()),
                "rows": int(rows.numel()), "cols": int(cols.numel()),
                "extent": extent, "centre": centre, "seed": seed,
            }
    if not best.get("extent"):
        return {"extent": 0, "note": "no gradient reached the input at any seed"}
    return best


def theoretical_receptive_field(size):
    """Exact support extent from the graph, independent of weights and input.

    Propagates sets of contributing input indices through the same operations the
    network applies: 3x3 convolutions with padding 1, 2x2 max-pools, 2x2 stride-2
    transposed convolutions, skip concatenation, and the 1x1 head. Every ReLU and
    BatchNorm is treated as a linear identity, which is what makes this the
    theoretical field: an upper bound on what any output pixel *can* depend on.

    The empirical gradient probe measures something smaller, because ReLU gating
    switches some paths off for a given input.
    """
    def conv3(sets):
        n = len(sets)
        out = []
        for i in range(n):
            support = set()
            for j in (i - 1, i, i + 1):
                if 0 <= j < n:
                    support |= sets[j]
            out.append(support)
        return out

    def pool2(sets):
        return [sets[2 * i] | sets[2 * i + 1] for i in range(len(sets) // 2)]

    def conv_transpose2(sets):
        # ConvTranspose2d(kernel=2, stride=2): out[2i] and out[2i+1] both come
        # from in[i], with no overlap between neighbouring inputs.
        out = []
        for support in sets:
            out.append(support)
            out.append(support)
        return out

    def block(sets):
        return conv3(conv3(sets))

    def skip(a, b):
        return [left | right for left, right in zip(a, b)]

    x = [{i} for i in range(size)]
    e1 = block(x)
    e2 = block(pool2(e1))
    e3 = block(pool2(e2))
    e4 = block(pool2(e3))

    c3 = block(skip(conv_transpose2(e4), e3))
    c2 = block(skip(conv_transpose2(c3), e2))
    c1 = block(skip(conv_transpose2(c2), e1))

    centre = c1[size // 2]
    return {
        "extent": len(centre),
        "first": int(min(centre)),
        "last": int(max(centre)),
        "centre_index": size // 2,
    }


def gradient_energy_profile(model, size, seed=0):
    """Radial profile of the gradient reaching the input from one output pixel."""
    channels = model.enc1[0].in_channels
    generator = torch.Generator().manual_seed(seed)
    x1 = torch.randn(1, channels, size, size, generator=generator).requires_grad_(True)
    x2 = torch.randn(1, channels, size, size, generator=generator).requires_grad_(True)
    model.zero_grad()
    out = model(x1, x2)
    centre = size // 2
    out[0, 0, centre, centre].backward()
    energy = x1.grad.abs().sum(dim=(0, 1))
    total = float(energy.sum())
    if total <= 0:
        return {"total_energy": 0.0}
    rows = torch.arange(size).view(-1, 1)
    cols = torch.arange(size).view(1, -1)
    distance = torch.maximum((rows - centre).abs(), (cols - centre).abs())
    cumulative = {}
    for fraction in (0.5, 0.75, 0.9, 0.99):
        reached, radius = 0.0, 0
        for radius in range(size // 2 + 1):
            reached = float(energy[distance <= radius].sum()) / total
            if reached >= fraction:
                break
        cumulative["r_for_%.0f_pct" % (100 * fraction)] = {
            "radius_px": int(radius), "fraction": round(reached, 4)}
    support = distance[energy > 0]
    return {
        "total_energy": total,
        "non_zero_pixels": int((energy > 0).sum()),
        "max_radius_px": int(support.max()) if support.numel() else 0,
        "cumulative": cumulative,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-sizes", default="64,96,128,160,192,256,320,384")
    parser.add_argument("--probe-sizes", default="126,127,128,129,130,255,256,257")
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    model, checkpoint, in_channels, base = load_model(args.checkpoint)
    print("checkpoint loaded: %s" % args.checkpoint)
    print("  epoch %s | in_channels %d | base %d | parameters %d"
          % (checkpoint.get("epoch"), in_channels, base, model.parameter_count()))
    print("  recorded bands: %s" % checkpoint.get("bands"))
    print("  recorded pos_weight: %s" % checkpoint.get("pos_weight"))

    sizes = [int(value) for value in args.candidate_sizes.split(",")]
    probes = [int(value) for value in args.probe_sizes.split(",")]

    pos_weight = float(checkpoint.get("pos_weight") or 1.0)
    smoke = {str(size): forward_smoke(model, size, pos_weight=pos_weight)
             for size in (128, 256)}
    for size, result in smoke.items():
        print("  forward %s: out %s | finite %s | loss %.4f"
              % (size, result["output_shape"], result["finite_output"], result["loss"]))

    accepted = accepted_sizes(model, sorted(set(sizes + probes)))
    failed = [size for size, value in accepted.items() if not value["accepted"]]
    print("  sizes rejected: %s" % (failed or "none"))

    empirical = {str(size): receptive_field(model, size) for size in (128, 256, 320)}
    theoretical = {str(size): theoretical_receptive_field(size) for size in (128, 256, 320)}
    energy = {str(size): gradient_energy_profile(model, size) for size in (128, 256)}
    for size in ("128", "256", "320"):
        print("  receptive field at %s px input: theoretical %d px, empirical %d px"
              % (size, theoretical[size]["extent"], empirical[size]["extent"]))
    for size, profile in energy.items():
        if profile.get("total_energy"):
            print("  gradient energy at %s px: %d px carry it, 90%% within radius %d px"
                  % (size, profile["non_zero_pixels"],
                     profile["cumulative"]["r_for_90_pct"]["radius_px"]))

    report = {
        "status": STATUS,
        "checkpoint": os.path.abspath(args.checkpoint),
        "architecture": "SiameseUNet",
        "checkpoint_metadata": {
            "epoch": checkpoint.get("epoch"),
            "bands": checkpoint.get("bands"),
            "base": checkpoint.get("base"),
            "seed": checkpoint.get("seed"),
            "pos_weight": checkpoint.get("pos_weight"),
            "val_metrics": checkpoint.get("val_metrics"),
            "run_config": checkpoint.get("run_config"),
        },
        "parameters": model.parameter_count(),
        "input_channels": in_channels,
        "base_channels": base,
        "fixed_dimension_layers": {
            "linear": [name for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Linear)],
            "adaptive_pool": [name for name, module in model.named_modules()
                              if isinstance(module, torch.nn.AdaptiveAvgPool2d)],
            "note": ("none present: the network is fully convolutional, so it "
                     "imposes no fixed input size. The only constraint is that "
                     "every skip connection concatenates tensors of equal size, "
                     "which requires the input to be divisible by 2**3 = 8."),
        },
        "forward_pass": smoke,
        "size_sweep": accepted,
        "receptive_field": {
            "theoretical": theoretical,
            "theoretical_definition": ("support of the graph with ReLU and "
                                       "BatchNorm treated as identity: an upper "
                                       "bound on what an output pixel can depend on"),
            "empirical": empirical,
            "empirical_definition": ("input pixels with non-zero gradient from one "
                                     "central output pixel, over three random seeds"),
            "gradient_energy": energy,
        },
        "training_performed": False,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()


