"""Smoke test through the production training entry point, not a copy of it.

Calls ``train_local_manual.run_training`` with the v2 dataset, five optimizer steps
per seed and one validation batch, writing to a throwaway directory. The production
checkpoint is never touched.

    python scripts/inference/smoke_production_path.py --output outputs/smoke/production
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="outputs/dataset/v2")
    parser.add_argument("--checkpoint",
                        default="outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt")
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit("refusing to overwrite %s" % args.output)

    with open("config.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    # Point the production code at the built dataset and its statistics file.
    cfg["data"]["local_patch_index"] = str(Path(args.root) / "train_index.csv")
    cfg["training"]["amp"] = True

    class RunArgs:
        normalisation = str(Path(args.root) / "normalisation.json")
        init_checkpoint = args.checkpoint
        pos_weight = None
        max_train_batches = 5
        max_val_batches = 1

        def __repr__(self):
            return "smoke-args"

    from train_local_manual import run_training

    summaries = run_training(cfg, RunArgs(), args.output, 1, 4, 16, 1e-4, 2, 0.5,
                             [42], torch.device("cuda"))
    run_dir = Path(args.output) / "seed_42"
    resolved = json.loads((run_dir / "run_config.json").read_text())
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    report = {
        "summaries": summaries,
        "precision": resolved.get("precision"),
        "checkpoint_precision": checkpoint.get("precision"),
        "train_log": (run_dir / "train_log.csv").read_text().splitlines(),
        "smoke_checkpoint": str(run_dir / "best.pt"),
        "note": "five optimizer steps; not a trained model",
    }
    out = Path(args.output) / "production_smoke_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("precision", "checkpoint_precision",
                                            "train_log", "summaries")}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
