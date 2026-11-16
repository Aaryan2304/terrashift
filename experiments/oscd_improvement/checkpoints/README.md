# Experiment checkpoints

One directory per training run. The name says what the run is.

| Directory | Status | What it is |
|---|---|---|
| `loss_bce_seed42` | valid | BCE only (dice_weight 0), otherwise identical to the baseline |
| `loss_focal_seed42` | valid | focal loss, gamma 2, same pos_weight |
| `arch_resnet18_seed42` | valid | Siamese U-Net with a pretrained ResNet-18 encoder |
| `_smoke_loss_flag` | smoke, 1 epoch | pipeline check that `--loss` reaches the OSCD trainer |
| `_smoke_resnet` | smoke, 1 epoch | pipeline check that the ResNet variant trains at batch 8 |
| `_invalid_loss_bce_seed42` | **invalid** | a run that silently ignored `--loss` |
| `_invalid_loss_focal_seed42` | **invalid** | same, duplicate of the one above |

## About the two invalid runs

The first attempt at the loss comparison produced two directories that looked
correct and were not. `--dataset oscd_original` dispatches to
`train_oscd_original.run_training`, which builds its own model and criterion and
ignored both new flags, so both runs trained with `DiceBCELoss` and reproduced
the baseline exactly. The tell was two runs printing identical epoch losses to
five decimals.

They are kept, renamed with an `_invalid_` prefix and with an `INVALID.md` in
each, rather than deleted, so the mistake stays visible. Their numbers must not
be cited.

The fix adds the same selection logic to `train_oscd_original.py`, and
`run_config.json` now records `loss` and `architecture` so a run states what it
actually did instead of relying on its directory name.

## Baseline

Not here. The baseline checkpoint is
`outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/seed_42/best.pt` and is
untouched by these experiments.
