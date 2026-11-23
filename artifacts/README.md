# Submission artifacts

Frozen outputs of the pipeline, copied here so the repository is self-contained
without shipping the multi-gigabyte working tree. Every file is listed with its
SHA-256 in `manifest.json`. These are copies: the authoritative versions live in
`outputs/`, which stays local.

## models/

| file | size | role |
|---|---|---|
| `best.pt` | 5.7 MB | selected checkpoint, seed 43, `pos_weight` 0.834, epoch 7 |
| `onboard_fp32.onnx` | 1.9 MB | ONNX export, dynamic batch/height/width |
| `onboard_int8.onnx` | 540 KB | static INT8 quantisation, 28.3 % of FP32 |
| `oscd_best.pt` | 5.6 MB | OSCD-trained checkpoint; the initialisation the local fine-tune used |

`best.pt` is the checkpoint every metric in the report comes from. `oscd_best.pt` is the
stage-one model: `train.py --dataset oscd_original` trains it, and
`train.py --dataset local_manual --init-checkpoint artifacts/models/oscd_best.pt` is the
fine-tune the reported figures come from. Load it with
`terrashift.inference.checkpoints.load_model`. Input bands are B02, B03, B04, B08,
each normalised by the per-pair statistics in `outputs/dataset/v2/normalisation.json`
(not shipped; regenerate with `scripts/utilities/build_dataset.py normalise`).

## results/

`test_metrics.json` is the locked-test evaluation: pooled AP 0.653204, AUC 0.747176,
macro AP 0.667955, Pair A 0.969537, Pair B 0.366373. `registration.json` holds the
ECC alignment parameters per pair. `quantization_report.json` records the FP32/INT8
comparison from `quantize.py`.

## register/

`detections.csv` is the reviewed detection register: 26 unique locations with type,
extent, confidence and evidence references. `candidate_provenance.csv` links each
register row back to the candidate components that produced it.
`detections.geojson` is the same register as WGS84 features.

## evidence/

Chips and overlays for the 26 reviewed locations: before/after true colour plus a
candidate outline, one panel per location, and the contact sheet. These are the
images referenced by the register's evidence columns.

## What these are not

The candidate chips are model output under human review, not confirmed
infrastructure change. The register distinguishes `accepted_man_made`,
`uncertain` and rejected cases; only the accepted rows are claimed as detections.
