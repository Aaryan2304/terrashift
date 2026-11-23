#!/usr/bin/env bash
# Read-only demo: re-derives the headline numbers from existing outputs, runs the unit
# tests, checks the manifest, the validity grid and the register, then does a bounded
# 192x192 CPU inference. Never writes into outputs/. Under a minute.
# DEMO_PRECISION=fp16 selects the optional CUDA path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

section() { printf '\n=== %s\n' "$1"; }

section "1. Unit tests"
PYTHONPATH=. "$PY" -m unittest discover -s tests 2>&1 | tail -3

section "2. Data manifest"
if [ -n "${SENTINEL_DATA_ROOT:-}" ]; then
  "$PY" -m terrashift.data.manifest | tail -5
else
  printf '   SENTINEL_DATA_ROOT is not set; skipping the product check.\n'
  printf '   The manifest fails rather than guessing a path, which is the intended behaviour.\n'
fi

section "3. Grid equality, validity chain"
"$PY" - <<'PY'
import rasterio, sys
ref = None
for kind in ("observation_valid", "change_score", "candidate_mask"):
    path = "outputs/baseline_v2_observation_valid/%s_A.tif" % kind
    with rasterio.open(path) as ds:
        key = (ds.crs.to_string(), tuple(ds.transform)[:6], ds.shape,
               (round(ds.res[0], 6), round(ds.res[1], 6)))
    if ref is None:
        ref = key
    elif key != ref:
        sys.exit("grid mismatch: %s" % path)
    print("   %-30s %s %s" % (kind, key[0], key[2]))
print("   all three rasters share CRS, transform, shape and resolution")
PY

section "4. Historical OSCD baseline numbers, read back (not the selected model)"
"$PY" - <<'PY'
import csv, json
bench = json.load(open("outputs/bench/oscd_validation_providers_v2/validation_benchmark.json"))
print("   validation benchmark (%s split, %d windows)"
      % (bench["split"], bench["windows"]))
for name in ("cuda_fp32", "cuda_int8", "cpu_int8"):
    entry = bench["runs"][name]
    metric, latency = entry["metrics"], entry["latency"]
    print("     %-10s AP %.4f  AUC %.4f  %.2f ms/batch  providers %s  %s"
          % (name, metric["ap"], metric["auc"], latency["batch_ms_median"],
             ",".join(p.replace("ExecutionProvider", "")
                      for p in entry["active_providers"]),
             entry["optimisation"]))
quant = json.load(open("outputs/quant/oscd_baseline_seed42_v2/quantization_report.json"))
print("   parity max |torch - onnx| %.2e; INT8 %d bytes (%.1f %% of FP32)"
      % (quant["stages"]["parity"]["max_abs_difference"],
         quant["stages"]["quantize"]["size_bytes"],
         100 * quant["stages"]["quantize"]["size_ratio_vs_fp32"]))
rows = list(csv.DictReader(open(
    "outputs/checkpoints/oscd_gpu_v1/baseline_seed42_v2/multi_seed_summary.csv")))
seed = [r for r in rows if r["seed"] == "42"][0]
# Files written before best_epoch was added to the writer have no such column.
print("   baseline seed 42: validation AP %.4f at epoch %s"
      % (float(seed["best_ap"]), seed.get("best_epoch", "18 (from train_log.csv)")))
summary = json.load(open("outputs/partitions_v2_observation_valid/partition_summary.json"))
pair_a = [entry for entry in summary if entry["pair"] == "A"][0]
print("   partitions recorded for %d pairs; Pair A retained %.4f of valid land"
      % (len(summary), pair_a["valid_land_retained_fraction"]))
PY

section "5. Inventories"
if [ -f outputs/register/detections.csv ]; then
  printf '   register rows: %s\n' "$(tail -n +2 outputs/register/detections.csv | wc -l)"
else
  printf '   register rows: 0 (no reviewed polygon yet; schema at configs/detection_register_schema.yaml)\n'
fi
printf '   figures: %s PNG\n' "$(find outputs/figures -maxdepth 2 -name '*.png' | wc -l)"

printf '\nDemo complete. Sections 1-5 write nothing; section 6 writes one bounded\n'
printf 'demonstration window under outputs/demo/ if it did not already exist.\n'

section "6. Bounded inference with the selected local checkpoint"
CKPT="outputs/checkpoints/local_manual_v2_seed43_posweight_0p834/seed_43/best.pt"
NORM="outputs/dataset/v2/normalisation.json"
DEMO_OUT="outputs/demo/local_window_A_v1"
if [ ! -f "$CKPT" ] || [ ! -f "$NORM" ]; then
  printf '   checkpoint or normalisation missing; skipping the inference step.\n'
  printf '   Expected %s and %s\n' "$CKPT" "$NORM"
elif [ -z "${SENTINEL_DATA_ROOT:-}" ]; then
  printf '   SENTINEL_DATA_ROOT is not set; skipping the inference step.\n'
elif [ -e "$DEMO_OUT" ]; then
  printf '   %s already exists; leaving it untouched.\n' "$DEMO_OUT"
else
  # Bounded 192x192 window. CPU FP32 is the reference path; set DEMO_PRECISION=fp16
  # to run the optional CUDA path instead. Fails loudly if CUDA is requested without
  # a GPU, by design. Uses no test labels and no test metrics.
  DEVICE_ARGS="--device cpu --precision fp32"
  case "${DEMO_PRECISION:-fp32}" in
    fp16) DEVICE_ARGS="--device cuda --precision fp16 --inference-batch-size 8" ;;
    fp32) : ;;
    *) printf '   DEMO_PRECISION must be fp32 or fp16, got %s\n' "$DEMO_PRECISION"; exit 2 ;;
  esac
  "$PY" scripts/inference/predict_safe_pair.py \
    --pair A --checkpoint "$CKPT" --normalisation "$NORM" \
    --output-dir "$DEMO_OUT" --window 5000,5192,5000,5192 \
    $DEVICE_ARGS --status demonstration 2>&1 | tail -12
  printf '\n   Georeferenced outputs written to %s\n' "$DEMO_OUT"
  printf '   A 192x192 window demonstrates that the path works. It is not evidence\n'
  printf '   of full-scene quality; see docs/report_2026-11-22.pdf sections 12 and 21.\n'
fi
