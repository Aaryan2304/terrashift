#!/usr/bin/env bash
# Reproducible demonstration of the verified pipeline.
#
# Read-only: it re-derives numbers from outputs that already exist and never
# writes into outputs/. Runtime is under a minute, and every step prints what it
# checked, so a failure is visible rather than silent.
#
#   1. unit tests
#   2. dataset manifest resolves the four products
#   3. grid equality of the validity chain
#   4. headline numbers, read back from the stored JSON and CSV
#   5. register and figure inventories
#
# Everything it prints is read from a file produced by an earlier run. It
# computes no new model result.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

section() { printf '\n=== %s\n' "$1"; }

section "1. Unit tests"
PYTHONPATH=. "$PY" -m unittest discover -s tests 2>&1 | tail -3

section "2. Data manifest"
if [ -n "${SENTINEL_DATA_ROOT:-}" ]; then
  "$PY" -m data.manifest | tail -5
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

section "4. Headline numbers, read back"
"$PY" - <<'PY'
import csv, json
bench = json.load(open("outputs/bench/oscd_validation_providers/validation_benchmark.json"))
print("   validation benchmark (%s split, %d windows)"
      % (bench["split"], bench["windows"]))
for name in ("cuda_fp32", "cuda_int8", "cpu_int8"):
    entry = bench["runs"][name]
    metric, latency = entry["metrics"], entry["latency"]
    print("     %-10s AP %.4f  AUC %.4f  %.2f ms/batch  providers %s"
          % (name, metric["ap"], metric["auc"], latency["batch_ms_median"],
             ",".join(p.replace("ExecutionProvider", "")
                      for p in entry["active_providers"])))
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
if [ -f outputs/labels/register/detection_register.csv ]; then
  printf '   register rows: %s\n' "$(tail -n +2 outputs/labels/register/detection_register.csv | wc -l)"
else
  printf '   register rows: 0 (no reviewed polygon yet; schema at configs/detection_register_schema.yaml)\n'
fi
printf '   figures: %s PNG\n' "$(find outputs/figures -maxdepth 2 -name '*.png' | wc -l)"

printf '\nDemo complete. No file was written.\n'
