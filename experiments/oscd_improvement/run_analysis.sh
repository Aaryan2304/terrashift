#!/usr/bin/env bash
# Post-training analysis for every experiment checkpoint.
#
# For each run: stitch validation predictions, sweep thresholds and
# post-processing, then render error figures. Analysis only: the test split is
# never read.
#
#   bash experiments/oscd_improvement/run_analysis.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
EXP=experiments/oscd_improvement

analyse() {
  local name=$1 checkpoint=$2
  echo "=== $name ==="
  "$PY" "$EXP/collect_predictions.py" \
    --checkpoint "$checkpoint" \
    --output "$EXP/predictions/${name}_val.npz" 2>&1 | tail -3
  "$PY" "$EXP/analyse_thresholds.py" \
    --predictions "$EXP/predictions/${name}_val.npz" \
    --output-dir "$EXP/results/$name" 2>&1 \
    | grep -E "AP |best pixel|best object|min   64" || true
  "$PY" "$EXP/make_figures.py" \
    --predictions "$EXP/predictions/${name}_val.npz" \
    --output-dir "$EXP/results/figures_$name" 2>&1 | tail -2
}

# Baseline: reuse the existing prediction file, re-run the analysis only if the
# results directory is missing.
if [ ! -d "$EXP/results/baseline_seed42" ]; then
  cp -r "$EXP/results/baseline_analysis_v2" "$EXP/results/baseline_seed42" 2>/dev/null || true
fi

for run in loss_bce_seed42 loss_focal_seed42 arch_resnet18_seed42; do
  checkpoint="$EXP/checkpoints/$run/seed_42/best.pt"
  if [ -f "$checkpoint" ]; then
    analyse "$run" "$checkpoint"
  else
    echo "skipping $run: no checkpoint"
  fi
done

"$PY" "$EXP/compare_checkpoints.py" --root "$EXP" \
  --output "$EXP/results/comparison.csv"
echo "=== analysis complete ==="
