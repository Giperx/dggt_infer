#!/usr/bin/env bash
# WideDrive render, then photometric, histogram-matched, CBSR, and PD.
#
# Inference scripts follow inference_nuscenes*.py: edit the globals at the top
# of the python file (DATA_DIR, SCENE_LIST, CKPT_PATH, OUTPUT_PATH). This
# launcher assumes those output paths are left at their defaults.
#
#   bash metrics/run_widedrive.sh              # 3-frame inference, then metrics
#   bash metrics/run_widedrive.sh single       # single-frame inference, then metrics
#   bash metrics/run_widedrive.sh benchmark    # timing only, no images or metrics
#   bash metrics/run_widedrive.sh metrics      # metrics on an existing multi-frame render
#
# Scored images are the raw composite when Difix is on (before_rgb), otherwise rgb.
# GT is the dense camera-2 wide image at 1554x294.
# CBSR and PD do not use GT. widedrive_CRCS.py and widedrive_IPS.py are kept
# for comparison and are not called.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-multi}"
GT_ROOT="data/datasets/WideDrive_processed/sparseWideFOVImages3_1554x294"
VAL_LIST="data/datasets/WideDrive_processed/WideDriveVal/val.txt"

case "$MODE" in
  multi|metrics)
    SAVE_ROOT="outputs_widedrive/widedrive_multiframes_inference"
    ;;
  single)
    SAVE_ROOT="outputs_widedrive/widedrive_inference"
    ;;
  benchmark)
    ;;
  *)
    echo "usage: bash metrics/run_widedrive.sh [multi|single|benchmark|metrics]" >&2
    exit 1
    ;;
esac

if [[ "$MODE" == "multi" ]]; then
  echo "Inference -> $SAVE_ROOT"
  python inference_widedrive_multiframes.py
elif [[ "$MODE" == "single" ]]; then
  echo "Inference -> $SAVE_ROOT"
  python inference_widedrive.py
elif [[ "$MODE" == "benchmark" ]]; then
  echo "Benchmark (set WARMUP and MEASURE to override)"
  python benchmark_widedrive_multiframes.py
  exit 0
fi

echo "Metrics on $SAVE_ROOT"
python metrics/widedrive_metrics.py \
  --render-root "$SAVE_ROOT" \
  --gt-root "$GT_ROOT" \
  --val-list "$VAL_LIST"
python metrics/widedrive_HM.py \
  --render-root "$SAVE_ROOT" \
  --gt-root "$GT_ROOT" \
  --val-list "$VAL_LIST"
python metrics/eval_consistency.py \
  --render-root "$SAVE_ROOT" \
  --val-list "$VAL_LIST"
