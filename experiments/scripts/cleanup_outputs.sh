#!/bin/bash
# Trim bulky, reproducible artifacts after a config has produced final metrics.
# Keeps predictions, labels, metrics, plots, logs, and train/test metadata.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  cleanup_outputs.sh <RUN_ROOT> <TAG> [--keep-checkpoints] [--keep-counters] [--force]

Example:
  bash scripts/cleanup_outputs.sh /path/to/run_full_matrix 128_128_caida_2016

By default this removes:
  - <TAG>_exp/tr_ts/input
  - <TAG>_exp/tr_ts_finetuned_continue/input
  - best_model_*.pth under <TAG>_exp/0_finetuned_results

It refuses to run unless <TAG>_final/plots/pipeline_eval/summary_metrics.csv exists,
unless --force is passed.
EOF
}

[ "${1:-}" ] || { usage; exit 2; }
[ "${2:-}" ] || { usage; exit 2; }

RUN_ROOT="$1"
TAG="$2"
shift 2

KEEP_CHECKPOINTS=0
KEEP_COUNTERS=0
FORCE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --keep-checkpoints) KEEP_CHECKPOINTS=1 ;;
    --keep-counters) KEEP_COUNTERS=1 ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

EXP="$RUN_ROOT/${TAG}_exp"
FIN="$RUN_ROOT/${TAG}_final"
SUMMARY="$FIN/plots/pipeline_eval/summary_metrics.csv"

[ -d "$EXP" ] || { echo "skip: missing exp dir $EXP"; exit 0; }
if [ "$FORCE" != "1" ] && [ ! -f "$SUMMARY" ]; then
  echo "refusing cleanup: missing summary $SUMMARY" >&2
  exit 1
fi

if [ "$KEEP_COUNTERS" != "1" ]; then
  rm -rf "$EXP/tr_ts/input" "$EXP/tr_ts_finetuned_continue/input" 2>/dev/null || true
  echo "removed counter input dirs for $TAG"
fi

if [ "$KEEP_CHECKPOINTS" != "1" ] && [ -d "$EXP/0_finetuned_results" ]; then
  deleted="$(
    find "$EXP/0_finetuned_results" -type f -name 'best_model_*.pth' -print -delete 2>/dev/null | wc -l
  )"
  echo "removed $deleted checkpoint files for $TAG"
fi
