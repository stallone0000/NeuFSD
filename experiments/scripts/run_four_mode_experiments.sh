#!/bin/bash
# Four-mode MRAC experiment:
#   pretrained? x continuous? on caida_2016, caida_2018, caida_2018_new.
#
# This script keeps every new output under RUN_ROOT and does not overwrite the
# original independent/scratch reproduction results.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-/usr/bin/python3}"
RUN_ROOT="${RUN_ROOT:-$ROOT/four_mode_runs}"
DATA_FULL_ROOT="${DATA_FULL_ROOT:-$ROOT/data_full}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-$ROOT/run_full_matrix}"

GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
MAX_THREADS="${MAX_THREADS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
INCLUDE_CAIDA_PRETRAIN="${INCLUDE_CAIDA_PRETRAIN:-0}"
CAIDA_PRETRAIN_SLICES="${CAIDA_PRETRAIN_SLICES:-5}"
SMOKE="${SMOKE:-0}"

if [ "$SMOKE" = "1" ]; then
  TRACES="${TRACES:-caida_2016 caida_2018}"
  RES_LIST="${RES_LIST:-64_64}"
  ONLINE_EPOCHS="${ONLINE_EPOCHS:-1}"
  PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
  PRETRAIN_VAL_TAIL_PER_ROOT="${PRETRAIN_VAL_TAIL_PER_ROOT:-0}"
  END_SEED="${END_SEED:-8}"
  MAX_TRAIN_BLOCKS="${MAX_TRAIN_BLOCKS:-2}"
  ZIPF_FORMATS="${ZIPF_FORMATS:-caida2016 caida2018}"
  ZIPF_FLOWS="${ZIPF_FLOWS:-80000}"
  ZIPF_ALPHAS="${ZIPF_ALPHAS:-1.0}"
  ZIPF_LIMIT="${ZIPF_LIMIT:-0}"
  STORE_LIMIT="${STORE_LIMIT:-30}"
else
  TRACES="${TRACES:-caida_2016 caida_2018 caida_2018_new}"
  RES_LIST="${RES_LIST:-128_128 64_64}"
  ONLINE_EPOCHS="${ONLINE_EPOCHS:-20}"
  PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
  PRETRAIN_VAL_TAIL_PER_ROOT="${PRETRAIN_VAL_TAIL_PER_ROOT:-3}"
  END_SEED="${END_SEED:-400}"
  ZIPF_FORMATS="${ZIPF_FORMATS:-caida2016 caida2018}"
  ZIPF_FLOWS="${ZIPF_FLOWS:-80000 100000}"
  ZIPF_ALPHAS="${ZIPF_ALPHAS:-0.0 0.2 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.5 3.0}"
  MAX_TRAIN_BLOCKS="${MAX_TRAIN_BLOCKS:-0}"
  ZIPF_LIMIT="${ZIPF_LIMIT:-0}"
  STORE_LIMIT="${STORE_LIMIT:-0}"
fi
export DATALOADER_NUM_WORKERS

mkdir -p "$RUN_ROOT"/{logs,tools}
mkdir -p "$RUN_ROOT"/{.tmp,.cache,.config/matplotlib}
export TMPDIR="$RUN_ROOT/.tmp"
export XDG_CACHE_HOME="$RUN_ROOT/.cache"
export MPLCONFIGDIR="$RUN_ROOT/.config/matplotlib"
export TORCH_HOME="$RUN_ROOT/.cache/torch"

trace_format() {
  case "$1" in
    caida_2016) echo caida2016 ;;
    caida_2018|caida_2018_new) echo caida2018 ;;
    imc|mawi) echo key13 ;;
    *) echo "unknown trace $1" >&2; return 1 ;;
  esac
}

counter_len() {
  case "$1" in
    128_128) echo 16384 ;;
    64_64) echo 4096 ;;
    *) echo "unknown res $1" >&2; return 1 ;;
  esac
}

cfg_dir() {
  echo "$ROOT/configs/${1}_${2}"
}

py_path() {
  local cfg="$1"
  if [ -d "$ROOT/local_deps" ]; then
    echo "$ROOT/src:$ROOT/local_deps:$cfg"
  else
    echo "$ROOT/src:$cfg"
  fi
}

build_store() {
  local input_dir="$1" out_root="$2" fmt="$3" clen="$4" limit="${5:-0}"
  if [ -f "$out_root/input_store/index.json" ]; then
    echo "  store exists: $out_root/input_store"
    return 0
  fi
  OMP_NUM_THREADS="$MAX_THREADS" "$PY" "$ROOT/scripts/build_counter_store.py" \
    --input-dir "$input_dir" \
    --out-root "$out_root" \
    --trace-format "$fmt" \
    --counter-len "$clen" \
    --start-seed 0 \
    --end-seed "$END_SEED" \
    --tmp-dir "$RUN_ROOT/.tmp_blocks" \
    ${limit:+--limit "$limit"}
}

link_labels() {
  local src="$1" dst="$2"
  mkdir -p "$dst"
  for sub in 1_10_real 1_10_chazhi 10_1e4_real 10_1e4_chazhi full_real; do
    [ -e "$src/$sub" ] || continue
    if [ -L "$dst/$sub" ] && [ "$(readlink "$dst/$sub")" = "$src/$sub" ]; then
      continue
    fi
    ln -sfn "$src/$sub" "$dst/$sub"
  done
  return 0
}

prepare_zipf_pretrain() {
  local zipf_root="$RUN_ROOT/zipf_exact"
  if [ ! -f "$zipf_root/manifest.json" ]; then
    echo "### generate exact Zipf pretrain data"
    "$PY" "$ROOT/scripts/generate_zipf_pretrain.py" \
      --out-root "$zipf_root" \
      --formats $ZIPF_FORMATS \
      --flows $ZIPF_FLOWS \
      --alphas $ZIPF_ALPHAS \
      --packets 1000000 \
      --seed 20260617 \
      ${ZIPF_LIMIT:+--limit "$ZIPF_LIMIT"}
  fi

  for res in $RES_LIST; do
    local clen; clen="$(counter_len "$res")"
    for fmt in caida2016 caida2018; do
      local src_root="$zipf_root/$fmt"
      local dst_root="$zipf_root/${fmt}_${res}"
      link_labels "$src_root/tr_ts" "$dst_root"
      build_store "$src_root/caida_1min_split" "$dst_root" "$fmt" "$clen"
    done
  done
}

prepare_caida_pretrain_root() {
  local res="$1" trace="$2"
  local fmt clen src dst
  fmt="$(trace_format "$trace")"
  clen="$(counter_len "$res")"
  src="$DATA_FULL_ROOT/$trace"
  dst="$RUN_ROOT/caida_pretrain/${res}_${trace}"
  link_labels "$src/tr_ts" "$dst"
  build_store "$src/caida_1min_split" "$dst" "$fmt" "$clen" "$CAIDA_PRETRAIN_SLICES" >&2
  echo "$dst"
}

run_pretrain_one() {
  local res="$1" head="$2" gpu="$3"
  local cfg roots out log
  cfg="$(cfg_dir "$res" caida_2018)"
  roots=("$RUN_ROOT/zipf_exact/caida2016_${res}" "$RUN_ROOT/zipf_exact/caida2018_${res}")
  if [ "$INCLUDE_CAIDA_PRETRAIN" = "1" ]; then
    for trace in $TRACES; do
      roots+=("$(prepare_caida_pretrain_root "$res" "$trace")")
    done
  fi
  out="$RUN_ROOT/pretrained/$res/$( [ "$head" = "1_10" ] && echo ViT_1_10_results_1e-2 || echo ViT_10_1e4_results_1e-2 )"
  if compgen -G "$out/best_model_*.pth" >/dev/null; then
    echo "pretrain exists: $out"
    return 0
  fi
  log="$RUN_ROOT/logs/pretrain_${res}_${head}.log"
  echo ">>> PRETRAIN $res $head GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$(py_path "$cfg")" "$PY" "$ROOT/scripts/train_pretrain.py" \
    --config-dir "$cfg" \
    --roots "${roots[@]}" \
    --head "$head" \
    --out-dir "$out" \
    --epochs "$PRETRAIN_EPOCHS" \
    --val-tail-per-root "$PRETRAIN_VAL_TAIL_PER_ROOT" \
    > "$log" 2>&1
}

prepare_online_exp() {
  local res="$1" trace="$2" mode_tag="$3"
  local fmt clen src exp store_base
  fmt="$(trace_format "$trace")"
  clen="$(counter_len "$res")"
  src="$DATA_FULL_ROOT/$trace"
  exp="$RUN_ROOT/online/${mode_tag}/${res}_${trace}_exp"
  store_base="$RUN_ROOT/counter_store/${res}_${trace}"
  mkdir -p "$exp/tr_ts" "$exp/tr_ts_finetuned_continue"
  cp -f "$src/train_test_name_key.json" "$exp/train_test_name_key.json"
  link_labels "$src/tr_ts" "$exp/tr_ts"
  link_labels "$src/tr_ts_finetuned_continue" "$exp/tr_ts_finetuned_continue"
  build_store "$src/caida_1min_split" "$store_base/tr_ts" "$fmt" "$clen" "$STORE_LIMIT"
  build_store "$src/caida_1min_split_finetune_continue" "$store_base/tr_ts_finetuned_continue" "$fmt" "$clen" "$STORE_LIMIT"
  ln -sfn "$store_base/tr_ts/input_store" "$exp/tr_ts/input_store"
  ln -sfn "$store_base/tr_ts_finetuned_continue/input_store" "$exp/tr_ts_finetuned_continue/input_store"
  echo "$exp"
}

ckpt_for() {
  local res="$1" head="$2" dir
  dir="$RUN_ROOT/pretrained/$res/$( [ "$head" = "1_10" ] && echo ViT_1_10_results_1e-2 || echo ViT_10_1e4_results_1e-2 )"
  ls "$dir"/best_model_*.pth 2>/dev/null | sort | head -1
}

run_online_head() {
  local res="$1" trace="$2" mode_tag="$3" ft_mode="$4" pretrain="$5" head="$6" gpu="$7"
  local cfg exp out_root log ckpt_arg=()
  cfg="$(cfg_dir "$res" "$trace")"
  exp="$RUN_ROOT/online/${mode_tag}/${res}_${trace}_exp"
  out_root="$exp/0_finetuned_results"
  if [ "$pretrain" = "1" ]; then
    local ckpt; ckpt="$(ckpt_for "$res" "$head")"
    [ -n "$ckpt" ] || { echo "missing pretrained ckpt for $res $head" >&2; return 1; }
    ckpt_arg=(--initial-ckpt "$ckpt")
  fi
  log="$RUN_ROOT/logs/online_${mode_tag}_${res}_${trace}_${head}.log"
  echo ">>> ONLINE $mode_tag $res $trace $head GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" COUNTER_BACKEND=memmap PYTHONPATH="$(py_path "$cfg")" "$PY" "$ROOT/scripts/train_online.py" \
    --config-dir "$cfg" \
    --exp-dir "$exp" \
    --raw-dir "$DATA_FULL_ROOT/$trace/caida_1min_split" \
    --trace "$trace" \
    --head "$head" \
    --mode "$ft_mode" \
    --out-root "$out_root" \
    --epochs "$ONLINE_EPOCHS" \
    --max-train-blocks "$MAX_TRAIN_BLOCKS" \
    "${ckpt_arg[@]}" \
    > "$log" 2>&1
}

plot_one() {
  local res="$1" trace="$2" mode_tag="$3" cfg exp final base_final
  [ "$MAX_TRAIN_BLOCKS" = "0" ] || return 0
  cfg="$(cfg_dir "$res" "$trace")"
  exp="$RUN_ROOT/online/${mode_tag}/${res}_${trace}_exp"
  final="$RUN_ROOT/online/${mode_tag}/${res}_${trace}_final"
  base_final="$BASELINE_RUN_ROOT/${res}_${trace}_final"
  mkdir -p "$final"
  ln -sfn "$exp/0_finetuned_results" "$final/0_finetuned_results"
  ln -sfn "$exp/tr_ts" "$final/tr_ts"
  ln -sfn "$base_final/EL" "$final/EL"
  cp -f "$exp/train_test_name_key.json" "$final/train_test_name_key.json"
  (cd "$final" && MPLBACKEND=Agg PYTHONPATH="$(py_path "$cfg")" "$PY" "$cfg/plot_final.py" > "$RUN_ROOT/logs/plot_${mode_tag}_${res}_${trace}.log" 2>&1)
}

echo "RUN_ROOT=$RUN_ROOT"
echo "TRACES=$TRACES RES_LIST=$RES_LIST END_SEED=$END_SEED ONLINE_EPOCHS=$ONLINE_EPOCHS PRETRAIN_EPOCHS=$PRETRAIN_EPOCHS SMOKE=$SMOKE STORE_LIMIT=$STORE_LIMIT"

prepare_zipf_pretrain

for res in $RES_LIST; do
  run_pretrain_one "$res" 1_10 "${GPU_LIST[0]}"
  run_pretrain_one "$res" 10_1e4 "${GPU_LIST[1]}"
done

modes=(
  scratch_independent independent 0
  scratch_continuous continuous 0
  pretrain_independent independent 1
  pretrain_continuous continuous 1
)

for res in $RES_LIST; do
  for trace in $TRACES; do
    i=0
    while [ $i -lt ${#modes[@]} ]; do
      mode_tag="${modes[$i]}"
      i=$((i+3))
      prepare_online_exp "$res" "$trace" "$mode_tag" >/dev/null
    done
  done
done

gpu_idx=0
jobs=()
for res in $RES_LIST; do
  for trace in $TRACES; do
    i=0
    while [ $i -lt ${#modes[@]} ]; do
      mode_tag="${modes[$i]}"; ft_mode="${modes[$((i+1))]}"; pretrain="${modes[$((i+2))]}"; i=$((i+3))
      for head in 1_10 10_1e4; do
        gpu="${GPU_LIST[$((gpu_idx % ${#GPU_LIST[@]}))]}"
        gpu_idx=$((gpu_idx + 1))
        run_online_head "$res" "$trace" "$mode_tag" "$ft_mode" "$pretrain" "$head" "$gpu" &
        jobs+=($!)
        if [ "${#jobs[@]}" -ge "${#GPU_LIST[@]}" ]; then
          wait -n
          mapfile -t jobs < <(jobs -pr)
        fi
      done
    done
  done
done
for pid in "${jobs[@]:-}"; do wait "$pid"; done

for res in $RES_LIST; do
  for trace in $TRACES; do
    for mode_tag in scratch_independent scratch_continuous pretrain_independent pretrain_continuous; do
      plot_one "$res" "$trace" "$mode_tag"
    done
  done
done

echo "DONE four-mode experiments"
