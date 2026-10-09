#!/bin/bash
# 全量忠实复现矩阵：8 配置【一个一个微调】(每块独立从头, 非连续), 统一 20 epoch。
# (注: 128/caida_org 的高频 trainer 原 regular 脚本 LR 打错成 1e-5, 已修回 1e-2;
#  参考结果用的就是 LR=1e-2 的 _con 变体。修好后全部 20ep 即收敛。)
# 跑完每个 config 删掉体积大的 counter，保留 ckpt + preds + plots（供后续画图/推理）。
# 已有结果的 config 自动跳过(复用)。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
DATA_FULL_ROOT="${DATA_FULL_ROOT:-$ROOT/data_full}"
RUN_ROOT="${RUN_ROOT:-$ROOT/run}"
export DATA_FULL_ROOT RUN_ROOT
OUT="$RUN_ROOT/matrix_summary"; mkdir -p "$OUT"
LOGS="$RUN_ROOT/_logs_full"; mkdir -p "$LOGS"
PY="${PYTHON:-python3}"; export PYTHON="$PY"
export END_SEED="${END_SEED:-400}" REPRO_EPOCHS="${REPRO_EPOCHS:-20}" GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" MAX_THREADS="${MAX_THREADS:-4}" DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}" TRAIN_PARALLEL="${TRAIN_PARALLEL:-1}"

data_ready(){
  local trace="$1"
  local df="$DATA_FULL_ROOT/$trace"
  [ -d "$df/caida_1min_split" ] && [ -f "$df/train_test_name_key.json" ] &&
  [ -d "$df/tr_ts/1_10_real" ] && [ -d "$df/tr_ts/1_10_chazhi" ] &&
  [ -d "$df/tr_ts/10_1e4_real" ] && [ -d "$df/tr_ts/10_1e4_chazhi" ] &&
  [ -d "$df/tr_ts_finetuned_continue/1_10_real" ] && [ -d "$df/tr_ts_finetuned_continue/1_10_chazhi" ] &&
  [ -d "$df/tr_ts_finetuned_continue/10_1e4_real" ] && [ -d "$df/tr_ts_finetuned_continue/10_1e4_chazhi" ]
}

CSV="$OUT/matrix_results_full.csv"; echo "res,trace,blocks,MRD,WMRD,status" > "$CSV"
for RES in 128_128 64_64; do for TRACE in caida_2016 caida_2018 caida_2018_new caida_org; do
  tag="${RES}_${TRACE}"; SM="$RUN_ROOT/${tag}_final/plots/pipeline_eval/summary_metrics.csv"
  if ! data_ready "$TRACE"; then
    echo ">>> SKIP $tag (missing prepared data: $DATA_FULL_ROOT/$TRACE/caida_1min_split)"
    echo "$RES,$TRACE,?,NA,NA,MISSING_DATA" >> "$CSV"
    continue
  fi
  if [ -f "$SM" ]; then echo ">>> SKIP $tag (已有结果，复用)"; else
    export REPRO_EPOCHS="${REPRO_EPOCHS:-20}"   # 全部 20ep 即收敛(128/caida_org 的 LR 已修 1e-5->1e-2)
    echo ">>> RUN $tag (one-by-one, ${REPRO_EPOCHS}ep, log: $LOGS/$tag.log)"
    bash "$HERE/run_full.sh" "$RES" "$TRACE" > "$LOGS/$tag.log" 2>&1 || { echo "  ❌ FAIL $tag"; }
    # 跑完删 counter（体积大、可重生），保留 ckpt/preds/plots/labels
    rm -rf "$RUN_ROOT/${tag}_exp/tr_ts/input" "$RUN_ROOT/${tag}_exp/tr_ts_finetuned_continue/input" 2>/dev/null || true
  fi
  if [ -f "$SM" ]; then
    line="$(grep OVERALL_AVG "$SM" | tail -1)"; mrd="$(echo "$line"|cut -d, -f2)"; wmrd="$(echo "$line"|cut -d, -f3)"
    blk="$("$PY" -c "import json;d=json.load(open('$RUN_ROOT/${tag}_exp/train_test_name_key.json'));print(sum(v for v in d.values())//5)" 2>/dev/null || echo '?')"
    cp -f "$SM" "$OUT/${tag}_summary.csv" 2>/dev/null || true
    printf '%s,%s,%s,%.4f,%.4f,OK\n' "$RES" "$TRACE" "$blk" "$mrd" "$wmrd" >> "$CSV"
    echo "    $tag: blocks=$blk MRD=$mrd WMRD=$wmrd"
  else
    echo "$RES,$TRACE,?,NA,NA,FAIL" >> "$CSV"
  fi
done; done
echo; echo "================= 全量忠实矩阵 ================="; column -t -s, "$CSV"
