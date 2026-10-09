#!/bin/bash
# =============================================================================
# 全量忠实复现：【一个一个微调】(one-by-one) × 20epoch（与原始 protocol 完全一致）
# -----------------------------------------------------------------------------
#   单配置驱动: 6 阶段 (sample_new -> counter -> ViT 训练/测试 -> Elastic -> 拼接评估 -> 计时/图)
#   * 用 data_full/<trace>/ 的【全部切片】(原始 train/test 划分，64≡128，md5 已核对)
#   * 训练/测试标签直接用从 sql 拷来的【原始 label】(= 原始 imcdc 产物，跳过慢解析)
#   * fine_ 微调流用 sample_new 现场重生成（确定性；counter 与包序无关，等价原始）
#   * 训练 = 一个一个微调：每个测试块都【独立从头 re-init CustomViT】再训 20 epoch，
#     非连续学习、无跨块权重累积、无预训练 (见 imcdc_train_test_*_vit.py 的 one-by-one re-init)
#   * 64 和 128 唯一区别 = counter 宽度 (64x64=16KB, 128x128=64KB)
#   注: 工作目录里 *_finetuned_continue / *_finetune_continue 是【原始命名】, 仅指"逐块微调流",
#       训练仍是一个一个独立从头 (勿被 continue 字样误导)。
# 用法: bash scripts/run_full.sh <RES> <TRACE>     RES=128_128|64_64
# 环境变量: END_SEED(400) REPRO_EPOCHS(20) MAX_THREADS(4) GPU0(0) GPU1(1) TRAIN_PARALLEL(1) PYTHON
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
DATA_FULL_ROOT="${DATA_FULL_ROOT:-$ROOT/data_full}"
RUN_ROOT="${RUN_ROOT:-$ROOT/run}"
if [ "${1:-}" = "clean" ]; then echo "🧹 清空 $RUN_ROOT"; rm -rf "$RUN_ROOT"; exit 0; fi
RES="${1:?RES}"; TRACE="${2:?TRACE}"
CFG="$ROOT/configs/${RES}_${TRACE}"; DF="$DATA_FULL_ROOT/${TRACE}"
[ -d "$CFG" ] || { echo "❌ 无配置 $CFG"; exit 1; }
[ -d "$DF/caida_1min_split" ] || { echo "❌ 无全量数据 $DF"; exit 1; }
for req in \
  "$DF/train_test_name_key.json" \
  "$DF/tr_ts/1_10_real" "$DF/tr_ts/1_10_chazhi" "$DF/tr_ts/10_1e4_real" "$DF/tr_ts/10_1e4_chazhi" \
  "$DF/tr_ts_finetuned_continue/1_10_real" "$DF/tr_ts_finetuned_continue/1_10_chazhi" \
  "$DF/tr_ts_finetuned_continue/10_1e4_real" "$DF/tr_ts_finetuned_continue/10_1e4_chazhi"; do
  [ -e "$req" ] || { echo "❌ 数据未准备完整，缺少 $req"; exit 1; }
done
EXP="$RUN_ROOT/${RES}_${TRACE}_exp"; FIN="$RUN_ROOT/${RES}_${TRACE}_final"
END_SEED="${END_SEED:-400}"; REPRO_EPOCHS="${REPRO_EPOCHS:-20}"; export REPRO_EPOCHS
MAX_THREADS="${MAX_THREADS:-4}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; PY="${PYTHON:-python3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"; export DATALOADER_NUM_WORKERS
LOCAL_DEPS="${LOCAL_DEPS:-$ROOT/local_deps}"
if [ -d "$LOCAL_DEPS" ]; then RUN_PYTHONPATH="$LOCAL_DEPS:$CFG"; else RUN_PYTHONPATH="$CFG"; fi
banner(){ echo; echo "########## [$RES/$TRACE] $* ##########"; }
echo "=== FULL $RES/$TRACE | END_SEED=$END_SEED EPOCHS=$REPRO_EPOCHS THREADS=$MAX_THREADS DATA_FULL_ROOT=$DATA_FULL_ROOT RUN_ROOT=$RUN_ROOT ==="

if [ "${ALLOW_CPU:-0}" != "1" ]; then
  for G in "$GPU0" "$GPU1"; do
    CUDA_VISIBLE_DEVICES="$G" "$PY" - <<'PYEOF'
import sys
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    sys.exit("CUDA is not available for the requested CUDA_VISIBLE_DEVICES")
print(f"  CUDA ok: {torch.cuda.get_device_name(0)}")
PYEOF
  done
fi

# 0. 工作目录 + 软链全量数据/原始标签（标签 = 原始 imcdc 产物，忠实）
rm -rf "$EXP" "$FIN"; mkdir -p "$EXP" "$FIN/EL"
ln -sfn "$DF/caida_1min_split" "$EXP/caida_1min_split"
rm -rf "$EXP/0_pretrained_weights"                          # 无预训练 -> 从头
mkdir -p "$EXP/tr_ts" "$EXP/tr_ts_finetuned_continue"
for seg in 1_10_real 1_10_chazhi 10_1e4_real 10_1e4_chazhi; do
  ln -sfn "$DF/tr_ts/$seg"                    "$EXP/tr_ts/$seg"
  ln -sfn "$DF/tr_ts_finetuned_continue/$seg" "$EXP/tr_ts_finetuned_continue/$seg"
done

# 1. 重生成 fine_ 微调流 + json，并核对划分与原始一致
banner "1/5 sample_new (regen fine_) + 核对划分"
( cd "$EXP" && PYTHONPATH="$RUN_PYTHONPATH" "$PY" "$CFG/sample_new.py" >/dev/null )
A=$(md5sum "$EXP/train_test_name_key.json" | cut -d' ' -f1)
B=$(md5sum "$DF/train_test_name_key.json"  | cut -d' ' -f1)
if [ "$A" = "$B" ]; then echo "  ✅ 划分与原始 md5 一致: $A"; else echo "  ❌ 划分不一致! local=$A orig=$B"; exit 1; fi
"$PY" - "$EXP/train_test_name_key.json" <<'PYEOF'
import json,sys; d=json.load(open(sys.argv[1]))
tr=sum(v for v in d.values()); te=sum(not v for v in d.values())
print(f"  连续块: {tr//5} 块 ({tr} 个 fine_ 训练 + {te} 个真实测试窗口)")
PYEOF

# 2. counter（全量切片 + fine_，per-res 宽度）
banner "2/5 gen counter (全量, END_SEED=$END_SEED)"
gcc -O3 -o "$EXP/gen" "$CFG/gen_ds.c"
gen_one(){ local f="$1" out="$2" n; n="$(basename "$f" .dat)"; mkdir -p "$out/input/$n"
  "$EXP/gen" --input "$f" --output-dir "$out/input/$n" --start-seed 0 --end-seed "$END_SEED" >/dev/null; }
export -f gen_one; export EXP END_SEED
cd "$EXP"
find -L caida_1min_split_finetune_continue -maxdepth 1 -name '*.dat' -print0 | xargs -0 -I{} -P "$MAX_THREADS" bash -c 'gen_one "$1" tr_ts_finetuned_continue' _ {}
find -L caida_1min_split                   -maxdepth 1 -name '*.dat' -print0 | xargs -0 -I{} -P "$MAX_THREADS" bash -c 'gen_one "$1" tr_ts' _ {}
echo "  counters: train=$(ls tr_ts_finetuned_continue/input | wc -l)  test=$(ls tr_ts/input | wc -l) 个数据集"

# 3. ViT 一个一个微调训练 + 测试（每块独立从头；标签直接用原始；两 head 并行）
banner "3/5 ViT 一个一个微调 (每块独立从头, EPOCHS=$REPRO_EPOCHS/块, GPU$GPU0/$GPU1)"
cd "$EXP"
if [ "${TRAIN_PARALLEL:-1}" = "0" ]; then
  CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$RUN_PYTHONPATH" "$PY" "$CFG/imcdc_train_test_10_vit.py"  > "$EXP/train_1_10.log" 2>&1 || { echo "❌ 低频训练失败"; tail -15 "$EXP/train_1_10.log"; exit 1; }
  CUDA_VISIBLE_DEVICES="$GPU1" PYTHONPATH="$RUN_PYTHONPATH" "$PY" "$CFG/imcdc_train_test_1e4_vit.py" > "$EXP/train_10_1e4.log" 2>&1 || { echo "❌ 高频训练失败"; tail -15 "$EXP/train_10_1e4.log"; exit 1; }
else
  CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$RUN_PYTHONPATH" "$PY" "$CFG/imcdc_train_test_10_vit.py"  > "$EXP/train_1_10.log" 2>&1 &  P1=$!
  CUDA_VISIBLE_DEVICES="$GPU1" PYTHONPATH="$RUN_PYTHONPATH" "$PY" "$CFG/imcdc_train_test_1e4_vit.py" > "$EXP/train_10_1e4.log" 2>&1 & P2=$!
  F=0; wait $P1 || F=1; wait $P2 || F=1
  [ $F -eq 0 ] || { echo "❌ 训练失败"; tail -15 "$EXP/train_1_10.log" "$EXP/train_10_1e4.log"; exit 1; }
fi

# 4. Elastic Sketch 基线（全部测试切片）
banner "4/5 Elastic Sketch"
g++ -O3 -std=c++17 -march=native -DNDEBUG -I"$CFG" -o "$FIN/heavy_processor" "$CFG/heavy_processor.cpp"
cd "$FIN"
for f in "$DF"/caida_1min_split/*.dat; do idx="$(basename "$f" .dat)"; idx="$((10#${idx##*_}))"; mkdir -p "EL/$idx"
  ./heavy_processor -i "$f" -d "EL/$idx" -b 0 -e 0 --m1 1000 --m2 10 >/dev/null; done

# 5. 拼接评估
banner "5/6 plot_final"
cd "$FIN"
ln -sfn "$EXP/0_finetuned_results" 0_finetuned_results
ln -sfn "$EXP/tr_ts" tr_ts
cp -f "$EXP/train_test_name_key.json" train_test_name_key.json
MPLBACKEND=Agg PYTHONPATH="$RUN_PYTHONPATH" "$PY" "$CFG/plot_final.py"

# 6. 计时(每样本推理耗时) + counter 可视化(复现 prev 的 obtain_counter_plot)
banner "6/6 计时 + counter 图"
IMG=$([ "$RES" = "128_128" ] && echo 128 || echo 64)
DS=$(ls "$EXP/tr_ts/input" 2>/dev/null | head -1)
if [ -n "$DS" ]; then
  PYTHONPATH="$RUN_PYTHONPATH" "$PY" "$ROOT/src/obtain_counter_plot.py" "$EXP/tr_ts/input/$DS" "$FIN/counter_images" "$IMG" 10 || echo "  counter 图跳过"
  CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$RUN_PYTHONPATH" "$PY" "$ROOT/src/time_infer.py" "$RES" "$EXP" || echo "  计时跳过"
else echo "  (counter 已清, 跳过计时/图; 单跑 run_full.sh 时会保留 counter)"; fi

echo "=== DONE $RES/$TRACE ==="; grep OVERALL_AVG "$FIN/plots/pipeline_eval/summary_metrics.csv"
