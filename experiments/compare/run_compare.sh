#!/bin/bash
# 全量对比复现(单 trace)：编译 Elastic/MRAC(sketch_test) + Array/Hash(traditional_sample),
# 在我们的真实切片上跑 memory∈{16,32,64,128,256}KB,collect + 画图 (Ours vs 4 baselines)。
# 用法: bash compare/run_compare.sh <TRACE>     TRACE=caida_2016|caida_2018|caida_2018_new|caida_org
# 环境变量: WINDOW(用哪个测试切片号, 默认10) PYTHON DATA_FULL_ROOT RUN_ROOT
set -euo pipefail
TRACE="${1:?TRACE}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
DATA_FULL_ROOT="${DATA_FULL_ROOT:-$ROOT/data_full}"
RUN_ROOT="${RUN_ROOT:-$ROOT/run}"
export RUN_ROOT
CDIR="$ROOT/compare/$TRACE"; DATA="$DATA_FULL_ROOT/$TRACE/caida_1min_split"
WINDOW="${WINDOW:-10}"; MEMS="16384 32768 65536 131072 262144"
PY="${PYTHON:-python3}"
[ -d "$CDIR" ] || { echo "❌ 无 compare 配置 $CDIR"; exit 1; }
[ -d "$DATA" ] || { echo "❌ 缺数据 $DATA"; exit 1; }
cd "$CDIR"

echo "=== [$TRACE] build elastic/mrac (make) + array/hash (g++) ==="
make >/dev/null 2>&1 && echo "  sketch_test ok" || { echo "make failed"; make; exit 1; }
g++ -std=c++11 -O3 -mavx2 -mbmi -mbmi2 -Wno-psabi -o traditional_sample traditional_sample.cpp src/common/BOBHash32.cpp -I. && echo "  traditional_sample ok"

# 用我们的测试切片作为数据 (data/<W>.dat -> dataset_<W>.dat; sketch_test 读 N-1, traditional 读 N)
printf -v DS "dataset_%04d.dat" "$WINDOW"
[ -f "$DATA/$DS" ] || { echo "❌ 缺窗口 $DATA/$DS"; exit 1; }
mkdir -p data; ln -sfn "$DATA/$DS" "data/${WINDOW}.dat"
rm -f *_array_results.csv *_hash_results.csv elastic_*.txt mrac_*.txt

for M in $MEMS; do KB=$((M/1024))
  echo "  [${KB}KB] elastic"; ./bin/sketch_test -t elastic -d data/ -s $((WINDOW+1)) -e $((WINDOW+1)) -m $M -r 1 > elastic_${M}.txt 2>&1
  echo "  [${KB}KB] mrac(慢)"; ./bin/sketch_test -t mrac    -d data/ -s $((WINDOW+1)) -e $((WINDOW+1)) -m $M -r 1 > mrac_${M}.txt 2>&1
  echo "  [${KB}KB] array/hash"; ./traditional_sample -t array -d data/ -s $WINDOW -e $WINDOW -m $M -o ${KB}_array_results.csv >/dev/null 2>&1
  ./traditional_sample -t hash -d data/ -s $WINDOW -e $WINDOW -m $M -o ${KB}_hash_results.csv >/dev/null 2>&1
done

echo "=== [$TRACE] collect + plot ==="
"$PY" "$ROOT/compare/collect_compare.py" "$TRACE"
MPLBACKEND=Agg "$PY" plot_compare.py >/dev/null 2>&1 || { echo "plot failed"; MPLBACKEND=Agg "$PY" plot_compare.py; }
echo "=== DONE compare/$TRACE — figures: ==="; ls -1 *.pdf 2>/dev/null
echo "--- com_error_wmrd.csv ---"; cat com_error_wmrd.csv
