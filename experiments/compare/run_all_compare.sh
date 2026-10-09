#!/bin/bash
# 全量复现 4 个 trace 的跨方法对比 (Ours vs Elastic/MRAC/Array/Hash):
#   每个 trace: 编译 sketch_test + traditional_sample -> 在我方测试切片上跑
#   memory∈{16,32,64,128,256}KB -> collect -> 画 4 张图
#   (caida_2016=16B / caida_org=13B / caida_2018(_new)=21B reader 已各自就位)
# 用法: bash compare/run_all_compare.sh [WINDOW]    (WINDOW 默认 10)
#   注: MRAC EM 解码每内存约 2min, 单 trace 约 10min, 4 trace 串行约 40min。
#       如需并行: 见 compare/run_remaining.sh 写法 (各 trace 目录独立, 可后台并行)。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WINDOW="${1:-10}"
for tr in caida_2018 caida_2018_new caida_2016 caida_org; do
  echo "########## $tr (window=$WINDOW) ##########"
  bash "$HERE/run_compare.sh" "$tr"
done
echo "########## 全部完成。各 trace 产物: compare/<trace>/*.pdf + com_error_{mrd,wmrd}.csv + {decode,insert}_time.csv ##########"
