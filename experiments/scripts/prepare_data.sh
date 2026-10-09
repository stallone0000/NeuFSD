#!/bin/bash
# =============================================================================
# prepare_data.sh <TRACE>
#   从【你自己放好的真实切片】 data_full/<TRACE>/caida_1min_split/*.dat
#   一键生成 run_full.sh 需要的全部派生数据（确定性，可复现）：
#     - caida_1min_split_finetune_continue/   (fine_ 一个一个微调流, by sample_new.py)
#     - train_test_name_key.json              (滑窗 train/test 划分)
#     - tr_ts/{1_10,10_1e4}_{real,chazhi}/     (真实测试切片的标签, by imcdc)
#     - tr_ts_finetuned_continue/{...}/        (fine_ 训练切片的标签, by imcdc)
#   跑完即可:  bash scripts/run_full.sh 128_128 <TRACE>   (以及 64_64)
#
# 说明:
#   * 标签只依赖 .dat（与 counter 分辨率无关），故 data_full/<TRACE>/ 在 64/128 间共用,
#     prepare_data 只需按 TRACE 跑一次。默认用 configs/128_128_<TRACE>/ 的脚本(64≡128)。
#   * 每条 trace 的 .dat 字节格式不同，脚本会自动用该 trace 自己的 sample_new/imcdc(已适配)。
# 用法: bash scripts/prepare_data.sh caida_2018
#       PYTHON=/path/to/python bash scripts/prepare_data.sh caida_org
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
DATA_FULL_ROOT="${DATA_FULL_ROOT:-$ROOT/data_full}"
TRACE="${1:?用法: bash scripts/prepare_data.sh <TRACE>   (caida_2018|caida_2018_new|caida_2016|caida_org)}"
PY="${PYTHON:-python3}"
CFG="$ROOT/configs/128_128_${TRACE}"        # sample_new/imcdc 在 64/128 间逐字节相同
DF="$DATA_FULL_ROOT/${TRACE}"
[ -d "$CFG" ] || { echo "❌ 无配置 $CFG (TRACE 拼写?)"; exit 1; }

cd "$DF"
# 0) Timestamp windows require provenance; old million-record chunks are not minutes.
WINDOW_MODE="${WINDOW_MODE:-time}"
SPLIT_ARGS=(--window-mode "$WINDOW_MODE" --window-seconds "${WINDOW_SECONDS:-60}"
            --window-records "${WINDOW_RECORDS:-1000000}" --last-window "${LAST_WINDOW:-keep}")
[ -z "${TIMESTAMP_FORMAT:-}" ] || SPLIT_ARGS+=(--timestamp-format "$TIMESTAMP_FORMAT")
[ -z "${TIMESTAMP_OFFSET:-}" ] || SPLIT_ARGS+=(--timestamp-offset "$TIMESTAMP_OFFSET")
if [ -d "$DF/caida" ]; then
  echo "--- 0/3 split_raw: validate/split timestamped records -> caida_1min_split/ ---"
  "$PY" "$ROOT/scripts/split_raw.py" "$TRACE" "$DF/caida" "$DF/caida_1min_split" "${SPLIT_ARGS[@]}"
fi
[ -f "$DF/caida_1min_split/window_manifest.json" ] || { echo "Missing window provenance: regenerate timestamp windows with split_raw.py into a fresh DATA_FULL_ROOT; key-only bytes cannot establish minute windows."; exit 1; }
"$PY" - "$DF/caida_1min_split" "$TRACE" "$WINDOW_MODE" "$ROOT/scripts" "${WINDOW_SECONDS:-60}" "${WINDOW_RECORDS:-1000000}" "${LAST_WINDOW:-keep}" <<'PY'
from decimal import Decimal
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[4])
from timestamp_windows import sha256_file
root = Path(sys.argv[1])
manifest = json.loads((root / "window_manifest.json").read_text())
expected_size = {"caida_2016": 16, "caida_2018": 21, "caida_2018_new": 21, "caida_org": 13}[sys.argv[2]]
if manifest["config"]["window_mode"] != sys.argv[3] or manifest["config"]["output_record_size"] != expected_size:
    raise SystemExit("Window mode/record layout does not match the requested trace; use a fresh DATA_FULL_ROOT.")
config = manifest["config"]
if config["last_window"] != sys.argv[7] or (sys.argv[3] == "time" and config["window_seconds_ns"] != Decimal(sys.argv[5]) * 10**9) or (sys.argv[3] == "records" and config["window_records"] != int(sys.argv[6])):
    raise SystemExit("Window duration/count or final-interval policy changed; use a fresh DATA_FULL_ROOT.")
windows = manifest["windows"]
if not windows or [w["index"] for w in windows] != list(range(len(windows))):
    raise SystemExit("Window indices must be nonempty and contiguous.")
if {p.name for p in root.glob("*.dat")} != {w["file"] for w in windows}:
    raise SystemExit("Window files differ from the provenance manifest.")
for window in windows:
    path = root / window["file"]
    if path.stat().st_size != window["records"] * expected_size or sha256_file(path) != window["sha256"]:
        raise SystemExit(f"Window integrity validation failed: {path}")
PY
ndat=$(ls "$DF"/caida_1min_split/*.dat 2>/dev/null | wc -l)
[ "$ndat" -gt 0 ] || { echo "❌ $DF/caida_1min_split/ 里没有 .dat"; exit 1; }
echo "=== prepare_data [$TRACE] | $ndat windows (mode=$WINDOW_MODE) | DATA_FULL_ROOT=$DATA_FULL_ROOT | CFG=$CFG ==="
# 1) fine_ 一个一个微调流 + 滑窗划分 json (sample_new 读 caida_1min_split, 写 *_finetune_continue + json)
echo "--- 1/3 sample_new: 生成 fine_ 微调流 + train_test_name_key.json ---"
PYTHONPATH="$CFG" "$PY" "$CFG/sample_new.py"

# 2) 真实测试切片的标签: imcdc(caida_1min_split) -> tr_ts/{1_10,10_1e4}_{real,chazhi}
echo "--- 2/3 imcdc: 真实测试切片标签 -> tr_ts/ ---"
PYTHONPATH="$CFG" "$PY" "$CFG/imcdc_10.py"  caida_1min_split tr_ts
PYTHONPATH="$CFG" "$PY" "$CFG/imcdc_1e4.py" caida_1min_split tr_ts

# 3) fine_ 训练切片的标签: imcdc(*_finetune_continue) -> tr_ts_finetuned_continue/{...}
echo "--- 3/3 imcdc: fine_ 训练切片标签 -> tr_ts_finetuned_continue/ ---"
PYTHONPATH="$CFG" "$PY" "$CFG/imcdc_10.py"  caida_1min_split_finetune_continue tr_ts_finetuned_continue
PYTHONPATH="$CFG" "$PY" "$CFG/imcdc_1e4.py" caida_1min_split_finetune_continue tr_ts_finetuned_continue

echo ""
echo "=== 完成 [$TRACE]。data_full/$TRACE/ 现在含: ==="
echo "  $DF/caida_1min_split/($ndat) + caida_1min_split_finetune_continue/ + train_test_name_key.json"
echo "  tr_ts/{1_10,10_1e4}_{real,chazhi}/ + tr_ts_finetuned_continue/{...}/"
date -Is > "$DF/.prepare_data_complete"
echo "→ 现在可跑:  bash scripts/run_full.sh 128_128 $TRACE   (以及 64_64 $TRACE)"
