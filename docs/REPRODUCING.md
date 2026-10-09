# Reproducing NeuFSD

Run the following commands in Bash from `experiments/`. You supply the traces;
the programs generate labels, counters, weights, predictions, and figures.
Use a fresh output directory: counter-store construction replaces its own
`input_store/`, and `run_full.sh` recreates the selected configuration's outputs.

## 1. Environment and inputs

Install a CUDA-compatible PyTorch/torchvision pair and `requirements.txt`.
A tested software combination is Python 3.10.12, PyTorch 2.6.0+cu124,
torchvision 0.21.0+cu124, and NumPy 2.2.6. The training scripts load the
bundled `local_deps/timm`.

```bash
export CUDA_VISIBLE_DEVICES=0
export COUNTER_BACKEND=memmap
export COUNTER_FEATURE_MODE=three
export COUNTER_INPUT_MODE=raw
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export DATALOADER_NUM_WORKERS=0
NEUFSD_WORK="$(mktemp -d "$PWD/neufsd-run.XXXXXX")"
NEUFSD_DATA=/absolute/path/to/data_full
NEUFSD_RUN="$NEUFSD_WORK/run"
NEUFSD_PRIOR="$NEUFSD_WORK/synthetic-prior/64_64"
```

Acquire the CAIDA2016, CAIDA2018, IMC, and MAWI traces described in the paper
from their providers. Preserve capture ordering, flow-key definitions, and
trace intervals. IMC/MAWI preprocessing reads timestamped PCAP captures and
writes 13-byte flow keys. CAIDA preprocessing reads timestamped fixed-width
records. The downstream counter and training programs read these layouts:

| Trace argument | Bytes/record | Key byte offset | Key bytes |
|---|---:|---:|---:|
| `caida_2016` | 16 | 8 | 8 |
| `caida_2018`, `caida_2018_new` | 21 | 0 | 13 |
| `caida_org`, `imc`, `mawi` | 13 | 0 | 13 |

For 13-byte five-tuple keys, retain source/destination IPv4 addresses, ports,
and protocol consistently. Do not merge directions or hash to a shorter key
when comparing full-five-tuple flows. The `caida_2016` format uses its own
8-byte key convention and is not interchangeable with the 13-byte format.
PCAP input supports classic Ethernet/VLAN and raw-IPv4 captures, including
gzip files; PCAPNG must first be converted to classic PCAP. IMC defaults to
IPv4 (zero ports for other protocols), and MAWI to TCP/UDP; noninitial
fragments and insufficient headers are excluded. The manifest records this
policy and filtering counts. `N_t` counts the retained packets. Input files
must be in chronological, nonoverlapping order; directory inputs use numeric
filename ordering. `--pcap-policy` selects the policy explicitly.

CAIDA files belong in `$NEUFSD_DATA/<trace>/caida/*.dat`, or can already be
split as `<trace>/caida_1min_split/dataset_0000.dat`, etc. Real-trace windows
span 60 seconds from the first capture timestamp, with half-open boundaries;
their packet counts are variable. The final partially observed window is
retained by default (`--last-window drop` excludes it). A manifest records
window boundaries and packet counts. Process contiguous capture intervals
separately: missing/empty intervals are rejected rather than compressed.

The 21-byte CAIDA2018 layout stores a little-endian float64 timestamp in
seconds at byte offset 13. For CAIDA2016, explicitly specify the timestamp
encoding of your input; its 8-byte prefix alone does not establish an encoding.
Key-only files contain no timing information and cannot be split into minutes.

```bash
DATA_FULL_ROOT="$NEUFSD_DATA" bash scripts/prepare_data.sh caida_2018

# Use original timestamped captures, not key-only exports.
for NEUFSD_TRACE in imc mawi; do
  python3 scripts/prepare_key13_dataset.py \
    --source "/absolute/path/to/${NEUFSD_TRACE}.pcap" --trace "$NEUFSD_TRACE" \
    --window-mode time --window-seconds 60 \
    --data-full-root "$NEUFSD_DATA" --run-root "$NEUFSD_RUN" --res 64_64 \
    --build-counter-store --start-seed 0 --end-seed 400 --workers 4
done
```

For CAIDA2016, first run `scripts/split_raw.py caida_2016 INPUT_DIR OUTPUT_DIR`
with `--timestamp-format` and `--timestamp-offset` matching your source, using
`$NEUFSD_DATA/caida_2016/caida_1min_split` as `OUTPUT_DIR`; then run
`DATA_FULL_ROOT="$NEUFSD_DATA" bash scripts/prepare_data.sh caida_2016`.
If the source `caida/` directory remains present, pass the same encoding via
`TIMESTAMP_FORMAT` and `TIMESTAMP_OFFSET` to that shell command as well.
See `split_raw.py --help` for supported timestamp encodings. Explicit
`--window-mode records --window-records 1000000` is available for separate
record-count experiments; those blocks are not elapsed-time windows.

The IMC/MAWI command also prepares the heavy inputs and evaluation links.
For CAIDA, prepare counter stores, labels, and the supplied heavy processor:

```bash
for NEUFSD_TRACE in caida_2016 caida_2018; do
  NEUFSD_FINAL="$NEUFSD_RUN/run_full_matrix/64_64_${NEUFSD_TRACE}_final"
  mkdir -p "$NEUFSD_FINAL/EL"
  ln -s "$NEUFSD_DATA/$NEUFSD_TRACE/tr_ts" "$NEUFSD_FINAL/tr_ts"
  python3 scripts/build_counter_store.py \
    --input-dir "$NEUFSD_DATA/$NEUFSD_TRACE/caida_1min_split" \
    --out-root "$NEUFSD_RUN/counter_store/64_64_$NEUFSD_TRACE/tr_ts" \
    --trace-format "$NEUFSD_TRACE" --counter-len 4096 \
    --start-seed 0 --end-seed 400
  g++ -O3 -std=c++17 -march=native -DNDEBUG \
    -I"configs/64_64_$NEUFSD_TRACE" \
    "configs/64_64_$NEUFSD_TRACE/heavy_processor.cpp" \
    -o "$NEUFSD_FINAL/heavy_processor"
  for NEUFSD_BLOCK in "$NEUFSD_DATA/$NEUFSD_TRACE/caida_1min_split/"*.dat; do
    NEUFSD_ID="${NEUFSD_BLOCK##*/}"
    NEUFSD_ID="${NEUFSD_ID%.dat}"
    NEUFSD_ID="$((10#${NEUFSD_ID##*_}))"
    mkdir -p "$NEUFSD_FINAL/EL/$NEUFSD_ID"
    "$NEUFSD_FINAL/heavy_processor" -i "$NEUFSD_BLOCK" \
      -d "$NEUFSD_FINAL/EL/$NEUFSD_ID" -b 0 -e 0 --m1 1000 --m2 10
  done
done
```

## 2. Build and train the prior

Generate the 60 synthetic distributions and both format-specific stores:

```bash
python3 scripts/generate_zipf_pretrain.py \
  --out-root "$NEUFSD_WORK/synthetic" --formats caida2016 caida2018 \
  --flows 80000 100000 \
  --alphas 0 0.2 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.5 3.0 \
  --packets 1000000 --seed 20260617

NEUFSD_ROOTS=()
for NEUFSD_FORMAT in caida2016 caida2018; do
  NEUFSD_ROOT="$NEUFSD_WORK/synthetic/$NEUFSD_FORMAT/tr_ts"
  python3 scripts/build_counter_store.py \
    --input-dir "$NEUFSD_WORK/synthetic/$NEUFSD_FORMAT/caida_1min_split" \
    --out-root "$NEUFSD_ROOT" --trace-format "$NEUFSD_FORMAT" \
    --counter-len 4096 --start-seed 0 --end-seed 400
  NEUFSD_ROOTS+=("$NEUFSD_ROOT")
done

for NEUFSD_HEAD in 1_10 10_1e4; do
  python3 scripts/train_pretrain.py --config-dir configs/64_64_caida_org \
    --roots "${NEUFSD_ROOTS[@]}" --head "$NEUFSD_HEAD" \
    --out-dir "$NEUFSD_PRIOR/ViT_${NEUFSD_HEAD}_results_1e-2" \
    --epochs 20 --batch-size 64 --lr 0.01 --val-tail-per-root 3 --seed 42
done
```

This primary recipe uses only the two synthetic roots: 30 datasets per root,
400 counter seeds per dataset, and the last three datasets of each root held
out for validation. It gives 24,000 candidates per network, split
21,600/2,400, and trains for 20 epochs. No real-trace windows enter this prior.
Output directories contain the `best_model_*.pth` checkpoints used below.

### Alternative mixed-data prior

To run the mixed-data configuration, add the first eight prepared windows from
each trace to the synthetic roots, and write separate checkpoints:

```bash
NEUFSD_MIXED_ROOTS=("${NEUFSD_ROOTS[@]}")
for NEUFSD_TRACE in caida_2016 caida_2018 imc mawi; do
  NEUFSD_ROOT="$NEUFSD_WORK/prior-data/${NEUFSD_TRACE}_first8"
  mkdir -p "$NEUFSD_ROOT"
  for NEUFSD_LABEL in 1_10_chazhi 10_1e4_chazhi 1_10_real 10_1e4_real; do
    ln -s "$NEUFSD_DATA/$NEUFSD_TRACE/tr_ts/$NEUFSD_LABEL" \
      "$NEUFSD_ROOT/$NEUFSD_LABEL"
  done
  python3 scripts/build_counter_store.py \
    --input-dir "$NEUFSD_DATA/$NEUFSD_TRACE/caida_1min_split" \
    --out-root "$NEUFSD_ROOT" --trace-format "$NEUFSD_TRACE" \
    --counter-len 4096 --start-seed 0 --end-seed 400 --limit 8
  NEUFSD_MIXED_ROOTS+=("$NEUFSD_ROOT")
done
for NEUFSD_HEAD in 1_10 10_1e4; do
  python3 scripts/train_pretrain.py --config-dir configs/64_64_caida_org \
    --roots "${NEUFSD_MIXED_ROOTS[@]}" --head "$NEUFSD_HEAD" \
    --out-dir "$NEUFSD_WORK/mixed-prior/64_64/ViT_${NEUFSD_HEAD}_results_1e-2" \
    --epochs 20 --batch-size 64 --lr 0.01 --val-tail-per-root 3 --seed 42
done
```

With eight windows per real trace, this alternative has 36,800 candidates,
split 29,600/7,200. Real windows 0–4 train the prior and 5–7 validate it;
evaluation starting at window 5 is therefore not prior-validation-disjoint.
It is not the primary synthetic-only configuration.

## 3. History-based adaptation

The following runs Avg-5 on CAIDA2018. Change `--trace`, `--config-dir`,
the cache/output names, and the ending window for other traces. For IMC/MAWI
use `configs/64_64_caida_org`.

```bash
NEUFSD_LAST_WINDOW=9  # Set from your prepared capture's window manifest.
python3 scripts/simulate_sliding_fullstack_online.py \
  --run-root "$NEUFSD_RUN" --res 64_64 --trace caida_2018 \
  --config-dir configs/64_64_caida_2018 --data-full-root "$NEUFSD_DATA" \
  --sample-cache-dir "$NEUFSD_WORK/sample-caida2018" \
  --out-dir "$NEUFSD_WORK/caida2018-avg5" \
  --pretrained-root "$NEUFSD_PRIOR" --init-from pretrained \
  --strategy avg5 --history-size 5 --start-minute 5 --end-minute "$NEUFSD_LAST_WINDOW" \
  --epochs 5 --batch-size 256 --lr 0.01 --mode full --holdout-frac 0.1 \
  --torch-seed 42 --sample-rate 0.1 --sample-seed 42 \
  --start-seed 0 --end-seed 400 --avg-replicas 5 --avg-replica-stride 10000 \
  --last-replicas 1 --last-replica-stride 10000 \
  --gpu-deadline-sec 60 --prep-workers 8 --counter-threads 2 \
  --gate-threshold 1.02 --gate-cap 1.20
```

Select inclusive window ranges from each capture's manifest, with at least
five preceding windows for Avg-5. Do not reuse record-block counts as minute
counts. Use the same capture interval and window boundaries across methods.

| Experiment | Change from the command above |
|---|---|
| Last, matched 2,000 examples | `--strategy last --last-replicas 5` |
| Window-5, matched 2,000 examples | `--strategy window5` |
| No pretraining | `--init-from scratch`; use a separate output directory |
| Sampling sensitivity | Change `--sample-rate` and use a separate sample cache |
| Epoch sensitivity | Change `--epochs`; retain the same inputs and evaluation range |
| One-time SFT | Prepare the first training root with `--prepare-cache-only --keep-train-roots`, then use `simulate_fixed_train_root_once.py` with that root's synthetic unit IDs |
| Mixed-data prior | Point `--pretrained-root` at the separate mixed-data checkpoints |

Use `--help` on the fixed-root runner for its `--train-root` and
`--train-minutes` arguments. It freezes weights after one update;
`--init-from scratch` alone still performs continuous adaptation.

## 4. Preceding-window IMC configuration

For window t, this path reloads the prior and adapts only on sampled labels
from window t−1, then decodes window t. Current-window sample-shape mixing is
disabled. The entry-point filename is retained for command compatibility.

```bash
python3 scripts/run_imc_current_oracle.py \
  --run-root "$NEUFSD_RUN" --trace imc --res 64_64 \
  --data-full-root "$NEUFSD_DATA" --out-dir "$NEUFSD_WORK/imc-previous" \
  --sample-cache-dir "$NEUFSD_WORK/sample-imc-q0p5" \
  --runtime-train-cache-dir "$NEUFSD_WORK/imc-previous-cache" \
  --pretrained-root "$NEUFSD_PRIOR" \
  --start-minute 5 --end-minute "$NEUFSD_LAST_WINDOW" --epochs 20 --batch-size 256 --lr 0.01 \
  --holdout-frac 0.1 --sample-rate 0.5 --sample-seed 42 --torch-seed 42 \
  --start-seed 0 --end-seed 2000 --train-replicas 1 --phi 100 --input-mode raw \
  --gate-threshold 1.02 --gate-cap 1.20 --gate-down-threshold 0.98 --gate-floor 0.70 \
  --sample-shape-weight 0
```

Set `NEUFSD_LAST_WINDOW` for the IMC capture before running this configuration.
The starting window must be at least 1. Use fresh output and training-cache
directories; outputs with another protocol are not resumed. Metrics record
`train_source_minute` separately from synthetic replica IDs in `train_minutes`.
Residual calibration reads each window's actual packet count from its raw
record file; retain those files and pass the matching `--data-full-root` to
online runners and performance sweeps.
The residual plotting tools also accept `--data-full-root`; the single-run
`summarize_pipeline_eval.py` instead requires `--raw-dir` and `--packet-size`
when light-mass correction is enabled.

## 5. Other runners, comparisons, and figures

The scripts below consume your generated outputs, not bundled result tables.
Inspect their `--help` and required input layout before launching a full run.

| Purpose | Entry point in `experiments/` |
|---|---|
| Independent per-block training, 64/128 resolution | `scripts/run_full.sh` |
| Eight independent-training configurations | `scripts/run_matrix_full.sh` |
| Online pretraining × continuous-training combinations | `scripts/run_four_mode_experiments.sh` |
| Baseline memory sweep | `compare/run_compare.sh`, `scripts/run_comprehensive_compare.py` |
| Pretraining/SFT ablations | `scripts/plot_pretrain_sft_ablation.py` |
| History-strategy curves | `scripts/plot_sft_strategy_comparison.py` |
| Residual calibration | `scripts/plot_residual_calibration_paper.py` |
| Per-block curves | `scripts/plot_minute_error.py` |
| Runtime performance sweeps | `scripts/perf_eval_sweeps_and_plots.py` |

The independent-training runner is **not** the pretrained Avg-5 runner.
After CAIDA data preparation, a sequential, single-GPU example is:

```bash
# This runner uses directory-based counter files rather than memmap stores.
COUNTER_BACKEND=files GPU0=0 GPU1=0 TRAIN_PARALLEL=0 \
  DATA_FULL_ROOT="$NEUFSD_DATA" RUN_ROOT="$NEUFSD_WORK/independent" \
  bash scripts/run_full.sh 64_64 caida_2018

DATA_FULL_ROOT="$NEUFSD_DATA" RUN_ROOT="$NEUFSD_WORK/independent" \
  bash compare/run_compare.sh caida_2018
```

The short baseline wrapper defaults to block 10; it is not a full-trace
aggregate benchmark. Its collector reads NeuFSD accuracy and timing from
`RUN_ROOT`; absent measurements stay blank. The comprehensive runner supplies
additional sweep logic. Both comprehensive plotting entry points require
`--ours-results caida2016=/path/a.csv caida2018=/path/b.csv imc=/path/c.csv mawi=/path/d.csv`
and `--ours-metric raw` (or a specified calibration metric). Fix these inputs
before evaluating; no CSV or metric is selected by its measured WMRD.
Use identical block ranges and metric definitions
for substantive comparisons. Plotting scripts have specific input naming
conventions; their path helpers document the required layout.

For finite data-plane insertion, compile the
[C++ smoke test](../dataplane/README.md). For switch profiling, follow the
[Tofino instructions](../p4/README.md); an SDE installation and compatible
switch/testbed are required. Redis/BESS deployments require integration with
those systems; this repository supplies algorithm interfaces, not a packaged
Redis/BESS service.

Training and evaluation create result CSVs, logs, checkpoints, and figures
locally. They are intentionally excluded from version control.
