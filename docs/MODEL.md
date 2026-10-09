# Model and training configuration

## Algorithm and implementation

NeuFSD represents traffic with counters and a heavy-flow branch. The decoder
turns counters into flow-size counts; a residual-mass correction can calibrate
the small-flow estimate before it is combined with the heavy-flow estimate.

| Component | Source |
|---|---|
| Counter construction | `experiments/src/gen_counter_store.c` |
| Counter features and loading | `experiments/src/mrac_data.py` |
| ViT and regression head | `experiments/configs/64_64_caida_org/model.py:CustomViT` |
| Transformer implementation | `experiments/local_deps/timm/models/vision_transformer.py` |
| Pretraining | `experiments/scripts/train_pretrain.py` |
| Sampling, adaptation, inference, calibration | `experiments/scripts/simulate_realtime_online.py` |
| Last / Window-5 / Avg-5 update strategies | `experiments/scripts/simulate_sliding_fullstack_online.py` |
| Hot Filter and cold-array deployment interface | `dataplane/include/neufsd_dataplane.hpp` |

The file-based accuracy runners construct full-stream counters. Key-only
IMC/MAWI preparation supplies exact-count heavy inputs; CAIDA preparation uses
the supplied heavy processor. The bounded insertion interfaces are tested
separately. Specify which path is used when reporting accuracy or memory.

## Neural architecture

There are **two independent networks**, each with its own complete ViT
backbone. “Head” in script options selects a network, not a shared-backbone
branch.

| Component | 64 × 64 configuration |
|---|---|
| Counter array | 4,096 uint32 entries, converted to float32 |
| Input | Ascending-sorted, original, descending-sorted counters as three 64 × 64 channels |
| Input scaling | `COUNTER_INPUT_MODE=raw`: no image or min–max normalization |
| Patch projection | Conv2d(3, 768, kernel_size=16, stride=16, bias=True) |
| Sequence | 16 patch tokens + learned CLS token; learned [1, 17, 768] positional embedding |
| Transformer | 12 pre-LayerNorm residual blocks; width 768; 12 attention heads |
| Attention | Head width 64; QKV and projection biases; scale 64^(-1/2) |
| MLP | Linear 768 → 3,072 → 768 with GELU and biases |
| Backbone LayerNorm | Epsilon 1e-6, affine |
| Backbone dropout | Zero |
| Pooling | Final LayerNorm, then CLS token |
| Regression head | LayerNorm(768, eps=1e-5) → Linear(768, 1,000) → GELU → Dropout(0.1) → Linear(1,000, output_dim) |
| Output activation | None |
| Network `1_10` | 10 outputs: sizes 1–10 |
| Network `10_1e4` | 1,080 outputs: sizes 11–1,000, followed by 1,001, 1,101, …, 9,901 |
| Parameters | 86,440,962 and 87,512,032; 173,952,994 in total |

At splice point 100, the small-flow prediction uses all ten outputs of the
first network and the first 90 outputs of the second. The heavy-flow branch
supplies sizes above 100. Patch neighborhoods are a fixed tokenization of
counters, not assumptions about spatial image semantics.

The 128 × 128 configuration has 16,384 counters, 64 patch tokens plus CLS,
and 174,026,722 total parameters. Width, depth, patch size, and regression heads
are otherwise unchanged. Both 64 × 64 networks' FP32 parameters occupy
695,811,976 bytes; this excludes activations, gradients, optimizer state, and
host sampling state.

The data loader also exposes `COUNTER_FEATURE_MODE=origin` and
`COUNTER_INPUT_MODE=log1p`/`log1p_total` for separately configured variants.
The documented three-channel model and commands use `three`/`raw`.

## Training settings

The table specifies the runnable configurations in the reproduction guide.
Command-line values and the explicit environment settings in that guide
override script defaults.

| Setting | Primary synthetic-only pretraining | History-based Avg-5 SFT | Preceding-window IMC | Alternative mixed-data pretraining |
|---|---|---|---|---|
| Source distributions | 60 Zipf datasets | Five preceding sampled FSDs | Preceding window's sampled FSD | 60 Zipf datasets + first 8 windows of each of four traces |
| Candidates per network | 24,000 | 2,000/update | 2,000/window | 36,800 |
| Training / validation | 21,600 / 2,400 | 1,800 / 200 | 1,800 / 200 | 29,600 / 7,200 |
| Split | Last 3 datasets of each of two synthetic roots | Random 10% example holdout | Random 10% example holdout | Last 3 datasets of each of six roots |
| Epochs | 20 | 5/update | 20/window | 20 |
| Training batch size | 64 | 256 | 256 | 64 |
| Learning rate | 0.01, constant | 0.01, constant | 0.01, constant | 0.01, constant |
| Parameters updated | All | All | All | All |
| Flow sampling probability | None | 0.1 | 0.5 | None |
| Counter seeds | 0–399 per dataset | 0–399 × 5 units | 0–1,999 × 1 unit | 0–399 per dataset |
| Initial weights | Random | Pretrained pair | Pretrained pair reloaded per block | Random |
| Weight update policy | Single pretraining run | Carry weights between updates | Independent adaptation per block | Single pretraining run |
| Saved/used weights | Lowest validation WMRD | Last SFT epoch | Last SFT epoch | Lowest validation WMRD |

All four use mean SmoothL1 loss with beta=1, on flow counts, and Adam with
betas=(0.9, 0.999), epsilon=1e-8, no weight decay, and no AMSGrad. There is no
learning-rate scheduler, gradient clipping, or mixed-precision training in
these trainers. Each SFT call creates a fresh Adam optimizer; carrying network
weights does not carry its moments. The training loader shuffles examples;
pretraining validation uses batch 128 and SFT holdout uses batch 512.

The reproduction commands use Python/NumPy/Torch seed 42, sampling seed 42,
and Zipf generation seed 20260617. SFT holdout uses
`torch_seed + 17 * synthetic_unit_id + head_offset`, with head offsets 0 and
100000. Record framework versions and GPU precision settings for each run;
identical seeds do not promise bitwise equality across GPU architectures.

### Prior construction

Each synthetic format uses flow counts {80,000, 100,000}, 1,000,000 packets,
and exponents {0, 0.2, 1.0, 1.1, …, 2.0, 2.5, 3.0}. Every flow receives one
packet, then Zipf weights and largest-remainder rounding allocate the rest.
Two record formats give 60 datasets. Synthetic targets are exact counts at
the output coordinates, including zeros for absent sizes.

The primary prior uses only these two synthetic roots, with 30 datasets and
400 counter seeds per dataset in each root. Holding out the last three
datasets per root gives 54 × 400 = 21,600 training examples and
6 × 400 = 2,400 validation examples per network. Both networks train for
20 epochs with batch size 64; no real traces enter this prior.

The alternative mixed prior adds windows 0–7 from CAIDA2016, CAIDA2018, IMC,
and MAWI. The last three datasets of each of six roots form validation:
real windows 0–4 train the prior and 5–7 validate it. Evaluation beginning at
window 5 therefore includes prior-validation windows; it is not a
trace-disjoint split. Use separate checkpoints for the two configurations.

## Sampling and decoding conventions

Real-trace windows are half-open 60-second intervals anchored at the first
capture timestamp. Their packet counts vary; `minute` and `caida_1min_split`
refer to those window indices. The last partially observed window is retained
unless explicitly dropped. Empty capture intervals are rejected, so process
contiguous captures separately. Key-only input cannot establish time windows.
Synthetic distributions retain their configured 1,000,000 packets; optional
record-count splitting is a separate preprocessing mode, not minute windowing.

The host sampler counts flow keys within a block, then selects entire flows
using BLAKE2b(key + uint32le(block) + uint32le(seed), digest_size=8) / 2^64 < q.
A selected flow retains its full block size and contributes 1/q to its FSD
bin. This host dictionary is not capacity-bounded. Sampled targets interpolate
and extrapolate occupied bins, clip negatives, return zeros for an empty
sample, and a constant for a single occupied size. Fresh synthetic keys and
counter hash seeds produce the training examples.

For matched 2,000-example updates:

- Last: the preceding FSD, five fresh-key replicas × 400 counter seeds.
- Window-5: five preceding FSDs separately, 400 seeds each.
- Avg-5: their average on the union of sizes, five replicas × 400 seeds.

History-based adaptation uses preceding windows. The separate IMC runner
reloads the prior, adapts on window t−1, and decodes window t. It disables
current-window sample-shape mixing. Current-window truth is used only for
scoring; training replicas all come from the preceding sampled FSD.

For each counter seed, the evaluator combines small and heavy predictions,
rounds bins, clips negatives, and scores on its loaded truth support. WMRD is
sum(|prediction − truth|) / sum((prediction + truth)/2). MRD averages the
per-bin ratio on nonzero denominators. Metrics are then averaged across
counter seeds, not computed from their averaged prediction. Pretraining
reports WMRD multiplied by 100; online CSVs use the unscaled ratio.

The heavy-mass loader counts sizes ≥100, whereas final heavy predictions
cover sizes >100 and neural predictions cover sizes ≤100. Preserve this
boundary when reproducing calibration. Report the selected metric column:
`wmrd`, `gate_wmrd`, `twoway_gate_wmrd`, or `sample_shape_wmrd`, with
the matching MRD column. Keep support, flow keys, and state budgets consistent
for baseline comparisons.

Residual calibration uses the current window's actual packet count N_t from
the raw records, with residual mass max(N_t − heavy_mass, 0). Heavy/light mass
fractions use the same N_t; there is no fixed-million-packet substitution.
