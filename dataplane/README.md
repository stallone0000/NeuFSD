# Data-plane Algorithm C++ Bundle

This directory collects the C++ data-structure implementations needed for a
Redis/BESS deployment pass. It intentionally separates two kinds of code:

- `original/`: source files copied from the experiment codebase. Use these when
  exact paper-experiment behavior matters.
- `include/`: small C++17 headers with deployment-oriented interfaces. Use
  these as starting points for Redis modules or BESS packet-processing modules.

## Algorithms

| Algorithm | Exact experiment source | Deployment-oriented entry |
|---|---|---|
| MRAC | `original/Sketchs/MRAC/MRAC.h` | use original class directly |
| Elastic Sketch | `original/Sketchs/elastic/ElasticSketch.h` | use original class directly |
| DaVinci Sketch | `original/Sketchs/DaVinci/DaVinci.h` | use original class directly |
| Array/Hash Sample | `runners/traditional_sample.cpp` | `include/sample_dataplane.hpp` |
| NeuFSD | `original/neufsd/src/gen_counter_store.c`, `original/neufsd/config_64_64_caida_2016/el.h` | `include/neufsd_dataplane.hpp` |

## Notes for Redis/BESS Integration

- The sample baselines use deterministic flow-level sampling: a flow is either
  fully retained or fully discarded. When the bounded table is full, new sampled
  flows are dropped; existing sampled flows continue to be incremented.
- NeuFSD's data plane is only the Hot Filter plus cold counter array. Neural
  decoding and residual-mass calibration are control-plane operations.
- MRAC, Elastic Sketch, and DaVinci retain their original dependencies under
  `original/common` and `original/Sketchs/*`. DaVinci additionally depends on
  `original/Sketchs/DaVinci/util/*`.
- The original experiment runners assume file-based traces. For BESS, replace
  the trace reader with packet-to-flow-key extraction and call each structure's
  `insert` method on every packet.

## Smoke Test

The lightweight headers can be checked with:

```bash
mkdir -p build
g++ -std=c++17 -O2 -I include smoke_test.cpp -o build/dataplane_smoke
./build/dataplane_smoke
```

The original experiment classes compile through their copied runners and
Makefiles in this bundle.
