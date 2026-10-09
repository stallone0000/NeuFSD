# NeuFSD: Rethinking Flow Size Distribution via Neural Decoding

Code for NeuFSD: counter construction, neural FSD decoding, pretraining,
online adaptation, residual calibration, and evaluation.

## Start here

- [Model and training configuration](docs/MODEL.md): the exact ViT, output
  coordinates, training settings, sampling, and decoding conventions.
- [Reproduction guide](docs/REPRODUCING.md): prepare your own traces, generate
  training pairs, train both decoders, run adaptation and comparisons.
- [C++ data plane](dataplane/README.md) and [Tofino implementation](p4/README.md).

## Layout

```text
experiments/
  configs/       Per-resolution and record-format models and preprocessing
  local_deps/    Bundled ViT implementation used by the training scripts
  scripts/       Data preparation, pretraining, adaptation, evaluation, plotting
  src/           Counter generator and shared data-loading utilities
  compare/       Sketch and sampling baselines
dataplane/       C++ insertion structures and deployment interfaces
p4/             Tofino programs, setup, and traffic-generation utilities
docs/           Model specification and reproduction instructions
tests/          Small, generated-input correctness checks
```

This is a source-only repository. It does not distribute traffic datasets,
trained weights, experiment-result tables, logs, or generated figures.
Obtain CAIDA, IMC, and MAWI traces under their providers' access conditions;
the guide specifies the binary record layouts and preprocessing. Outputs are
generated locally from the supplied code.

## Environment

Use Python 3.10 or later, GCC/G++ with OpenMP, and a CUDA-compatible
PyTorch/torchvision pair for training. Install PyTorch for your machine, then:

```bash
python3 -m pip install -r experiments/requirements.txt
python3 -m unittest discover -s tests -v
```

The model imports the bundled `experiments/local_deps/timm`; do not replace it
with an unrelated installed ViT when comparing architectures or checkpoints.
Training commands in the guide select one GPU and fresh output directories.
No GPU is needed for the unit tests.

The file-based accuracy pipeline and the deployable data-plane components have
separate entry points. Switch measurements additionally require a configured
Tofino/SDE environment; Redis/BESS integration requires the corresponding host
systems. See their component documentation for these prerequisites.
