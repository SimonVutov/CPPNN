# CPPNN

C++ neural-network experiments for MNIST and CIFAR-10, plus a Python CIFAR-10
example powered by [Invariant](https://github.com/SimonVutov/invariant)'s C++ kernels.

| Folder | Implementation |
| --- | --- |
| `1mnist` | C++ dense network; optional OpenCV drawing UI |
| `2CIFAR10` | Original C++ convolutional/dense network |
| `3CIFAR10-invariant` | Python MLP; all dense products use Invariant C++ |
| `4SVHN` | Notes only; no implemented trainer |

## Get the code

Clone the repositories side by side (skip cloning any checkout you already have):

```sh
git clone https://github.com/SimonVutov/invariant.git
git clone https://github.com/SimonVutov/CPPNN.git
cd CPPNN
```

These instructions require the Invariant revision containing Python bindings.
Until these changes are merged, check out `codex/release-0.1-hardening` in
Invariant and `codex/polish-invariant-integration` in CPPNN.

## Data

Run from the repository root. Python 3.10+ can download and checksum the datasets:

```sh
python scripts/download_data.py mnist
python scripts/download_data.py cifar10
```

Downloads use the [MNIST mirror used by torchvision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)
and the [official CIFAR-10 binary archive](https://www.cs.toronto.edu/~kriz/cifar.html).
Please cite the dataset authors when using their data. Existing local datasets
can be supplied with `--data PATH` instead. Downloads are explicit, never part of tests.

## C++ examples

Requires CMake 3.20+ and a C++17 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
./build/mnist --data 1mnist --output runs/mnist --epochs 5 --seed 42
./build/cifar10 --output runs/cifar --epochs 10 --seed 42
```

On Windows/multi-configuration builds, use executables under `build/Release`.
A local five-epoch MNIST run (seed 42) achieved **91.16% test accuracy**.
Both trainers accept `--help`, `--data`, `--output`, `--epochs`, `--seed`, and
`--limit` (limits both training and test samples for smoke runs). The C++ CIFAR
architecture is a slow experimental baseline; its export contains **dense weights
only**, not a reloadable complete CNN. Use the Invariant example for a complete
checkpoint/evaluation workflow. `1mnist/test.cpp` is an archived training experiment.

The optional drawing UI needs OpenCV: configure with `-DCPPNN_BUILD_GUI=ON`, then
run `./build/digit_ui runs/mnist`. It uses the trainer's leaky-ReLU activation.

## CIFAR-10 with Invariant

Keep the two checkouts side by side, then run from CPPNN:

```sh
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install '../invariant[test]'
python -m pytest tests -q
python 3CIFAR10-invariant/train.py --epochs 20 --output runs/invariant
python 3CIFAR10-invariant/train.py --evaluate runs/invariant/best.npz
```

`pip install '../invariant[test]'` compiles the sibling checkout and installs it
into this Python environment. It is a regular installation: reinstall after
changing Invariant. `model.py` imports `invariant.matmul`; no source path is
hardcoded into the trainer. The original C++ trainers do not depend on Invariant.

CI separately checks out a pinned Invariant commit in `.github/workflows/ci.yml`.
That commit is available on GitHub. If it is replaced by a squash/rebase,
update the workflow's `ref` to the published replacement commit.

The model uses 2×2 average pooling, 768 input features, 128 hidden ReLU units, and
momentum SGD. NumPy handles preprocessing/elementwise operations; forward and
backward matrix products run in Invariant. A seeded 45,000/5,000 train/validation
split selects the checkpoint, then evaluates the 10,000 test images once.
Normalization uses training statistics only.

A local default run (seed 42, Apple M5, Python 3.14) achieved **54.14% test
accuracy**, with **54.48% validation accuracy**, in 13.29 seconds for training,
validation, checkpoint writes, and final evaluation (data loading excluded).
This is a simple MLP baseline, not CNN-level performance or a speedup claim.
`best.npz` stores all parameters and preprocessing statistics; `metrics.json`
records settings, epoch timings, throughput, and scores. Results vary by hardware.

For a quick smoke run, add `--train-limit 1000 --test-limit 200 --epochs 2`.
Tests check NumPy forward parity, finite-difference gradients, tiny-data overfitting,
checkpoint round trips, and malformed datasets; they require no downloads.
Generated binaries, models, logs, and datasets are excluded from new commits.
