# HanSig + MS-SigNet Signature Verification

This repository is a PyTorch/uv scaffold for offline Chinese handwritten
signature verification. The first reproducibility target is MS-SigNet with
Co-Tuplet Loss on HanSig.

## Layout

```text
configs/              Experiment and path configs
datasets/             Dataset docs, manifests, and interfaces
src/common/           Shared training utilities
src/ms_signet/        MS-SigNet model, loss, sampler, train, inference
evaluate/             Metrics, threshold calibration, pair evaluation
scripts/              Data preparation and smoke-test entrypoints
tests/                Unit tests for parsing, model, and metrics
```

Raw images are intentionally ignored by git. Put HanSig under
`datasets/raw/HanSig`, keep private local photos where they are, and commit only
code, configs, docs, tests, and small manifest examples.

## Local Setup on macOS

```bash
uv sync
uv run pytest
uv run python scripts/smoke_test.py
```

The default PyPI PyTorch build works for CPU and Apple Silicon MPS smoke tests.
Use this machine for code, data parsing, and small correctness checks.

## Server Setup for RTX 4090

On the server, clone this repo, copy or mount `datasets/raw/HanSig`, then install
the CUDA PyTorch build appropriate for the server driver. A typical CUDA 12.1
setup is:

```bash
uv sync --no-install-project
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv sync
```

Adjust the CUDA wheel index if the server uses a different supported CUDA stack.

## Prepare HanSig Manifests

```bash
uv run python scripts/prepare_hansig_manifest.py --root datasets/raw/HanSig
```

This writes `hansig_full.csv`, `hansig_train.csv`, `hansig_val.csv`, and
`hansig_test.csv` into `datasets/manifests/`. These generated CSV files are
ignored by default; commit only small `.example.csv` files when needed.

## Train MS-SigNet

```bash
uv run python -m src.ms_signet.train --config configs/ms_signet_hansig.yaml
```

The v1 pipeline assumes cropped offline signatures. Detection, restoration, and
document-level workflows can be added later as separate methods under `src/`.
