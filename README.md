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

## Data Annotation

This section describes the recommended annotation workflow for collecting and
exporting signature bounding boxes. The current project assumes annotations are
exported from MakeSense as CSV files, then converted into cropped signature
images and manifests by local scripts.

### 标注平台

https://www.makesense.ai/

### 标注流程

1. 打开 MakeSense，导入待标注图片，例如 `annotation/scan_sign_data/`。
2. 任务类型选择 `Object Detection`。
3. 添加标签。标签建议来自 `annotation/scan_sign_label`，也可以在标注过程中逐步补充。
4. 使用默认的 `Rect` 矩形框工具框选每个签名。
5. 每画完一个框，在右侧标注列表中选择对应标签。
6. 所有图片完成后，点击左上角 `Actions` -> `Export Annotations`。
7. 导出格式选择 `Single CSV file`。

### 标注格式

真实签名和仿签签名建议分开标注、分开导出：

- `true.csv`: 真实签名标注。
- `fake.csv`: 仿签或伪造签名标注。

`label_name` 表示签名对应的姓名或身份标签，必须在整个数据集中保持唯一。对于
`fake.csv`，`label_name` 仍然表示“被模仿的目标姓名”。如果能够记录实际仿写人，
建议额外提供 `sign_label_name` 字段，用于表示仿写人或志愿者标签。

如果没有修改原始图片文件名，`raw_data/` 不需要随结果重复返回；也不建议改动原始
图片文件名。若多人分批标注，建议每位标注人员提交自己负责批次的 CSV，CSV 文件名
应与对应图片批次保持一致，便于后续合并和排查。

```text
annotation_result
├── raw_data/
├── true.csv
└── fake.csv
```

建议提供的标注字段：

```text
# MakeSense 默认导出的字段
label_name
bbox_x
bbox_y
bbox_width
bbox_height
image_name
image_width
image_height

# 仿签数据建议额外提供的字段
sign_label_name  # 仿写人或志愿者标签
```

### 仿写流程（供参考）

1. 如果原始文件中已经包含假签名，按同样的矩形框流程标注，并导出到 `fake.csv`。
2. 完成真实签名标注后，可为每个姓名裁切 1-2 个样例，作为志愿者仿写参考。
3. 准备仿签材料，例如常规文件签字处、表格或统一签名纸。
4. 志愿者完成仿签后，对结果拍照保存。建议保存为 JPG；不要自行做扫描增强或格式转换。
5. 对仿签图片使用同样的标注流程，最终导出 `fake.csv`。

注意事项：

1. 建议同一位志愿者对同一个姓名的仿签写在同一页纸上，便于后续标注和追踪。
2. 每个姓名建议由多位志愿者仿写，每人写 5-10 个样本，保证训练数据量。
3. 如果记录仿写人身份，请使用稳定匿名 ID，例如 `volunteer_001`，不要直接使用真实姓名。

推荐的仿签数据交付结构：

```text
fake_sign_result
├── data/
│   ├── 志愿者_01/
│   │   ├── 志愿者_01_001.jpg
│   │   ├── 志愿者_01_002.jpg
│   │   └── ...
│   ├── 志愿者_02/
│   │   ├── 志愿者_02_001.jpg
│   │   ├── 志愿者_02_002.jpg
│   │   └── ...
│   └── ...
└── fake.csv
```

如果数据被拆分给多人标注，可以改为每个批次一份 CSV，例如：

```text
fake_sign_result
├── data/
│   ├── batch_001/
│   └── batch_002/
├── batch_001.csv
└── batch_002.csv
```

## Preprocess Local Images

Convert supported images in a folder to JPEG files under
`datasets/processed/{folder_name}_jpg/`:

```bash
uv run python -m src.common.preprocess_images datasets/签名2
```

The converter supports PNG, JPG/JPEG, HEIC, and HEIF. Generated processed images
are ignored by git.

## Prepare annotation_v2

Build the `csw_0116` processed dataset and manifests from
`datasets/raw/annotation_v2`:

```bash
uv run python -m src.common.prepare_annotation_v2 \
  --source-root datasets/raw/annotation_v2 \
  --dataset-id csw_0116 \
  --output-root datasets/processed \
  --manifest-dir datasets/manifests \
  --overwrite
```

This creates `datasets/processed/csw_0116/{true,fake}/` plus
`datasets/manifests/csw_0116_true.csv` and
`datasets/manifests/csw_0116_fake.csv`. The manifest columns are
`image_id,image_path,label,authenticity`, where `label` is the
`original_label` from `patch_mapping_summary.csv`.

## Data Loading

The project uses standard manifests as the shared data boundary:

```csv
image_id,image_path,label,authenticity
csw0116_t000001,datasets/processed/csw_0116/true/0_00fac820.jpg,0,true
```

Generic manifest reading and single-image loading live in `src/common/`.
Method-specific sampling lives under each method directory. For example,
MS-SigNet builds co-tuplets from the standard manifest by sampling one true
anchor, true positives with the same `label`, and fake negatives with the same
`label`.

Train MS-SigNet on `csw_0116` manifests:

```bash
uv run python -m src.ms_signet.train --config configs/ms_signet_csw_0116.yaml
```

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

Each training run writes a self-contained experiment directory:

```text
outputs/runs/ms_signet_hansig/YYYY-MM-DD_HHMMSS/
├── config.yaml
├── run.json
├── checkpoints/
│   ├── epoch_001.pth
│   ├── latest.pth
│   └── best.pth
├── metrics/
│   ├── latest_train.json
│   └── train_history.json
├── predictions/
├── logs/
│   └── train.log
└── embeddings/
```

Use `outputs/runs/` for regular experiments. Keep `models/` only for manually
selected release/export weights. Both directories are ignored by git.

The v1 pipeline assumes cropped offline signatures. Detection, restoration, and
document-level workflows can be added later as separate methods under `src/`.
