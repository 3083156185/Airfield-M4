# Airfield-M4: Multimodal D-FINE and PAPF

Official code release for **Airfield-M4: A Multimodal Object Detection
Benchmark for Airport Surface Traffic Surveillance**.

This repository uses D-FINE as the primary detector implementation and provides
a unified RGB-auxiliary input pipeline for thermal infrared and depth. It
includes four controlled RGB+IR fusion strategies: input-level early fusion,
plain middle fusion, Progressive Auxiliary-to-Primary Fusion (PAPF), and
late-neck fusion. Category-level text guidance is also supported through a
precomputed text-embedding cache.

The Airfield-M4 dataset, training outputs, and model checkpoints are not stored
in this repository. No training weights are included in the Git history.

## Highlights

- Paired RGB and dense auxiliary-modality dataloading with synchronized geometry
  transforms.
- D-FINE-S configurations for RGB, IR, depth, RGB+IR, and RGB+IR+text.
- Reproducible Early, Plain-middle, PAPF, and Late-neck fusion baselines.
- Airfield-M4 and M3FD experiment configurations with portable relative paths.
- A detector-agnostic guide for extending CNN, DETR, and Mamba architectures
  from single-sensor to multimodal input.

## Repository layout

```text
Airfield-M4/
├── configs/
│   ├── airfield_m4/       # Airfield-M4 training and evaluation configs
│   ├── m3fd/              # M3FD external-validation configs
│   └── multimodal/yaml/   # D-FINE multimodal architecture graphs
├── engine/                # Model, data, loss, and solver implementation
├── tools/
│   ├── inference/         # RGB and multimodal inference
│   ├── ov/                # Text-embedding cache construction
│   └── visualization/     # Qualitative and feature-response utilities
├── train.py
├── requirements.txt
└── docs/ADAPTING_OTHER_DETECTORS.md
```

## Installation

The release was validated on Linux with Python 3.10, PyTorch 2.2.2, and CUDA
12.1. Other recent PyTorch releases may also work. Install a PyTorch build that
matches the local CUDA driver before installing the remaining packages.

```bash
git clone https://github.com/3083156185/Airfield-M4.git
cd Airfield-M4

conda create -n airfield-m4 python=3.10 -y
conda activate airfield-m4

# Select the correct PyTorch/CUDA command from https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install -r requirements.txt
```

Some optional modules import MMCV operators. Install MMCV with a wheel matching
the installed PyTorch and CUDA versions:

```bash
pip install -U openmim
mim install "mmcv==2.2.0"
```

## Data preparation

Annotations use COCO detection format. Store each auxiliary image as a
single-channel `.npy` or `.npz` array. The auxiliary filename stem must equal the
corresponding RGB filename stem.

```text
data/Airfield-M4/
├── rgb/
│   ├── train/images/000001.jpg
│   └── val/images/000101.jpg
├── ir/
│   ├── train/npy/000001.npy
│   └── val/npy/000101.npy
├── depth/
│   ├── train/npy/000001.npy
│   └── val/npy/000101.npy
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── text/
    └── category_embeddings.pth  # generated locally; ignored by Git
```

M3FD uses the following layout:

```text
data/M3FD/
├── vi/
├── ir_npy/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

Update the paths and `num_classes` in a configuration if a different layout or
category set is used.

## D-FINE experiment configurations

| Input or fusion policy | Airfield-M4 config | M3FD config |
|---|---|---|
| RGB only | `configs/airfield_m4/rgb.yml` | `configs/m3fd/rgb.yml` |
| IR only | `configs/airfield_m4/ir.yml` | `configs/m3fd/ir.yml` |
| Depth only | `configs/airfield_m4/depth.yml` | - |
| RGB+IR Early | `configs/airfield_m4/rgb_ir_early.yml` | `configs/m3fd/rgb_ir_early.yml` |
| RGB+IR Plain-middle | `configs/airfield_m4/rgb_ir_plain_middle.yml` | `configs/m3fd/rgb_ir_plain_middle.yml` |
| RGB+IR PAPF | `configs/airfield_m4/rgb_ir_papf.yml` | `configs/m3fd/rgb_ir_papf.yml` |
| RGB+IR Late-neck | `configs/airfield_m4/rgb_ir_late_neck.yml` | `configs/m3fd/rgb_ir_late_neck.yml` |
| RGB+IR+Text | `configs/airfield_m4/rgb_ir_text.yml` | - |

## Training

Train D-FINE-S with PAPF on one GPU:

```bash
python train.py \
  -c configs/airfield_m4/rgb_ir_papf.yml \
  --device 0 \
  --seed 0
```

For distributed training:

```bash
torchrun --nproc_per_node=4 train.py \
  -c configs/airfield_m4/rgb_ir_papf.yml \
  --seed 0
```

Replace the configuration path with another row from the table to reproduce a
single-modality or fusion-stage baseline. Keep the same seed, resolution,
optimizer, augmentation schedule, and number of epochs for controlled
comparisons.

## Evaluation

```bash
python train.py \
  -c configs/airfield_m4/rgb_ir_papf.yml \
  -r /path/to/checkpoint.pth \
  --test-only \
  --device 0
```

The evaluator reports COCO-style AP, AP50, AP75, APs, APm, and APl.

## Multimodal inference

```bash
python tools/inference/multimodal/torch_inf.py \
  -c configs/airfield_m4/rgb_ir_papf.yml \
  -r /path/to/checkpoint.pth \
  --input-rgb /path/to/rgb/image_or_directory \
  --input-npy /path/to/ir/npy_file_or_directory \
  --output inference_results/papf \
  --device 0
```

## Text-guided configuration

Generate category embeddings locally from a COCO annotation file and a TIPS
text encoder:

```bash
python tools/ov/build_tips_text_cache.py \
  --ann data/Airfield-M4/annotations/instances_train.json \
  --out data/Airfield-M4/text/category_embeddings.pth \
  --model-path /path/to/text_encoder.ts \
  --tokenizer-path /path/to/tokenizer.model
```

Then train with `configs/airfield_m4/rgb_ir_text.yml`. Text encoders and cached
embeddings are external artifacts and are intentionally excluded from Git.

## Adapting other detector families

See [Adapting Single-Sensor Detectors to Multimodal Input](docs/ADAPTING_OTHER_DETECTORS.md)
for the common data contract, architecture-specific insertion points, and a
fair-comparison checklist for CNN-, DETR-, and Mamba-based detectors.

## Checkpoints and dataset

Model checkpoints and dataset files will be distributed separately from the
source repository. The `.gitignore` blocks common checkpoint formats, output
directories, and local data folders to prevent accidental publication.

## Acknowledgements

This implementation is built on
[DEIM](https://github.com/ShihuaHuang95/DEIM) and its D-FINE implementation.
We thank the upstream authors and retain their Apache-2.0 license and
attribution. The original DEIM README is preserved in
[`docs/UPSTREAM_DEIM_README.md`](docs/UPSTREAM_DEIM_README.md).

## License

Released under the [Apache License 2.0](LICENSE). Third-party components remain
subject to their respective licenses.
