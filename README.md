<div align="center">

# SAG-AlignNet

### Semantic-Aligned Grounding for Agricultural Image–Text Retrieval

**A reproducible codebase for cross-modal retrieval on AgriSearch.**

![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1%2Bcu121-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Image--Text%20Retrieval-496A81)
![Dataset](https://img.shields.io/badge/Dataset-AgriSearch-658C6A)

<br>

<img src="asset/model.png" alt="SAG-AlignNet model architecture" width="980">

<br>

*Overview of SAG-AlignNet: semantic-aligned grounding, cross-modal prototype distillation, and collaborative retrieval.*

</div>

---

## Overview

SAG-AlignNet is an image–text retrieval implementation for agricultural data. It combines semantic-aligned grounding with cross-modal prototype distillation and collaborative retrieval to learn a shared image–text representation. This repository provides the model implementation, dataset preparation utilities, experiment configurations, training and evaluation entry points, and released inference checkpoints.

> **Data availability.** AgriSearch images and annotations are not included. Prepare the dataset locally and follow the layout and commands below before training or evaluation.

## Contents

- [Benchmark datasets](#benchmark-datasets)
- [Environment](#environment)
- [Dataset preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints](#checkpoints)
- [Reproducibility notes](#reproducibility-notes)

## Benchmark datasets

The benchmark covers five agricultural image–text retrieval datasets. Counts below follow the dataset statistics used by the project.

| Dataset | Classes | Images | Train | Val | Test |
|:--|--:|--:|--:|--:|--:|
| FRS-CMR | 13 | 22,602 | 18,081 | 2,260 | 2,261 |
| CroDis4CMR | 13 | 9,460 | 7,567 | 946 | 947 |
| CCPRD | 102 | 6,628 | 5,302 | 663 | 663 |
| GjPest4CMR | 17 | 9,496 | 7,596 | 949 | 951 |
| FruDis4CMR | 17 | 18,524 | 14,820 | 1,852 | 1,852 |

The repository does not redistribute these datasets. Annotation files available in earlier project materials had slightly different totals for CCPRD and FruDis4CMR; use the annotations that match your local dataset and record their version when reporting results.

## Environment

The reference environment uses Python 3.8, PyTorch 2.1.1, torchvision 0.16.1, and CUDA 12.1. Create it from the checked-in environment file:

```bash
conda env create -f environment.yml
conda activate sag-alignnet
python -m pip install -r requirements.txt
```

The original training environment named `yu` used PyTorch 2.0.1 with CUDA 11.8. The checked-in requirements describe the reproducible environment for this repository; if using the historical environment, install its matching PyTorch and torchvision builds rather than mixing CUDA wheel versions.

CLIP ViT-B/16 initialization weights are downloaded to `~/.cache/clip` when required. For an offline run, set the corresponding `pretrain_choice` in the experiment configuration to a local pretrained-weight path.

## Dataset preparation

The preparation utility expects source folders for `farm`, `agr`, and `wp`, plus `data_cddm/crop` and `data_cddm/fruit`. Each dataset has an image folder (`img` or `imgs`) and a `json/<dataset>.json` annotation file. For example:

```text
<source-root>/
├── farm/{img|imgs}/, json/farm.json
├── agr/{img|imgs}/, json/agr.json
├── wp/{img|imgs}/, json/wp.json
└── data_cddm/
    ├── crop/{img|imgs}/, json/crop.json
    └── fruit/{img|imgs}/, json/fruit.json
```

Each annotation record must include `id`, `img_path`, `captions` (a list of strings), and `split`. The script normalizes `dev`, `valid`, and `validation` to `val`, preserves image IDs, links image folders, and copies normalized JSON files. Its default operation preserves supplied splits. To generate deterministic 80/10/10 splits, use `--generate-splits --seed 1`.

For example:

```bash
python tools/prepare_agrisearch.py \
  --source-root /path/to/retrieval/data \
  --output-root data/AgriSearch \
  --check-images
```

To generate the deterministic train/validation/test split files with seed 1, add:

```bash
  --generate-splits --seed 1
```

The expected output folders are `data/AgriSearch/{FRS-CMR,CroDis4CMR,CCPRD,GjPest4CMR,FruDis4CMR}/{img,json}`. The script prints record, caption, training-ID, and split counts. Run `python tools/prepare_agrisearch.py --help` for all options. It exits with a nonzero status for malformed annotations, missing images when `--check-images` is enabled, or non-contiguous training IDs. Use the printed counts as the authoritative totals for your local annotations.

## Training

The configurations under `configs/agrisearch/` use 30 epochs, seed 1, deterministic PyTorch settings, and validation split `val` for model selection. Dataset-to-config mapping:

| AgriSearch dataset | Config | Internal key |
|:--|:--|:--|
| FRS-CMR | `configs/agrisearch/FRS-CMR.yaml` | `farm` |
| CroDis4CMR | `configs/agrisearch/CroDis4CMR.yaml` | `crop` |
| CCPRD | `configs/agrisearch/CCPRD.yaml` | `agr` |
| GjPest4CMR | `configs/agrisearch/GjPest4CMR.yaml` | `wp` |
| FruDis4CMR | `configs/agrisearch/FruDis4CMR.yaml` | `fruit` |

A single-device run:

```bash
python train.py --config-file configs/agrisearch/CCPRD.yaml
```

For two GPUs (for example GPU 0 and GPU 1):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --config-file configs/agrisearch/CCPRD.yaml
```

Batch size is per process/GPU when using `torchrun`. Training selects the checkpoint with the highest validation bidirectional mR. Run artifacts, including the resolved config, training log, TensorBoard events, and `best.pth`, are written under `outputs/<internal-key>/<timestamp>_sag-alignnet/`. Optional shell wrappers are available in `scripts/`.

## Evaluation

Evaluate a released checkpoint on the validation split and save the complete metrics as JSON:

```bash
python evaluate.py \
  --config-file configs/agrisearch/CCPRD.yaml \
  --checkpoint checkpoints/CCPRD.pth \
  --split val \
  --eval_output outputs/eval/CCPRD_val.json
```

To evaluate a checkpoint selected during a new training run on the held-out test split:

```bash
python evaluate.py \
  --config-file configs/agrisearch/CCPRD.yaml \
  --checkpoint outputs/agr/<timestamp>_sag-alignnet/best.pth \
  --split test \
  --eval_output outputs/eval/CCPRD_test.json
```

The evaluator reports text-to-image (T2I) and image-to-text (I2T) retrieval metrics, including R@1, R@5, R@10, mAP, and mINP. The bidirectional mean Recall is computed as

\[
\mathrm{mR}=\frac{1}{6}\sum_{d\in\{\mathrm{I2T},\mathrm{T2I}\}}
\left(R@1_d+R@5_d+R@10_d\right),
\]

and is reported on a percentage scale. The evaluator prints T2I and I2T rows and writes JSON results. Check `python evaluate.py --help` for additional evaluation options.

## Checkpoints

Inference checkpoints are stored under `checkpoints/` and tracked with Git LFS. Their dataset and provenance metadata are recorded in `checkpoints/metadata.json`. These are model-only artifacts; optimizer and scheduler states are not included.

**Evaluation caveat.** The historical released weights were selected as `best.pth` using the test split. Their test-set metrics are therefore not an unbiased held-out estimate. For a clean experimental result, train with validation-based model selection and evaluate the selected checkpoint on the untouched test split, as in the commands above.

## Reproducibility notes

- Use the checked-in YAML configuration and record any local path or hyperparameter changes.
- The default seed is 1; the split-generation utility also accepts an explicit seed.
- Keep training/validation/test partitions fixed when comparing methods.
- Record the checkpoint, dataset annotation version, software environment, and evaluation JSON together with reported metrics.
- Do not compare historical test-selected checkpoint scores as if they were produced by validation-only model selection.

---

<div align="center">

**SAG-AlignNet · AgriSearch image–text retrieval**

</div>
