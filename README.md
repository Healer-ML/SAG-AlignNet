# SAG-AlignNet

Reference implementation of **SAG-AlignNet** for text–image retrieval on the five datasets used in AgriSearch: FRS-CMR, CroDis4CMR, CCPRD, GjPest4CMR, and FruDis4CMR.

This repository contains the method code, environment specification, dataset preparation/split utilities, runnable training and evaluation commands, and model checkpoints. It does **not** distribute AgriSearch images or annotation files. You must obtain the data under its own terms and prepare it locally.

## Contents

- `model/`, `datasets/`, `processor/`, `solver/`, `utils/`: model and experiment code.
- `configs/agrisearch/`: fixed 30-epoch recipes for the five datasets.
- `tools/prepare_agrisearch.py`: normalize existing AgriSearch annotations and link the image folders, or create deterministic splits from an unsplit annotation file.
- `train.py`, `evaluate.py`: training and bidirectional retrieval evaluation entry points.
- `checkpoints/`: model-only inference checkpoints, tracked with Git LFS.
- `requirements.txt`, `environment.yml`: Python/CUDA environment.

## Environment

Tested with Python 3.8, PyTorch 2.1.1, torchvision 0.16.1, and CUDA 12.1. A compatible NVIDIA driver is required for GPU training.

```bash
conda env create -f environment.yml
conda activate sag-alignnet
python -m pip install -r requirements.txt
```

The first model initialization downloads the OpenAI CLIP ViT-B/16 initialization to `~/.cache/clip` and checks its SHA256. For an offline run, download that official CLIP checkpoint in advance and set `pretrain_choice` in the YAML config to its local file path. The supplied SAG-AlignNet checkpoint contains the full trained model, but model construction still initializes the CLIP module before loading it.

## Dataset layout

The preparation script expects the original data root to have these folders (case variants `data_CDDM` and `data_cddm` are accepted):

```text
<source-root>/
├── farm/{img|imgs}/, json/farm.json
├── agr/{img|imgs}/, json/agr.json
├── wp/{img|imgs}/, json/wp.json
└── data_cddm/
    ├── crop/{img|imgs}/, json/crop.json
    └── fruit/{img|imgs}/, json/fruit.json
```

Each JSON record must have `id`, `img_path`, `captions` (a list of strings), and `split`. `dev`, `valid`, and `validation` are normalized to `val`. Image IDs are preserved because they are the retrieval match IDs and must be zero-based/contiguous among training records, as in the provided annotations. Images are symlinked; JSON files are copied and normalized. Nothing is copied into Git except code and model checkpoints.

```bash
python tools/prepare_agrisearch.py \
  --source-root /path/to/retrieval/data \
  --output-root data/AgriSearch \
  --check-images
```

To create reproducible 80/10/10 splits from an annotation file that has no usable `split` values, pass `--generate-splits --seed 1`. The script assigns whole image records (and all their captions) to one split. The default operation preserves the supplied splits.

The expected output folders are `data/AgriSearch/{FRS-CMR,CroDis4CMR,CCPRD,GjPest4CMR,FruDis4CMR}/{img,json}`. The preparation command prints per-dataset image-record, caption, training-ID, and split counts. It exits with a nonzero status for malformed annotations, missing images when `--check-images` is enabled, or non-contiguous training IDs.

The paper's reported totals are 22,602 / 9,460 / 6,628 / 9,496 / 18,524 images in that order. The available annotation files previously inspected for this repository had 6,527 CCPRD records (instead of 6,628) and 18,524 FruDis4CMR records split 14,819/1,852/1,853; the paper reports 14,820/1,852/1,852. Use the output of the preparation script as the authoritative count for the data files you actually have.

## Train

The checked-in configs use 30 epochs, seed 1, deterministic PyTorch settings, and validation split `val` for checkpoint selection. Each dataset has its own recipe and the internal dataset key expected by the loader:

| AgriSearch dataset | Config | Internal key |
| --- | --- | --- |
| FRS-CMR | `configs/agrisearch/FRS-CMR.yaml` | `farm` |
| CroDis4CMR | `configs/agrisearch/CroDis4CMR.yaml` | `crop` |
| CCPRD | `configs/agrisearch/CCPRD.yaml` | `agr` |
| GjPest4CMR | `configs/agrisearch/GjPest4CMR.yaml` | `wp` |
| FruDis4CMR | `configs/agrisearch/FruDis4CMR.yaml` | `fruit` |

Single GPU:

```bash
python train.py --config-file configs/agrisearch/CCPRD.yaml
```

Use both GPUs with DistributedDataParallel (batch size is per GPU):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --config-file configs/agrisearch/CCPRD.yaml
```

Change the config path to train one of the other datasets. Runs write `configs.yaml`, `train_log.txt`, TensorBoard events, and `best.pth` under `outputs/<internal-key>/<timestamp>_sag-alignnet/`. The checkpoint selected during a new run is the highest validation bidirectional mR.

Optional shell wrappers are in `scripts/`. For example:

```bash
bash scripts/train_agrisearch.sh CCPRD 2
```

## Evaluate

Evaluate on validation while selecting a config that matches the checkpoint's dataset and model settings:

```bash
python evaluate.py \
  --config-file configs/agrisearch/CCPRD.yaml \
  --checkpoint checkpoints/CCPRD.pth \
  --split val \
  --eval_output outputs/eval/CCPRD_val.json
```

For final reporting, run the held-out test split once:

```bash
python evaluate.py \
  --config-file configs/agrisearch/CCPRD.yaml \
  --checkpoint outputs/agr/<run>/best.pth \
  --split test \
  --eval_output outputs/eval/CCPRD_test.json
```

Expected console output is a table with `t2i` and `i2t` rows (R@1, R@5, R@10, mAP, mINP), followed by JSON in this form. Metric values are computed from the selected checkpoint and the local data; no numbers are hard-coded as expected results.

```json
{
  "method": "SAG-AlignNet",
  "dataset": "agr",
  "split": "val",
  "checkpoint": ".../best.pth",
  "metrics": {
    "t2i": {"R1": 0.0, "R5": 0.0, "R10": 0.0, "mAP": 0.0, "mINP": 0.0},
    "i2t": {"R1": 0.0, "R5": 0.0, "R10": 0.0, "mAP": 0.0, "mINP": 0.0},
    "mR": 0.0
  }
}
```

`mR` is the arithmetic mean of I2T/T2I R@1, R@5, and R@10. The evaluator returns retrieval percentages on a 0–100 scale.

## Checkpoint provenance

The bundled weights are model-only exports of the available 30-epoch seed-1 checkpoints. Their original run configs selected the saved `best.pth` using the **test** split. They are included to make the prior runs directly evaluable and are not a clean held-out test protocol. The provided configs use `val` for all new training, and final test evaluation should only be run after selecting the model on validation. Per-checkpoint provenance and source run locations are recorded in `checkpoints/metadata.json`.

## Reproducibility notes

- Seed defaults to 1; deterministic cuDNN behavior is enabled by default and can be disabled with `--no-deterministic`.
- Training recipes use the same 384×128 image input and ViT-B/16 CLIP initialization as the source runs.
- `--batch_size` is per process/GPU under `torchrun`; lower it if device memory is insufficient.
- The JSON prep script preserves the provided split by default. Do not regenerate splits when reproducing the supplied annotation protocol.
- The historical run configs used to produce bundled weights were executed in a different `yu` environment (PyTorch 2.0.1+cu118); the published environment file describes the cleaned reference environment, not an assertion that those runs used it.
