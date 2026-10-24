<div align="center">

# AgriSearch: SAG-AlignNet

**A Multi-Domain Benchmark and Alignment Framework for Agricultural Image-Text Retrieval**

[![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1%2Bcu121-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-Image--Text%20Retrieval-496A81)](#sag-alignnet)

<img src="asset/model.png" alt="SAG-AlignNet architecture from the paper" width="880">

**SAG-AlignNet architecture.** [Open the vector PDF](asset/model.pdf).

</div>

---

AgriSearch introduces **SAG-Bench**, a five-dataset agricultural image-text retrieval benchmark, and **SAG-AlignNet**, a model combining Cross-modal Prototype Distillation (CProD), Masked Alignment (MasAlign), JS-divergence alignment, and Auxiliary Modality Preservation (AMP). The diagram above and implementation details in this repository correspond to the current manuscript.

## SAG-Bench datasets

The counts and splits below follow Table `Dataset statistics, image sources, and data splits used in AgriSearch` in the manuscript.

| Dataset | Source | Classes | Images | Texts / image | Train | Validation | Test | Config |
|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| FRS-CMR | FIT | 13 | 22,602 | 3 | 18,081 | 2,260 | 2,261 | [`configs/FRS-CMR.yaml`](configs/FRS-CMR.yaml) |
| CroDis4CMR | CDDM | 13 | 9,460 | 3 | 7,567 | 946 | 947 | [`configs/CroDis4CMR.yaml`](configs/CroDis4CMR.yaml) |
| CCPRD | IP102 | 102 | 6,628 | 5 | 5,302 | 663 | 663 | [`configs/CCPRD.yaml`](configs/CCPRD.yaml) |
| GjPest4CMR | WPIT9K | 17 | 9,496 | 3 | 7,596 | 949 | 951 | [`configs/GjPest4CMR.yaml`](configs/GjPest4CMR.yaml) |
| FruDis4CMR | CDDM | 16 | 18,524 | 3 | 14,819 | 1,852 | 1,853 | [`configs/FruDis4CMR.yaml`](configs/FruDis4CMR.yaml) |

Dataset images and annotations are not included in this repository. **The dataset release is planned after paper acceptance.** Until then, obtain the source data through its original access route and follow its terms. The preparation utility preserves supplied splits; its optional `--generate-splits` mode creates new splits and does not reproduce the manuscript's reported results.

Expected local layout:

```text
data/AgriSearch/<dataset>/{img,json/<dataset>.json}
```

Prepare annotations and check image paths:

```bash
python tools/prepare_agrisearch.py \
  --source-root /path/to/retrieval/data \
  --output-root data/AgriSearch \
  --check-images
```

## SAG-AlignNet

- **Encoder:** CLIP ViT-B/16 with 512-dimensional embeddings.
- **Cross-modal Transformer:** 4 layers and 8 attention heads.
- **CProD:** `K` shared learnable prototypes, per-prototype 512-to-512 projections, and a `K`-way prototype prediction head. Prototype attention reads the complete image sequence (CLS and patches) and the complete text sequence through EOS (SOS, tokens, and EOS), excluding only batch padding. It stores detached prototype features from the preceding epoch, retrieves top-`m` neighbors for each prototype, and uses the final prototype branch's neighborhood to form the neighbor-guided soft target, as specified in the paper. Epoch 1 uses hard-label supervision only; subsequent epochs mix hard and soft label losses with `lambda`.
- **MasAlign:** masks eligible text tokens at probability `rho`, then uses visual context for token reconstruction and binary masked-position prediction. EOS-trailing padding is excluded from Transformer attention and position-prediction loss. Its error-augmented branch samples from the top `Ke=32` predictions and avoids replacing a masked token with its original token.
- **JS alignment:** symmetric image-to-text and text-to-image JS divergence, with `tau=0.02` and fixed `gamma=0.04`.
- **AMP:** a lightweight modality classifier trained on the unnormalized global CLS/EOS features with the visual/text modality labels.

The global ITC, JS, AMP, and matching-label branches use the original, unmasked caption. MasAlign receives its separately masked caption. The overall objective sums ITC, JS, CProD, MasAlign, and AMP with equal outer weights; CProD and MasAlign retain their internal loss definitions.

The model uses 384 × 128 images, a maximum text length of 77, and a 49,408-token prediction vocabulary. Image augmentation uses random horizontal flipping, cropping, and erasing.

### Training recipe

All five dataset configs use the manuscript's 30-epoch recipe: Adam, batch size 64, base learning rate `1e-5`, a `5×` learning-rate multiplier for randomly initialized modules, and maximum gradient norm 5.0. The shared fixed settings are `theta=0.001`, `Ke=32`, `tau=0.02`, `gamma=0.04`; each dataset config specifies `alpha`, `rho`, `K`, `n`, `m`, and `lambda`. The base config sets seed 2 for the single-run results described in the manuscript.

| Dataset | `alpha` | `rho` | `K` | `n` | `m` | `lambda` |
|:--|--:|--:|--:|--:|--:|--:|
| FRS-CMR | 0.55 | 0.40 | 2 | 0.50 | 100 | 0.50 |
| CroDis4CMR | 0.10 | 0.25 | 6 | 0.50 | 100 | 0.50 |
| CCPRD | 0.70 | 0.20 | 6 | 0.75 | 25 | 0.50 |
| GjPest4CMR | 0.55 | 0.30 | 6 | 0.75 | 100 | 0.25 |
| FruDis4CMR | 0.55 | 0.55 | 2 | 1.00 | 25 | 0.50 |

### Sensitivity analysis and module comparisons

[`tools/run_sensitivity.py`](tools/run_sensitivity.py) sweeps `alpha`, `rho`, `K`, `n`, `m`, and `lambda` using the symbols and values reported in Figs. 8–9. Definitions and commands are in [`ABLATION_PARAMETERS.md`](ABLATION_PARAMETERS.md). The optional `mechanism_variant` setting supports the module replacements used in the paper's comparisons; the default `native` setting selects the proposed CProD and MasAlign modules.

## Environment

Reference setup: Python 3.8, PyTorch 2.1.1, CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate sag-alignnet
pip install -r requirements.txt
```

The first run downloads the CLIP ViT-B/16 initialization weights. For offline use, set `pretrain_choice` in the config to a local weight file.

## Train

Train on CCPRD with one GPU:

```bash
python train.py --config-file configs/CCPRD.yaml
```

Use GPU 0 and GPU 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --config-file configs/CCPRD.yaml
```

Replace `CCPRD.yaml` with another dataset config to select a different benchmark dataset. The best validation checkpoint and logs are written under `outputs/`.

## Evaluate

```bash
python evaluate.py \
  --config-file configs/CCPRD.yaml \
  --checkpoint checkpoints/CCPRD.pth \
  --split val \
  --eval_output outputs/CCPRD_val.json
```

The output reports I2T and T2I R@1, R@5, and R@10, together with bidirectional mean recall (mR).

## Citation

```bibtex
@article{yu2026agrisearch,
  author  = {Yu, Kejian and Ma, Wentao and Zhou, Tongqing and Wang, Mengyu and Tu, Wenxuan and Wang, Siwei and Cai, Zhiping and Liu, Xinwang},
  title   = {AgriSearch: A Multi-Domain Benchmark and Alignment Framework for Agricultural Image-Text Retrieval},
  note    = {Under review}
}
```
