# 🌾 SAG‑AlignNet: Semantic‑Aligned Grounding for Agricultural Cross‑Modal Retrieval

Official implementation of the paper:  
**“SAG‑AlignNet: Semantic‑Aligned Grounding for Agricultural Cross‑Modal Retrieval under Ubiquitous Perception”**  
*(Submitted to xxx, 2026).*

<p align="center">
  <img src="assets/overview.png" width="800">
</p>

---

## 🔍 Overview
**SAG‑AlignNet** is a unified cross‑modal retrieval framework designed for **agricultural understanding** across multi‑scale visual data — including **satellite**, **drone**, and **ground-level imagery**.  
It introduces two key modules:
- **MasAlign**: a masked alignment mechanism leveraging visual–textual reasoning to reconstruct masked tokens.
- **CProD**: a cross‑modal prototype distillation strategy that learns shared semantic anchors for fine‑grained alignment.

Together, they enable robust and interpretable retrieval of **field semantics**, **disease symptoms**, and **pest features** across heterogeneous modalities.

---

## 🚀 Features
- 🌐 Unified benchmark **AgriSearch / SAG‑Bench** for cross‑modal agricultural retrieval  
- 🧠 Cross‑modal reasoning via **MasAlign** masked reconstruction  
- 📦 Prototype‑based semantic alignment (**CProD**) for local correspondence  
- ⚙️ CLIP‑based dual encoder with lightweight cross‑modal fusion  
- 📈 State‑of‑the‑art performance on FRS‑CMR, CroDis4CMR, CCPRD, and WPIT9K  

---

## 📦 Installation
```bash
git clone https://github.com/AgriSearch/SAG-AlignNet.git
cd SAG-AlignNet
conda create -n sagalign python=3.9
conda activate sagalign
pip install -r requirements.txt
