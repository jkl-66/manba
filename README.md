# CausalMambaSA: Hierarchical Causal Token Intervention Mamba for Multimodal Sentiment Analysis

Official implementation of the **HCTI-Mamba** framework, designed for high-performance and robust multimodal sentiment analysis (MSA).

## 🚀 Key Innovations

### 1. Hierarchical Causal Token Intervention (HCTI)
Unlike traditional global de-confounding, HCTI introduces a **Hierarchical Memory Bank** to separate dataset-wide global confounders (e.g., speaker style) from batch-specific local confounders. By applying backdoor adjustment at the **token-level**, the model extracts pure sentiment-content features $Z$, significantly enhancing robustness against Out-of-Distribution (OOD) noise.

### 2. 4-Way Cross-Scan Deep Mamba Fusion (CS-Mamba)
We propose a **Cross-Scan** mechanism that parallelizes four scanning paths: forward/backward time and forward/backward modality interaction. This architecture achieves $O(L)$ linear complexity while capturing deep cross-modal synergies equivalent to multi-head cross-attention.

## 🛠️ Architecture

The model is designed for high-capacity hardware (e.g., A100/A800 80G):
- **Intra-modal Encoder:** 6-layer stacked Mamba blocks for deep modality-specific abstraction.
- **Inter-modal Fusion:** 4-layer stacked Cross-Scan Mamba blocks.
- **Learning Objectives:** Causal Orthogonality + Supervised Contrastive Learning (SupCon) + Multi-task Self-Supervision (MTL).

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚄 Training & Ablation

To train the full SOTA model on A800:
```bash
python src/training/train.py --hidden_dim 1024 --batch_size 128 --lr 2e-4
```

### Ablation Study Commands
- **No Causal:** `python src/training/train.py --ablation no_causal`
- **No Cross-Scan:** `python src/training/train.py --ablation no_cross_scan`
- **No MTL:** `python src/training/train.py --ablation no_mtl`

## 📊 Dataset
This repository is optimized for the **CMU-MOSEI** dataset (unaligned version).
Note: Data files should be placed in the `data/` directory (excluded from git).
