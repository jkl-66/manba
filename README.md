# Causal-MambaSA: Robust Unaligned Multimodal Sentiment Analysis via Structural Causal Disentanglement and State-Space Models

Official implementation of **Causal-MambaSA**, a framework designed for robust, fine-grained multimodal sentiment analysis (MSA) in unaligned and long-context scenarios.

## 🚀 Core Motivation & Solutions

### 🧠 Pain Point 1: Unaligned & Long-Context Dilemma
In real-world multimodal dialogues, video and audio signals are often **unaligned** and extremely long. Traditional models either perform aggressive truncation/alignment (losing information) or use Transformers, which suffer from $O(N^2)$ complexity and memory explosion.
- **Solution 1:** We utilize **Temporal Flattening** combined with **Bi-Mamba-2 (SSD)**, achieving $O(N)$ linear complexity. This ensures full-sequence, fine-grained information retention even for thousands of steps.

### 🛡️ Pain Point 2: Speaker & Environment Bias
Models often take "shortcuts" by fitting to a speaker's inherent voice (Speaker Bias) or specific background conditions (Environment Bias), leading to catastrophic failure in noisy or Out-of-Distribution (OOD) scenarios.
- **Solution 2:** We introduce **Structural Causal Models (SCM)** with a **Hierarchical Causal Token Intervention (HCTI)**. By utilizing a two-stage intervention dictionary (Global & Local), we explicitly cut spurious correlations and extract pure sentiment-content features $Z$.

## 🔬 Key Innovations

### 1. Hierarchical Causal Token Intervention (HCTI)
HCTI introduces a **Hierarchical Memory Bank** to separate dataset-wide global confounders from batch-specific local confounders. Backdoor adjustment is applied at the **token-level**, enhancing robustness against OOD noise.

### 2. 4-Way Cross-Scan Deep Mamba Fusion (CS-Mamba)
A **Cross-Scan** mechanism parallelizes four scanning paths: forward/backward time and forward/backward modality interaction, capturing deep cross-modal synergies with $O(L)$ complexity.

### 3. Weakly-supervised Fine-grained Probe
To bridge the gap between global polarity labels and fine-grained understanding, we include a **Fine-grained Probe** head. While training only on global labels, the model spontaneously learns to identify "who is expressing what to whom" (Holder, Target, Aspect, etc.) by minimizing causal loss, as demonstrated in our qualitative attention heatmaps.

## 📊 OOD Robustness Benchmark
Our model is rigorously tested against noise injection:
- **Audio:** 10%/20%/30% Gaussian noise (simulating background interference).
- **Vision:** 10%/20%/30% Zero-masking (simulating occlusion/blackout).
Check the `results/ood_benchmark.csv` for performance stability comparisons against baselines like MISA and Self-MM.

## 🚄 Training & Optimization (Target: F1 0.86+)
The training pipeline is optimized for SOTA performance:
- **Dynamic Threshold Search:** Every epoch performs a validation-set threshold search to report the true F1 potential.
- **Differentiated Learning Rates:** Encoders, Fusion blocks, and Heads use tailored LRs for stable convergence.
- **Multi-Task Balance:** 0.8 Regression + 0.2 Auxiliary tasks (Sextuplet Probe + SupCon).

To run the optimized training:
```bash
python src/training/train.py --hidden_dim 1024 --batch_size 128 --lr 2e-4
```

## 🛡️ Robustness Switch (Optional)
Once F1 targets are met, enable OOD testing via:
```bash
python src/training/train.py --eval_only --load_ckpt checkpoints/best_model.pth --noise_level 0.2
```
