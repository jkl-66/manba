# Causal-MambaSA: 实验设计与数据采集终极指南 (ACMMM 2026 强录用标准)

本文档旨在规范化实验流程，确保所有关键数据被正确记录，以绘制出具有统治力的论文图表。

## 1. 核心方法论 (Methodology)

### 1.1 架构核心：Temporal Flattening + Causal Adjustment
*   **输入**：非对齐的多模态序列 (Text $L_T$, Audio $L_A$, Vision $L_V$)。
*   **预处理**：BERT (Text), COVAREP (Audio), Facet (Vision)。
*   **模块一：Temporal Flattening (时序展平)**
    *   **策略**：放弃强制对齐，保留原始时序结构。
    *   **操作**：将不同模态的序列在时间维度上拼接：$X_{flat} = [X_T, X_A, X_V] \in \mathbb{R}^{(L_T+L_A+L_V) \times D}$。
    *   **Modality Embeddings**：引入可学习的模态类型向量 $E_T, E_A, E_V$ 以区分不同模态片段。
*   **模块二：Causal Memory Bank (因果解耦)**
    *   维护全局混杂因子字典 $U \in \mathbb{R}^{K \times D}$。
    *   **两阶段训练 (Two-Stage Training)**：
        *   **Warmup**：仅更新字典，积累环境噪声特征。
        *   **Causal Phase**：启用正交对比损失 $\min \text{Sim}(Z, U)$ 与因果干预。
    *   **推理时干预 (Backdoor Adjustment)**：$Z_{clean} = Z - \mathbb{E}[U]$，通过减去字典的加权平均来消除混杂偏差。
*   **模块三：Bi-Cross-Mamba (长序列融合)**
    *   利用 Mamba 的线性复杂度 $O(N)$ 处理超长拼接序列。
    *   **Masked Mean Pooling**：严格屏蔽 Padding Token，防止 "Padding Poison" 污染全局表征。
    *   双向扫描：$H = \text{Mamba}_{fwd}(X_{flat}) + \text{Mamba}_{bwd}(\text{Flip}(X_{flat}))$。

---

## 2. 实验体系 (Experimental Design)

### 2.1 Baseline 矩阵 (必须跑通的数据)
在 **CMU-MOSEI** 和 **CH-SIMS** 上记录以下模型的 Acc-2 和 F1-Score：

| Model | Type | Key Weakness to Attack |
| :--- | :--- | :--- |
| **MulT (2019)** | Transformer | $O(N^2)$ 慢，易过拟合 |
| **MISA (2020)** | Orthogonal | 静态正交，无因果干预 |
| **Self-MM (2021)** | Self-Supervised | 依赖单模态标签 |
| **UniMSE (2022)** | Unified Space | 噪声鲁棒性差 |
| **MSAmba (2025)** | Mamba | 无因果解耦，OOD 易崩 |
| **Causal-MambaSA** | **Ours** | **全能 (SOTA + Robust)** |

### 2.2 消融实验 (Ablation Studies)
**关键动作**：在代码中设置开关 (`--no_dict`, `--no_ortho`, `--uni_mamba`)，分别运行并记录日志。

| Variant | 预期结果 | 证明点 |
| :--- | :--- | :--- |
| **Full Model** | **Best** | 完整性 |
| **w/o Dictionary** | F1 ↓ 2-3% | 字典记忆了环境噪声 |
| **w/o Backdoor** | F1 ↓ 1-2% | 推理时去偏有效 |
| **w/o Ortho Loss** | F1 ↓ 1.5% | 强制正交是必要的 |
| **Uni-Mamba** | F1 ↓ 0.5% | 双向时序建模的重要性 |

---

## 3. 关键图表绘制与数据采集 (Data Collection for Figures)

为了画出论文中的 4 张神图，必须在训练/推理阶段保存以下数据 (`.npy` 或 `.csv`)。

### 图 1: 模型架构图 (Figure 1)
*   **工具**：Visio / Draw.io
*   **内容**：左侧 SCM 因果图 (节点 $X, Z, U, Y$)，右侧神经网络流程图。
*   **数据需求**：无。

### 图 2: OOD 鲁棒性曲线 (Figure 2 - The Killer Plot)
*   **类型**：折线图 (Line Chart)
*   **X 轴**：噪声强度 (Noise Level: 0, 0.1, 0.3, 0.5) 或 Mask 比例。
*   **Y 轴**：F1-Score
*   **数据采集**：
    运行 `run_ood_benchmark.py`，记录每个噪声等级下的 F1。
    *   `results_ood.csv`:
        ```csv
        Noise_Level, Method, F1_Score
        0.0, MSAmba, 84.5
        0.1, MSAmba, 80.2
        0.3, MSAmba, 72.1
        0.0, Ours, 85.2
        0.1, Ours, 84.8 (Stable!)
        0.3, Ours, 83.5 (Stable!)
        ```

### 图 3: 效率对比图 (Figure 3)
*   **类型**：双轴图 或 散点图
*   **X 轴**：序列长度 (Sequence Length)
*   **Y1 轴**：显存占用 (GPU Memory MB)
*   **Y2 轴**：推理耗时 (Latency ms)
*   **数据采集**：
    在 `train.py` 中使用 `torch.cuda.max_memory_allocated()`。
    *   `efficiency.csv`:
        ```csv
        Seq_Len, Model, Memory_MB, Latency_ms
        50, Transformer, 1200, 15
        500, Transformer, 8000, 120
        1000, Transformer, OOM, -
        50, Ours, 800, 10
        500, Ours, 1200, 25
        1000, Ours, 1500, 45
        ```

### 图 4: t-SNE 特征解耦可视化 (Figure 4)
*   **类型**：2D 散点图
*   **内容**：展示因果特征 $Z$ (蓝色) 与 混杂因子 $U$ (红色) 的分布。
*   **数据采集**：
    在 `validate` 函数中启用 `return_features=True`。
    保存文件：
    *   `z_features.npy`: Shape $(N, D)$
    *   `u_dictionary.npy`: Shape $(K, D)$
    *   `labels.npy`: Shape $(N, 1)$ (可选，用于按情感着色)

---

## 4. 下一步执行清单 (Action Items)

1.  **准备数据**：确保 `data/MOSEI/aligned.pkl` 就位。
2.  **全量训练**：运行 `python -m src.training.train --batch_size 128 --epochs 20`。
3.  **生成 OOD 数据**：运行 `python run_ood_benchmark.py --checkpoint ...`。
4.  **导出可视化数据**：确保 `vis_data/` 目录下生成了 `.npy` 文件。
5.  **绘图**：使用 Matplotlib/Seaborn 读取上述 CSV/NPY 绘制最终图表。

---

## 5. 审稿人潜在挑战与优化方向 (Potential Reviewer Challenges & Optimization)

为了应对顶会（ACMMM/CVPR/AAAI）审稿人的严苛审查，以下三个潜在的 Challenge 必须在后续实验中予以解决：

### 5.1 消融实验的广度 (Ablation Breadth: Dictionary Size $K$)
*   **挑战点**：审稿人会质疑 `CausalMemoryBank` 中字典大小 $K=64$ 的合理性。
*   **对策**：需要补充一组超参数敏感度实验，测试 $K \in \{16, 32, 64, 128, 256\}$ 对 F1-Score 和收敛速度的影响。
    *   **预期逻辑**：$K$ 过小无法覆盖复杂的环境混杂因子；$K$ 过大可能导致过拟合或字典表示冗余。

### 5.2 Mamba vs. Transformer 的归因分析 (Mamba vs. Transformer with Causal Module)
*   **挑战点**：审稿人可能会问：“性能提升是来自 Mamba 的时序建模，还是来自因果模块？”
*   **对策**：增加一个 **Transformer + Causal Module** 的 Baseline 版本。
    *   **预期逻辑**：证明在相同因果约束下，Mamba 在长序列非对齐场景下比 Transformer 具有更好的特征对齐能力和更高的推理效率。

### 5.3 计算效率的定量分析 (Quantitative Efficiency Analysis for $L > 1000$)
*   **挑战点**：Mamba 的优势在于线性复杂度。审稿人需要看到在极长序列（$L > 1000$）下的显存和耗时优势。
*   **对策**：在 `efficiency.csv` 中补充序列长度为 $1000, 2000, 5000$ 的对比数据。
    *   **预期逻辑**：Transformer 在 $L=2000$ 时应出现 OOM（显存溢出）或指数级耗时增加，而 Mamba 应保持平稳的显存增长。

---

## 6. 顶会基线差距诊断 (Gap vs. Top-Conference Baselines)

本节针对 MulT / MISA / UniMSE / MSAmba 进行定性-定量对标，明确差距来源与代码改进闭环。

### 6.1 结构性差距（Baseline 优势点）
*   **MulT / UniMSE**：跨模态对齐与显式交互强，尤其在 Text 主导的样本上更鲁棒。
*   **MISA**：虽然不做因果干预，但其静态正交分解对 label-irrelevant 信号更稳定。
*   **MSAmba**：纯 Mamba 的时序建模更平滑，推理速度快，但缺乏因果去偏与鲁棒验证。

### 6.2 我们的历史短板（已被代码修正）
*   **解耦不彻底**：旧版 $U$ 仅随机投影，缺乏对抗约束导致 $U$ 泄露情感信息。
    *   **修正**：加入对抗锚点与重构约束，提升 $U$ 的不可预测性（见 [causal_module.py](file:///root/autodl-tmp/.autodl/Causal_MambaSA/src/models/causal_module.py)）。
*   **模态稀释**：简单平铺使 Text 强信号被音视频弱信号稀释。
    *   **修正**：新增模态权重门控，动态估计 Text/Audio/Vision 的贡献度（见 [mamba_fusion.py](file:///root/autodl-tmp/.autodl/Causal_MambaSA/src/models/mamba_fusion.py)）。
*   **反事实硬标签缺失**：此前仅特征层面反事实一致，缺少标签层面的约束。
    *   **修正**：加入基于 $U$ 的反事实硬标签惩罚项，形成因果闭环（见 [mamba_fusion.py](file:///root/autodl-tmp/.autodl/Causal_MambaSA/src/models/mamba_fusion.py) 与 [train.py](file:///root/autodl-tmp/.autodl/Causal_MambaSA/src/training/train.py)）。

### 6.3 仍需补齐的顶会必要实验
*   **Transformer + Causal Module** 的直接对标。
*   **$K$ 字典大小的系统消融曲线**。
*   **长序列（$L>1000$）的显存与耗时曲线**。

---

## 7. A800 显存预测分析报告 (Memory Utilization Forecast)

### 7.1 已观测训练显存（历史日志）
*   旧训练配置最大显存峰值约 **7.49 GB**（Batch 128，Hidden 256/512，未使用显存优化），远低于 A800 的 80GB。
*   说明当时尚未进入算力瓶颈，模型容量与 batch size 未充分释放硬件潜力。

### 7.2 理论显存构成
显存主要由三部分构成：
*   **参数显存**：与 hidden_dim、层数、字典大小 $K$ 线性相关。
*   **激活显存**：与 batch size、序列长度 $L$、hidden_dim 线性相关。
*   **优化器状态**：AdamW 约为参数显存的 2 倍（m/v）。

### 7.3 A800 配置建议与预测（理论估计）
*   **建议基线**：`hidden_dim=512`, `batch_size=512`, `L_total≈(Lt+La+Lv)`。
*   **预测显存**：约 **38–52 GB**（可在 A800 安全运行），仍留有 30%–50% 的冗余用于更长序列或更大字典 $K$。
*   **如扩展到 `hidden_dim=768` 或 `batch_size=768`**：显存预计进入 **65–78 GB** 区间，接近满载。

### 7.4 结论
当前配置尚未充分榨干 A800。通过增大 batch size、hidden_dim 或扩展序列长度，可将显存占用推至 60–75GB 区间，以释放硬件潜力并提升模型表征能力。
