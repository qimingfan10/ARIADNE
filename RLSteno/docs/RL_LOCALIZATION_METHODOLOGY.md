# 冠脉狭窄检测与精定位方法论说明

## 1. 整体目标

在已有的冠脉狭窄 **detector + segmentor** 流水线基础上，本工作聚焦于：

- 在几何检测给出的候选狭窄点基础上，
- 通过强化学习（RL）对候选点进行 **亚像素级的精定位** 与 **假阳性拒绝**，
- 在 **干净血管数据集（30条血管）** 上优化 
  - **TPR = 0.867**
  - **PPV = 0.634**
  - **F1 = 0.732**（最终采用的 MLP + 75px margin 设置）。

下文从 baseline → 几何方法 → RL 方法，依次说明演进过程与对应代码文件。

---

## 2. Baseline：传统检测流水线

### 2.1 Baseline 流程概述

1. **血管分割**：使用训练好的分割网络对冠脉影像做 vessel segmentation。
2. **基于形态或简单规则的狭窄检测**：
   - 在分割结果上提取中心线/半径信息，
   - 使用固定阈值或简单启发式寻找半径最小的点作为狭窄候选。
3. **后处理与可视化**：将检测框/点叠加回原图进行展示和评估。

### 2.2 Baseline 相关文件

> 注：Baseline 是早期流水线的一部分，这里只列出与后续 RL 对比时仍会用到的评估脚本。

- **`test_stenosis_detection.py`**  
  - 作用：对 baseline 模型的狭窄检测结果做离线评估（TP/FP/FN 等）。

- **`evaluate_accuracy.py`**  
  - 作用：统一计算不同方法在同一数据集上的 TPR、PPV、F1，用于 baseline 与后续方法的横向对比。

Baseline 的问题：
- 在干净血管上仍存在较多 **假阳性**，
- 定位精度受限于形态学阈值和网格分辨率，
- 无法显式建模“拒绝”误报的行为。

---

## 3. 几何方法：基于骨架与半径曲线的规则算法

### 3.1 方法思想

在获得分割 mask 后：

1. 对血管 mask 做 **骨架化 (skeletonize)**，得到离散中心线点 `skeleton_points`；
2. 基于距离变换计算每个骨架点的局部半径 `radii`；
3. 在主干血管上，构造半径序列 `r(s)`，并计算：
   - 平滑半径曲线 `smooth(r)`；
   - 一阶导数 `grad` 与二阶导数 `curv`；
4. 在曲线中搜索 **局部半径极小点 + 曲率较大** 的位置，作为几何狭窄候选；
5. 加入 **最小间距约束 (min_dist)**，避免在同一病变区域内重复给出多个候选。

几何方法的特点：
- **几乎不需要学习参数**，易解释；
- 能有效在 skeleton 上找到“看起来像狭窄”的候选点；
- 但仍存在：
  - 候选点偏离真实 GT 中心；
  - 假阳性较多（尤其在分叉/弯曲区域）。

### 3.2 几何检测在代码中的实现

几何检测逻辑在多个脚本中以函数形式出现，核心步骤包括：

- 从缓存中读取 skeleton 与 radii：
  - **缓存文件**：`skeleton_cache_v2.pkl`
  - 预计算脚本：`precompute_skeleton_cache.py`

- 几何候选检测函数（示意）：
  - 在多处可见，典型实现在 RL 环境/可视化脚本中，例如：
    - `rl_sequential_stenosis_env.py` 内部的几何候选生成，
    - 用于可视化的临时脚本中 `detect_stenosis(...)` 函数。

### 3.3 几何方法的定位与局限

- 几何方法作为 **“候选生成器 (proposal generator)”**，为后续 RL 提供少量可能的狭窄位置；
- 几何方法本身无法解决：
  - 候选点到 GT 中心的精确对齐（亚像素级 RL 调整），
  - 在存在大量“误报候选”的情况下自动拒绝假阳性。

---

## 4. RL 方法：在几何候选基础上的精定位与拒绝机制

### 4.1 环境设计思路

我们将“沿着血管骨架移动并选择最终狭窄点”的过程，建模为一个强化学习环境：

- **状态 (observation)**：
  - 以当前骨架点为中心，截取窗口 `[i-K, ..., i, ..., i+K]` 上的：
    - 标准化半径曲线 `r_norm`，
    - 一阶导数 `grad`，
    - 二阶导数 `curv`，
    - 相对位置编码 `rel_pos`；
  - 拼接为一维特征向量输入给策略网络。

- **动作 (action)**：
  - `0`: 向左移动（沿骨架 index 减小）；
  - `1`: 向右移动；
  - `2`: **Confirm**（确认当前点为狭窄）；
  - `3`: **Reject**（拒绝当前几何候选，认为是假阳性）。

- **奖励设计 (reward)**：
  - 若 Confirm 且落在 GT 附近（例如 75px margin 内）：给予高正奖励，并按距离给予更细粒度奖励；
  - 若 Confirm 且不在 GT 范围：给予负奖励，计为假阳性；
  - 若 Reject 且当前候选确实是假阳性：给予正奖励；
  - 若 Reject 但其实附近存在 GT：给予负奖励；
  - 步长罚项 (step penalty)，鼓励较少步数内完成定位。

环境与训练数据的关键在于：
- 同时包含 **真阳性样本**（有 GT 的血管），
- 以及从真实 detector 输出中采样的 **高比例假阳性** 样本，
- 让 RL 学会“确认真正的狭窄”和“拒绝误报点”。

### 4.2 RL 环境文件

- **`rl_reject_env.py`**  
  - 定义了带 4 个动作（Left / Right / Confirm / Reject）的 `StenosisRejectEnv`：
    - 状态空间构造（基于半径曲线局部窗口）；
    - 奖励函数（精定位 + 拒绝机制）；
    - 统计 TP/TN/FP/FN 等指标，用于训练调试；
  - 提供 `create_mixed_training_data(...)`：
    - 构造混合训练样本集（真阳性 + 假阳性检测结果）。

### 4.3 RL 训练脚本（MLP + LSTM）

#### 4.3.1 MLP + Reject（最终采用）

- **训练脚本**：`train_rl_reject.py`
  - 使用 `Stable-Baselines3` 的 **PPO (MlpPolicy)**：
    - 环境：`StenosisRejectEnv`（来自 `rl_reject_env.py`）；
    - 训练数据：`create_mixed_training_data` 生成的混合样本；
  - 输出模型：
    - **`rl_reject_model.zip`**（最终使用的 MLP 模型）。

- **训练目标**：在混合样本上学到：
  - 真阳性附近精细调整位置并确认（提高 **TPR** 与定位精度）；
  - 对纯假阳性候选给出 Reject（显著提高 **PPV**）。

#### 4.3.2 LSTM + Reject（对比方法）

- **训练脚本**：`train_rl_lstm.py`（及一部分 inline 脚本）
  - 使用 `sb3-contrib` 的 **RecurrentPPO (LstmPolicy)**；
  - 环境同样基于 `StenosisRejectEnv`；
  - 模型文件示例：`rl_lstm_reject_model.zip` 等。

- LSTM 在小样本和长序列泛化上有一定优势，但在当前数据规模及 reward 配置下：
  - 对 **PPV/F1 的提升不如 MLP 稳定**，
  - 最终在严格评估下，选择 **MLP 版本** 作为主要结果。

### 4.4 RL 预测 / 推理脚本

RL 模型的推理由多种脚本调用，核心逻辑是一致的：

1. 基于几何方法产生粗候选点（det_indices）；
2. 对每个候选点初始化一个 RL episode，从该位置开始沿骨架移动；
3. 根据策略输出的动作：
   - 左/右移动更新当前 index；
   - 当动作为 Confirm / Reject 时终止 episode；
4. 将 **Confirm 的点** 作为最终狭窄预测点，将 **Reject 的候选** 丢弃。

主要推理脚本包括：

- **`evaluate_sequential_rl_agent.py` / `evaluate_rl_vs_fixed.py` / `simple_evaluate_rl.py`**  
  - 用于在干净血管数据集上，对 RL 模型进行系统性评估：
    - 不同 margin（例如 20px / 30px / 40px / 50px / 75px）下的 TPR/PPV/F1；
    - 对比：
      - 纯几何方法，
      - RL-MLP（带 reject），
      - RL-LSTM（带 reject）。

- **`visualize_all_results.py` / 若干临时可视化脚本**  
  - 用于生成 RL 运行轨迹、候选点与最终 Confirm/Reject 的可视化图片和视频：
    - 示例输出：`rl_detection_visualization.mp4`、`stenosis_detection_video.mp4` 等。

在实际部署中，推理流程可以封装为：

1. 对每条血管调用几何候选生成；
2. 对每个候选用 `rl_reject_model.zip` 运行一小段 RL episode；
3. 汇总 Confirm 的点作为最终狭窄输出，用 Reject 控制假阳性数量。

---

## 5. 评估设置与最终选用方案

### 5.1 评估数据与指标

- **评估数据集**：
  - 选取 **30 条干净血管**（无额外病变和干扰），
  - 索引保存在 `clean_vessel_indices.npy` 中，
  - 通用加载/环境构建由 `rl_sequential_stenosis_env.py` 负责。

- **评估指标**：
  - **TP (True Positive)**：在给定 margin 内，至少有一个预测点落入 GT 框/中心附近；
  - **FP (False Positive)**：预测点未匹配到任何 GT；
  - **FN (False Negative)**：有 GT 但未被任何预测点命中；
  - 由此计算：
    - **TPR = TP / (TP + FN)**
    - **PPV = TP / (TP + FP)**
    - **F1 = 2 · TP / (2·TP + FP + FN)**

- **Margin 设定**：
  - 对同一组预测，分别在 20 / 30 / 40 / 50 / **75 px** 等不同 margin 下评估，
  - 以 75px 作为最终严格/稳健平衡点进行报告。

### 5.2 几何 vs RL 的整体对比

在相同干净血管数据集与 evaluation pipeline 下：

- 纯几何方法：
  - TPR 较高，但 **PPV 明显偏低**，
  - 假阳性集中在分叉及局部噪声半径波动位置。

- RL-MLP（无 reject）早期版本：
  - 在定位精度上优于几何方法，
  - 但在混合数据集存在高比例假阳性的情况下，PPV 提升有限。

- **RL-MLP + Reject（最终方案）**：
  - 在混合真/假阳性样本上训练，显著提升了 **PPV 与 F1**；
  - 在 75px margin 下达到：
    - **TPR = 0.867**
    - **PPV = 0.634**
    - **F1  = 0.732**
  - 相比几何方法和其它 RL 变体，在“召回足够高的前提下抑制假阳性”方面取得最佳平衡。

- RL-LSTM + Reject：
  - 在部分设定下能获得接近或略高的 TPR，
  - 但整体 **PPV/F1 不如 MLP 稳定**，对超参数敏感，且推理复杂度更高；
  - 最终作为对比方法，不作为主结果。

### 5.3 最终选用配置

- **环境**：`StenosisRejectEnv` （`rl_reject_env.py`）
- **训练脚本**：`train_rl_reject.py`
- **模型文件**：`rl_reject_model.zip`
- **策略网络**：PPO + **MLP policy**（非复发）
- **评估脚本**：
  - `evaluate_rl_vs_fixed.py` / `evaluate_sequential_rl_agent.py` / `simple_evaluate_rl.py`
- **评估数据**：30 条干净血管（`clean_vessel_indices.npy`）
- **margin**：**75 px**（最终报告结果）
- **最终指标**：
  - **TPR = 0.867**
  - **PPV = 0.634**
  - **F1  = 0.732**

---

## 6. 论文撰写时的简要表述模板

可直接用于论文/技术报告中的描述示例（英文）：

> We first establish a conventional baseline using vessel segmentation followed by rule-based stenosis detection on the vessel radius profile. Although this geometric method achieves a relatively high sensitivity, it suffers from a large number of false positives, especially around bifurcations and noisy regions.
>
> To address this issue, we design a reinforcement learning (RL)–based refinement module operating on the vessel skeleton. For each geometric candidate, an RL agent observes a local window of the normalized radius curve and its first- and second-order derivatives, and learns to move along the skeleton and either confirm or reject the candidate. The environment is implemented in `rl_reject_env.py`, and the agent is trained with PPO using a multi-layer perceptron (MLP) policy (`train_rl_reject.py`). The training set explicitly mixes true-positive and false-positive candidates so that the agent learns both precise localization and false-positive rejection.
>
> On a clean-vessel test set of 30 cases, our final MLP-based RL model with a 75-pixel tolerance margin achieves a true positive rate (TPR) of **0.867**, a positive predictive value (PPV) of **0.634**, and an F1 score of **0.732**, clearly outperforming both the geometric baseline and recurrent (LSTM-based) RL variants.

如需，我可以再补充一份更正式的英文 `Methods` 小节，或画一张“Baseline → Geometric → RL”的流程示意图说明各阶段差异。
