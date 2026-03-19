# RL血管狭窄检测方法论文档

## 📌 概述

本项目使用**强化学习（Reinforcement Learning）** 方法进行血管狭窄点的序列检测。Agent沿着血管骨架逐点移动，在每个位置决定是否标记为狭窄。

---

## 🏆 当前最优模型

### 推荐使用：MLP V9

| 指标 | 值 |
|------|-----|
| **F1-Score** | 0.069 |
| **Sensitivity** | 4.0% |
| **Precision** | 25.0% |
| **对比Baseline** | +64% |

**权重文件位置**：
```
# Best Model（推荐用于推理）
rl_simple_logs/run_v9_20251202_182539/run_20251202_182543/best_model/best_model.zip

# Final Model（完整训练）
rl_simple_logs/run_v9_20251202_182539/run_20251202_182543/final_model.zip
```

---

## 🧠 方法论

### 1. 问题建模 (MDP)

将狭窄检测建模为**马尔可夫决策过程**：

- **状态 (State)**: 当前骨架点的局部特征向量（16维）
  - 局部半径序列 (5个点)
  - 一阶导数、二阶导数
  - Z-score异常分数
  - 归一化位置、平均半径
  
- **动作 (Action)**: 二元决策
  - `0`: 正常点
  - `1`: 狭窄点

- **奖励 (Reward)**: 基于预测准确性

### 2. Reward设计（V9配置）

基于**期望收益**数学推导，确保极端策略（全0/全1）都亏损：

```python
# V9 Reward配置
TP = +50.0   # True Positive（正确检测狭窄）
FP = -5.0    # False Positive（误报）
FN = -10.0   # False Negative（漏检）
TN = +0.1    # True Negative（正确放过）
```

**期望收益分析**（GT比例约4.37%）：
```
全1策略: 0.0437×50 + 0.9563×(-5) = -2.60 📉 亏损
全0策略: 0.0437×(-10) + 0.9563×0.1 = -0.34 📉 亏损
完美预测: 0.0437×50 + 0.9563×0.1 = +2.28 📈 盈利
```

### 3. 算法

- **MLP模型**: PPO (Proximal Policy Optimization) + MLP网络
- **LSTM模型**: RecurrentPPO + LSTM网络（效果不佳，不推荐）

### 4. Ground Truth匹配

- 使用**Bounding Box + Margin**方式匹配
- Margin = 10像素（补偿标注偏差）
- GT坐标从原始图像尺寸缩放到512×512

---

## 📂 文件结构

```
SAM-VMNet/
├── rl_sequential_stenosis_env.py    # RL环境定义
├── train_sequential_rl_simple.py    # MLP训练脚本
├── train_sequential_rl_recurrent.py # LSTM训练脚本
├── skeleton_cache.pkl               # 预计算骨架缓存（334MB）
│
├── rl_simple_logs/                  # MLP训练日志
│   └── run_v9_20251202_182539/
│       └── run_20251202_182543/
│           ├── best_model/best_model.zip  ⭐ 推荐
│           ├── final_model.zip
│           └── checkpoints/
│
└── rl_recurrent_logs/               # LSTM训练日志（效果不佳）
```

---

## 🚀 使用方法

### 1. 推理/预测

```python
import sys
sys.path.insert(0, '/mnt/sda1/luoyu/SAM-VMNet')

from stable_baselines3 import PPO
from rl_sequential_stenosis_env import SequentialStenosisEnv

# 加载模型
model_path = "rl_simple_logs/run_v9_20251202_182539/run_20251202_182543/best_model/best_model.zip"
model = PPO.load(model_path)

# 创建环境
env = SequentialStenosisEnv(
    dataset_dir="/mnt/sda1/luoyu/xzjc_data/dataset",
    mask_dir="/mnt/sda1/luoyu/xzjc_data/masks"
)

# 预测
obs, info = env.reset()
stenosis_points = []

while True:
    action, _ = model.predict(obs, deterministic=True)
    if action == 1:
        # 当前点被标记为狭窄
        current_point = env.skeleton_points[env.current_step - 1]
        stenosis_points.append(current_point)
    
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

print(f"检测到 {len(stenosis_points)} 个狭窄点")
```

### 2. 训练新模型

```bash
cd /mnt/sda1/luoyu/SAM-VMNet

# MLP训练（推荐）
python train_sequential_rl_simple.py \
    --total_timesteps 200000 \
    --log_dir rl_simple_logs/run_v10_$(date +%Y%m%d_%H%M%S)

# LSTM训练（不推荐）
python train_sequential_rl_recurrent.py \
    --total_timesteps 200000 \
    --log_dir rl_recurrent_logs/run_v10_$(date +%Y%m%d_%H%M%S)
```

### 3. 使用预计算缓存加速训练

缓存文件：`skeleton_cache.pkl`（已生成，334MB，包含8325个样本）

需要修改环境代码以支持缓存加载（待实现）。

---

## 📊 实验历史

| 版本 | Reward配置 | MLP F1 | LSTM F1 | 备注 |
|------|------------|--------|---------|------|
| V5 | TP=20, FP=-2 | 0 | 0 | 全0策略 |
| V6 | TP=50, FP=-0.5 | ~0 | ~0 | 全1策略 |
| V7 | TP=100, FP=-1 | 0.0017 | 0 | 全1策略 |
| V8 | TP=100, FP=-3 | 0.0088 | 0 | 改善但不足 |
| **V9** | **TP=50, FP=-5** | **0.069** | 0 | ✅ 超过Baseline |

---

## 🔧 关键Bug修复记录

1. **骨架点坐标顺序**：`(point['x'], point['y'])` → `(row, col)`
2. **GT坐标缩放**：原始尺寸 → 512×512
3. **GT匹配方式**：距离阈值 → Bounding Box + Margin

---

## 📈 下一步优化方向

1. **提高Sensitivity**：当前仅4%，漏检较多
   - 增加训练时间
   - 调整FN惩罚
   - 尝试课程学习

2. **使用缓存加速**：预计提升10-20倍训练速度

3. **网络结构优化**：
   - 增加MLP层数/宽度
   - 尝试Attention机制

4. **数据增强**：增加正样本权重

---

## 📝 统计Baseline对比

| 方法 | F1-Score | Sensitivity | Precision |
|------|----------|-------------|-----------|
| Statistical (Z-score) | 0.042 | 高 | 低 |
| **RL MLP V9** | **0.069** | 4% | 25% |

RL方法在**精确度**上有明显优势，但灵敏度需要提升。

---

*文档更新时间：2024年12月3日*
*当前推荐模型：MLP V9*
