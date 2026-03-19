# 序列决策RL狭窄检测 - 使用指南

基于**RL思路.md**设计的**PPO-LSTM序列决策Agent**，用于血管狭窄检测。

## 📋 核心概念

### 与传统方法的区别

| 特性 | Statistical方法 | **RL Sequential方法** ⭐ |
|------|----------------|------------------------|
| 决策方式 | 滑动窗口 + 阈值 | 序列决策 + 记忆模块 |
| 上下文感知 | 局部窗口（固定） | 全局趋势（LSTM记忆） |
| 自适应能力 | 固定参数 | 自适应策略 |
| 误报控制 | 依赖阈值调优 | RL自动学习 |

### 算法亮点

1. **序列决策**：Agent像机器人沿血管中心线逐点判断
2. **LSTM记忆**：记住走过的血管粗细，区分"渐变"vs"突变"
3. **奖励塑造**：
   - TP (正确检测): +1.0
   - FP (误报): -2.0 (严厉惩罚)
   - FN (漏检): -2.0 (医疗AI大忌)
   - TN (正确放过): +0.1
   - 连续性奖励: 鼓励连续标记狭窄区域

---

## 🚀 快速开始

### 前置条件

确保已经：
1. ✅ 运行过 `batch_inference_xzjc.py` 生成血管分割mask
2. ✅ 数据集目录包含 `.bmp` 图像和 `.xml` 标注

### 数据准备

检查数据结构：
```bash
/mnt/sda1/luoyu/xzjc_data/
├── dataset/
│   ├── 14_002_5_0016.bmp      # 原始图像
│   ├── 14_002_5_0016.xml      # 狭窄标注
│   ├── 14_002_5_0017.bmp
│   └── ...
└── masks/
    ├── 14_002_5_0016.png      # 分割mask
    ├── 14_002_5_0017.png
    └── ...
```

---

## 📝 Step 1: 训练RL Agent

### 基础训练（推荐从这里开始）

```bash
cd /mnt/sda1/luoyu/SAM-VMNet

# 小规模训练（快速验证）
python train_sequential_rl_agent.py \
    --total_timesteps 50000 \
    --n_envs 1 \
    --save_freq 5000 \
    --eval_freq 2500

# 预计时间: ~30分钟（取决于数据集大小）
```

### 完整训练（发表论文级别）

```bash
python train_sequential_rl_agent.py \
    --total_timesteps 200000 \
    --n_envs 4 \
    --learning_rate 3e-4 \
    --lstm_hidden_size 128 \
    --batch_size 64 \
    --save_freq 10000 \
    --eval_freq 5000

# 预计时间: 2-3小时
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total_timesteps` | 100000 | 总训练步数 |
| `--n_envs` | 1 | 并行环境数（建议1-4） |
| `--learning_rate` | 3e-4 | 学习率 |
| `--lstm_hidden_size` | 128 | LSTM隐藏层大小 |
| `--window_size` | 5 | 局部窗口大小 |
| `--max_episode_steps` | 500 | 每条血管最大步数 |

### 训练输出

```
rl_sequential_logs/
└── run_20231129_150230/          # 训练运行目录
    ├── best_model/                # 最佳模型（F1最高）
    │   └── best_model.zip
    ├── final_model.zip            # 最终模型
    ├── checkpoints/               # 中间Checkpoint
    │   ├── ppo_lstm_stenosis_10000_steps.zip
    │   └── ...
    ├── training_curves.png        # 训练曲线
    ├── training_metrics.npz       # 训练指标
    └── tensorboard/               # TensorBoard日志
```

### 监控训练

```bash
# 实时查看训练曲线
tensorboard --logdir rl_sequential_logs/run_20231129_150230/tensorboard

# 浏览器打开: http://localhost:6006
```

---

## 📊 Step 2: 评估RL Agent

### 对比评估（RL vs Statistical Baseline）

```bash
python evaluate_sequential_rl_agent.py \
    --model_path rl_sequential_logs/run_20231129_150230/best_model/best_model.zip \
    --num_samples 100

# 评估结果保存在 ./rl_evaluation_results/
```

### 评估输出

```
rl_evaluation_results/
├── comparison_report.txt          # 详细对比报告
└── comparison_chart.png           # 可视化对比图
```

### 示例报告

```
================================================================================
序列决策RL Agent vs Statistical Baseline - 对比评估报告
================================================================================

【RL Agent (PPO-LSTM)】
--------------------------------------------------------------------------------
Ground Truth总数: 450
检测总数: 423
True Positives: 380
False Positives: 43
False Negatives: 70

Sensitivity: 0.8444 (84.44%)
Precision: 0.8984 (89.84%)
F1-Score: 0.8705

【Statistical Baseline】
--------------------------------------------------------------------------------
Ground Truth总数: 450
检测总数: 467
True Positives: 360
False Positives: 107
False Negatives: 90

Sensitivity: 0.8000 (80.00%)
Precision: 0.7709 (77.09%)
F1-Score: 0.7852

【性能对比】
--------------------------------------------------------------------------------
Sensitivity提升: +0.0444 (+4.44%)
Precision提升: +0.1275 (+12.75%)
F1-Score提升: +0.0853 (+8.53%)

✅ RL Agent性能更优！F1提升 8.53%
```

---

## 🎯 Step 3: 消融实验（Ablation Study）

验证LSTM模块的有效性（论文必备）。

### 实验A: PPO Agent (无LSTM，只有全连接层)

修改 `rl_ppo_lstm_policy.py` 中的 `lstm_hidden_size=0` 或直接使用MLP。

```bash
# 训练无LSTM版本
python train_sequential_rl_agent.py \
    --lstm_hidden_size 0 \
    --total_timesteps 100000 \
    --log_dir rl_sequential_logs_no_lstm
```

### 实验B: PPO + LSTM Agent (完整版)

```bash
# 训练完整版本
python train_sequential_rl_agent.py \
    --lstm_hidden_size 128 \
    --total_timesteps 100000 \
    --log_dir rl_sequential_logs_with_lstm
```

### 预期结果

| 方法 | Sensitivity | Precision | F1-Score | 备注 |
|------|------------|-----------|----------|------|
| PPO (无LSTM) | 78-82% | 75-80% | 76-81% | 对渐变血管误报多 |
| **PPO-LSTM** ⭐ | **84-88%** | **88-92%** | **86-90%** | **记忆机制有效** |

---

## 🔧 高级使用

### 1. 自定义奖励函数

编辑 `rl_sequential_stenosis_env.py` 中的 `_compute_reward()` 方法：

```python
def _compute_reward(self, action: int, current_point: Tuple) -> float:
    # 自定义奖励
    if action == 1:  # 检测为狭窄
        if is_near_gt:
            reward = 2.0  # 增加TP奖励
        else:
            reward = -3.0  # 更严厉惩罚FP
    else:
        if is_near_gt:
            reward = -3.0  # 更严厉惩罚FN
        else:
            reward = 0.2  # 增加TN奖励
    
    return reward
```

### 2. 调整窗口大小

不同窗口大小的效果：

```bash
# 小窗口（更敏感，适合细节）
python train_sequential_rl_agent.py --window_size 3

# 大窗口（更稳定，适合粗血管）
python train_sequential_rl_agent.py --window_size 7
```

### 3. 多GPU并行训练

```bash
# 使用4个并行环境
python train_sequential_rl_agent.py \
    --n_envs 4 \
    --total_timesteps 200000
```

---

## 📈 性能优化建议

### 如果Sensitivity偏低 (<80%)

- ✅ **降低FN惩罚**：`FN_reward = -3.0`（更严厉）
- ✅ **增加窗口大小**：`--window_size 7`
- ✅ **延长训练**：`--total_timesteps 300000`

### 如果Precision偏低 (<75%)

- ✅ **增加FP惩罚**：`FP_reward = -3.0`
- ✅ **减小窗口大小**：`--window_size 3`
- ✅ **增加探索**：`--ent_coef 0.02`

### 如果训练不稳定

- ✅ **降低学习率**：`--learning_rate 1e-4`
- ✅ **增加Batch Size**：`--batch_size 128`
- ✅ **使用单环境**：`--n_envs 1`

---

## 🐛 常见问题

### Q1: 训练时报错 "CUDA out of memory"

**解决**：
```bash
# 方法1: 使用CPU
export CUDA_VISIBLE_DEVICES=""

# 方法2: 减少并行环境
python train_sequential_rl_agent.py --n_envs 1
```

### Q2: F1-Score一直在0.3左右不提升

**原因**: 奖励函数可能不平衡

**解决**:
1. 检查数据集是否有Ground Truth
2. 调整奖励权重（增加TP奖励）
3. 延长训练时间

### Q3: 评估时与训练指标差距大

**原因**: 过拟合或评估集与训练集分布不同

**解决**:
1. 使用 `EvalCallback` 在训练时定期评估
2. 增加 `--ent_coef 0.02` 鼓励探索
3. 数据增强

---

## 📚 论文写作建议

### 方法论部分（Methodology）

引用核心设计：
```
我们提出了一种基于深度强化学习的序列决策方法（PPO-LSTM），
将狭窄检测建模为马尔可夫决策过程。Agent沿血管中心线逐点判断，
通过LSTM记忆模块区分自然变细和病理狭窄。
```

### 实验设计（Experiments）

必备实验：
1. ✅ **对比实验**: RL vs Statistical Baseline
2. ✅ **消融实验**: PPO vs PPO-LSTM
3. ✅ **泛化实验**: 测试集性能

### 结果展示（Results）

表格示例：
```markdown
| Method | Sensitivity | Precision | F1-Score |
|--------|------------|-----------|----------|
| Statistical | 80.0% | 77.1% | 78.5% |
| PPO (no LSTM) | 81.2% | 79.3% | 80.2% |
| **PPO-LSTM (Ours)** | **87.1%** | **89.8%** | **88.4%** |
```

---

## 🎓 理论参考

- **PPO算法**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- **LSTM网络**: Hochreiter & Schmidhuber, "Long Short-Term Memory", 1997
- **医疗RL**: Gottesman et al., "Guidelines for RL in Healthcare", 2019

---

## 📞 联系与支持

如果遇到问题，请检查：
1. ✅ 数据集格式是否正确
2. ✅ Mask是否已生成
3. ✅ XML标注是否存在

**Happy Training! 🚀**
