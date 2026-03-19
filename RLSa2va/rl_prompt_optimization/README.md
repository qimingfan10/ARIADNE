# Sa2VA Prompt优化强化学习

使用强化学习（PPO）优化Sa2VA的prompt策略，提高血管分割的Dice和Recall指标。

## 📊 背景

当前Sa2VA模型性能：
- **Dice Score**: 0.8191
- **Recall**: 0.7763 ⚠️ (偏低，约22%血管未检测)
- **Precision**: 0.8742 ✅ (较好)

**目标**: 通过RL学习最优prompt策略，将Dice提升至0.85+，Recall提升至0.85+

---

## 🎯 核心思想

不修改模型参数，而是学习如何选择最优的文本prompt来引导Sa2VA生成更完整的分割结果。

### Prompt候选库（11个）
1. 基础: `"segment the blood vessel"`
2. 强调完整性: `"segment all blood vessels including small branches"`
3. 强调细节: `"carefully segment blood vessel with thin branches"`
4. 强调低对比度: `"segment blood vessel even in low contrast regions"`
5. ...等

### RL框架
- **环境**: `PromptOptimizationEnv` (基于Gymnasium)
- **算法**: PPO (Proximal Policy Optimization)
- **状态**: 当前Dice, Recall, 步数, GT统计等
- **动作**: 选择11个prompt之一
- **奖励**: Dice提升 + Recall提升 + Precision保持

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /home/ubuntu/Sa2VA/rl_prompt_optimization
pip install -r requirements.txt
```

### 2. 快速测试（5-10分钟）

使用少量样本快速验证RL框架是否正常工作：

```bash
bash quick_start.sh
```

### 3. 完整训练（1-2小时）

使用全部数据集训练完整的RL策略：

```bash
bash full_train.sh
```

### 4. 评估训练好的策略

```bash
python evaluate_rl_prompt.py \
    --rl_model_path outputs/rl_prompt_*/best_model/best_model.zip \
    --split val
```

---

## 📂 项目结构

```
rl_prompt_optimization/
├── env/
│   ├── __init__.py
│   └── prompt_env.py           # RL环境定义
│
├── train_rl_prompt.py          # 训练脚本
├── evaluate_rl_prompt.py       # 评估脚本
│
├── quick_start.sh              # 快速测试
├── full_train.sh               # 完整训练
│
├── requirements.txt            # 依赖包
└── README.md                   # 本文档
```

---

## 🔧 使用方法

### 训练自定义参数

```bash
python train_rl_prompt.py \
    --model_path /path/to/sa2va_vessel_hf \
    --data_root /path/to/data \
    --output_dir ./outputs \
    --max_steps 5 \
    --total_timesteps 50000 \
    --learning_rate 3e-4 \
    --batch_size 64
```

### 主要参数说明

- `--max_steps`: 每个episode最大步数（每步尝试一个prompt）
- `--total_timesteps`: PPO总训练步数
- `--learning_rate`: 学习率
- `--batch_size`: PPO批次大小
- `--max_samples`: 限制训练样本数（用于快速测试）

---

## 📊 查看训练进度

### TensorBoard

```bash
tensorboard --logdir outputs/rl_prompt_*/logs
```

打开浏览器访问 `http://localhost:6006`

### 主要监控指标
- **episode_reward**: 每个episode的总奖励
- **dice_score**: 平均Dice分数
- **recall_score**: 平均Recall
- **policy_loss**: 策略损失
- **value_loss**: 价值损失

---

## 🎓 技术细节

### 环境设计

```python
class PromptOptimizationEnv(gym.Env):
    # 状态空间: [当前Dice, Recall, 步数, GT统计, ...]
    observation_space = Box(low=0, high=1, shape=(16,))
    
    # 动作空间: 选择11个prompt之一
    action_space = Discrete(11)
```

### 奖励函数

```python
reward = (
    dice_improvement * 20.0 +      # Dice提升奖励
    recall_improvement * 15.0 +    # Recall提升奖励（重点）
    bonus_high_quality +           # 高质量奖励（Dice>0.85）
    efficiency_bonus -             # 效率奖励
    precision_penalty              # Precision保持惩罚
)
```

### PPO配置

- **学习率**: 3e-4
- **批次大小**: 64
- **优化轮数**: 10
- **GAE Lambda**: 0.95
- **Clip范围**: 0.2

---

## 📈 预期效果

### 快速测试（5000步）
- 验证框架正常工作
- 初步观察prompt效果
- 时间: 5-10分钟

### 完整训练（50000步）
- **Dice提升**: +2-5% → 0.84-0.86
- **Recall提升**: +3-6% → 0.81-0.84
- **训练时间**: 1-2小时（取决于GPU）

---

## 🔍 调试技巧

### 1. 检查环境是否正常

```python
from rl_prompt_optimization.env import PromptOptimizationEnv
# 创建环境并测试一个episode
```

### 2. 监控GPU使用

```bash
watch -n 1 nvidia-smi
```

### 3. 查看日志

训练日志会保存在 `outputs/rl_prompt_*/logs/` 目录下

---

## 🛠️ 常见问题

### Q1: 训练太慢怎么办？
**A**: 使用 `--max_samples 100` 限制训练样本数，或减少 `--total_timesteps`

### Q2: 内存不足怎么办？
**A**: 减少 `--batch_size` 或 `--n_steps`

### Q3: 如何选择最优模型？
**A**: 使用 `best_model/best_model.zip`，这是在验证集上性能最好的模型

### Q4: 如何添加新的prompt？
**A**: 编辑 `env/prompt_env.py` 中的 `self.prompts` 列表

---

## 📝 下一步

1. ✅ **方案1完成**: Prompt优化RL
2. ⏭️ **方案2**: 后处理优化RL（如果方案1效果不足）
3. ⏭️ **方案3**: Reward Network微调（如果需要更大提升）

---

## 📚 参考

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **PPO论文**: https://arxiv.org/abs/1707.06347
- **RL4Seg**: https://arxiv.org/abs/2406.17902

---

**创建时间**: 2025-11-29  
**状态**: 已实现，待测试  
**推荐**: 先运行 `quick_start.sh` 进行快速验证
