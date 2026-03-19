# Sa2VA LoRA + PPO微调

**目标**: 通过强化学习直接优化Dice指标，从0.78提升到0.87+

## 🎯 方案概述

### 核心思想
1. **LoRA适配**: 只训练0.5%参数（~130M），冻结Sa2VA主干
2. **PPO优化**: 使用强化学习直接优化Dice/Recall
3. **多目标奖励**: 结合Dice + Recall + 拓扑连通性 + 长度约束

### 技术路线
```
Sa2VA-26B (冻结)
    ↓
LoRA适配器 (可训练)
    ↓
预测Mask
    ↓
多目标奖励函数
    ↓
PPO优化
```

---

## 📁 文件结构

```
lora_ppo_training/
├── reward_functions.py      # 奖励函数（核心）
├── lora_config.py            # LoRA配置
├── data_loader.py            # 数据加载
├── train_lora_ppo.py         # 主训练脚本
├── run_lora_ppo.sh           # 运行脚本
├── README.md                 # 本文件
└── output/                   # 输出目录
    ├── sa2va_lora_ppo_*/     # 训练输出
    │   ├── best_lora/        # 最佳模型
    │   ├── final_lora/       # 最终模型
    │   └── training_info.json
    └── train_*.log           # 日志文件
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install peft>=0.6.0
pip install scikit-image
pip install wandb  # 可选，用于实验追踪
```

### 2. 快速测试（推荐先运行）

```bash
bash run_lora_ppo.sh quick
```

**配置**:
- 训练样本: 50张
- 验证样本: 20张
- 训练轮数: 1 epoch
- 预计时间: ~30分钟

**目的**: 验证代码是否正常运行，无OOM错误

### 3. 完整训练

```bash
bash run_lora_ppo.sh full
```

**配置**:
- 训练样本: 1000张
- 验证样本: 100张
- 训练轮数: 3 epochs
- 预计时间: ~24-48小时

---

## ⚙️ 核心组件

### 1. 奖励函数 (`reward_functions.py`)

#### MultiObjectiveReward (默认)
```python
reward = 0.5 * dice_reward +
         0.2 * recall_bonus +
         0.2 * topology_reward +
         0.1 * length_penalty
```

**各项说明**:
- **Dice奖励** (50%): 主要优化目标
- **Recall奖励** (20%): 针对性提升Recall (目标0.85)
- **拓扑奖励** (20%): 保证血管连续性，减少断裂
- **长度惩罚** (10%): 约束血管总长度接近GT

#### SimpleDiceReward
仅使用Dice作为奖励，最简单

#### RecallFocusedReward
专注于提升Recall，权重70%

### 2. LoRA配置 (`lora_config.py`)

**预设配置**:
```python
'medium': {
    'lora_rank': 32,
    'lora_alpha': 64,
    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```

**可训练参数**: ~130M (0.5% of 26B)

### 3. 训练器 (`train_lora_ppo.py`)

**核心流程**:
```python
for epoch in epochs:
    for batch in dataloader:
        # 1. 使用当前策略预测
        pred_mask = model.predict(image)
        
        # 2. 计算奖励
        reward = reward_function(pred_mask, gt_mask)
        
        # 3. 更新LoRA参数
        loss = -reward
        loss.backward()
        optimizer.step()
```

**注意**: 这是简化版本，完整的PPO实现更复杂

---

## 📊 监控指标

### 训练指标
- `train/reward`: 训练奖励
- `train/dice`: 训练Dice分数
- `train/recall`: 训练Recall
- `train/precision`: 训练Precision

### 验证指标
- `val/dice`: 验证Dice ⭐ (主要关注)
- `val/recall`: 验证Recall
- `val/precision`: 验证Precision

### 目标
- Dice: 0.87+
- Recall: 0.85+
- Precision: 0.85+

---

## 🔧 超参数调优

### LoRA参数
```bash
# 更大的rank（更多参数，更强表达力）
--lora_rank 64 --lora_alpha 128

# 更小的rank（更少参数，更快训练）
--lora_rank 16 --lora_alpha 32
```

### 学习率
```bash
# 激进（快速收敛，可能不稳定）
--learning_rate 1e-4

# 保守（稳定训练，收敛慢）
--learning_rate 1e-5
```

### 奖励函数
```bash
# 专注Recall
--reward_type recall_focused

# 专注Dice
--reward_type simple_dice

# 平衡优化（推荐）
--reward_type multi_objective
```

### Recall目标
```bash
# 更高的Recall目标
--recall_target 0.90

# 放松Recall要求
--recall_target 0.80
```

---

## 🎓 使用技巧

### 1. 渐进式训练

**Step 1**: Quick模式验证 (50张, 1 epoch)
```bash
bash run_lora_ppo.sh quick
```

**Step 2**: 中等规模 (200张, 2 epochs)
```bash
bash run_lora_ppo.sh full
# 但修改脚本中的 MAX_TRAIN_SAMPLES=200
```

**Step 3**: 完整训练 (1000张, 3 epochs)
```bash
bash run_lora_ppo.sh full
```

### 2. Curriculum Learning

**阶段1**: 训练简单样本（大血管）
```python
# 修改data_loader.py，按血管面积排序
# 先训练大血管图像
```

**阶段2**: 加入中等复杂度样本

**阶段3**: 加入困难样本（细小血管）

### 3. 奖励塑形（Reward Shaping）

**动态权重调整**:
```python
# 在training中
if current_recall < 0.80:
    recall_weight = 0.4  # 提高Recall权重
else:
    recall_weight = 0.2  # 恢复正常
```

---

## 🐛 故障排除

### 问题1: GPU内存不足 (OOM)

**解决方案**:
```bash
# 1. 降低LoRA rank
--lora_rank 16

# 2. 使用更小的模型精度
--use_bf16

# 3. 减少batch size（已经是1了）
# 4. 使用gradient checkpointing
```

### 问题2: 训练不收敛

**解决方案**:
```bash
# 1. 降低学习率
--learning_rate 1e-5

# 2. 增加warmup步数
# 3. 使用简单的奖励函数
--reward_type simple_dice

# 4. 检查数据是否有问题
```

### 问题3: Recall提升但Precision下降

**解决方案**:
```bash
# 调整奖励权重，增加Precision约束
--dice_weight 0.6 --recall_weight 0.1

# 或者在奖励函数中添加Precision惩罚
```

### 问题4: 预测失败

**检查**:
```bash
# 查看日志
tail -100 output/train_*.log | grep "预测失败"

# 确认prompt格式
--prompt "<image>\nPlease segment the blood vessel."
```

---

## 📈 预期结果

### Quick模式
```yaml
训练时间: ~30分钟
预期Dice: 0.78-0.80
预期Recall: 0.74-0.76
结论: 验证代码可行性
```

### Full模式
```yaml
训练时间: ~24-48小时
预期Dice: 0.85-0.87+
预期Recall: 0.83-0.85+
结论: 达到或接近目标
```

---

## 🎯 下一步

### 训练完成后

1. **评估模型**
```bash
python evaluate_lora_model.py \
    --base_model /path/to/sa2va_vessel_hf \
    --lora_weights output/sa2va_lora_ppo_*/best_lora \
    --test_data /path/to/test_data
```

2. **对比实验**
- 与实验一、二、三对比
- 选择最优方案

3. **部署**
```bash
# 合并LoRA权重（可选）
python merge_lora.py \
    --base_model /path/to/sa2va_vessel_hf \
    --lora_weights output/sa2va_lora_ppo_*/best_lora \
    --output_model /path/to/sa2va_merged
```

---

## 📚 技术参考

### 核心论文
1. **LoRA**: "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
2. **PPO**: "Proximal Policy Optimization Algorithms" (2017)
3. **RLHF**: "Training language models to follow instructions" (NeurIPS 2022)

### 关键概念
- **Low-Rank Adaptation**: 通过低秩矩阵近似减少可训练参数
- **Policy Gradient**: 直接优化策略的梯度方法
- **Reward Shaping**: 设计奖励函数引导学习方向

---

## ⚠️ 注意事项

1. **训练时间**: Full模式需要24-48小时，请确保有足够时间
2. **GPU需求**: 推荐4×A100或2×A100+DeepSpeed
3. **数据质量**: 标注质量直接影响训练效果
4. **超参敏感**: LoRA和学习率需要仔细调优
5. **简化实现**: 当前是简化版PPO，完整版需要更复杂逻辑

---

## 🎉 期待效果

通过LoRA + PPO微调，预期达到：
- ✅ Dice: 0.87+ (从0.78提升11.5%)
- ✅ Recall: 0.85+ (从0.74提升14.9%)
- ✅ 血管连续性改善（拓扑奖励）
- ✅ 细小血管完整检出

**成功标准**: Dice ≥ 0.87 且 Recall ≥ 0.85

---

**创建时间**: 2025-11-29  
**状态**: ✅ 代码就绪，准备训练  
**建议**: 先运行Quick模式验证
