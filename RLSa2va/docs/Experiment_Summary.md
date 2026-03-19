# Sa2VA血管分割优化实验总结

## 基准性能
- **Baseline Dice: 0.8191** (10样本评估)
- 模型: sa2va_vessel_hf (26B参数)

## 实验结果

### 1. DPO系列 (V1-V9)
| 版本 | 学习率 | 方法 | 结果 |
|------|--------|------|------|
| V8a | 1e-7 | DPO + Dice hybrid | 无显著提升 |
| V8b | 1e-6 | DPO + Dice hybrid | 性能下降 |
| V8c | 1e-8 | DPO + Dice hybrid | 无显著提升 |
| V9 IPO | 1e-6 | IPO Loss + 激进筛选 | 无显著提升 |

**结论**: DPO/IPO无法提升像素级精度，目标错位。

### 2. Test Time Augmentation (TTA)
- 4x增强 (原图 + 水平翻转 + 垂直翻转 + 双翻转)
- **结果: Dice = 0.7800 (下降 3.9%)**
- **原因**: 模型对翻转变换不够鲁棒

### 3. REINFORCE (策略梯度)
- 直接用Dice作为reward
- **结果: Dice = 0.6804 (下降 16.9%)**
- **原因**: RL不收敛，策略梯度方差太大

### 4. Hard Sample SFT
- 只在Dice < 0.75的困难样本上训练
- **结果**: 模型保存后推理异常
- **原因**: LoRA merge_and_unload破坏模型结构

### 5. SAM2 Decoder Only
- 冻结LLM，只训练SAM2 mask decoder
- 训练集Dice: 0.7587
- **结果**: 训练有效但评估未完成

## 核心发现

1. **DPO目标错位**: DPO优化token概率，无法告诉模型"往左移一个像素"
2. **Embedding漂移**: 调整LLM权重导致[SEG] hidden state漂移，SAM2 decoder需要重新适应
3. **TTA假设不成立**: 模型对几何变换不够鲁棒，TTA反而引入噪声
4. **RL方差太大**: REINFORCE需要更多样本和更复杂的baseline设计

## 建议

1. **接受Baseline性能**: 0.82的Dice对于血管分割已经是不错的结果
2. **数据增强**: 在训练阶段加入翻转、旋转等增强，提高模型鲁棒性
3. **更多数据**: 收集更多困难样本，扩大训练集
4. **后处理**: 使用CRF或形态学操作优化分割边界

## 文件索引

### 训练脚本
- `/home/ubuntu/Sa2VA/scripts/train_dpo_v8_full_forward.py` - DPO V8
- `/home/ubuntu/Sa2VA/scripts/train_dpo_v9_ipo.py` - IPO Loss
- `/home/ubuntu/Sa2VA/scripts/train_reinforce.py` - REINFORCE
- `/home/ubuntu/Sa2VA/scripts/train_hard_sample_sft.py` - Hard Sample SFT
- `/home/ubuntu/Sa2VA/scripts/train_sam2_decoder_only.py` - SAM2 Decoder Only

### 评估脚本
- `/home/ubuntu/Sa2VA/evaluate_10_images.py` - Baseline评估
- `/home/ubuntu/Sa2VA/evaluate_with_tta.py` - TTA评估

### 文档
- `/home/ubuntu/Sa2VA/docs/DPO_Training_Documentation.md` - DPO方法论文档
- `/home/ubuntu/新策略.md` - 策略分析文档
