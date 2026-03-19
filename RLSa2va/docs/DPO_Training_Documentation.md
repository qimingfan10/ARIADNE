# Sa2VA DPO Training 技术文档

## 一、概述

本文档详细描述了对 Sa2VA-26B 模型进行 DPO (Direct Preference Optimization) 训练的方法论、实验过程和结论。

### 1.1 目标

通过 DPO 训练提升 Sa2VA 模型在血管分割（Vessel Segmentation）任务上的 Dice 系数。

### 1.2 基线性能

| 指标 | Baseline 值 |
|------|-------------|
| Mean Dice | 0.8191 |
| Mean IoU | 0.6966 |
| Precision | 0.8743 |
| Recall | 0.7763 |

---

## 二、文件结构

### 2.1 模型文件

```
/home/ubuntu/Sa2VA/models/sa2va_vessel_hf/    # Baseline模型（HuggingFace格式）
├── config.json
├── modeling_sa2va_chat.py                     # 模型定义
├── sam2.py                                    # SAM2 分割模块
├── tokenizer.json
├── pytorch_model-*.bin                        # 模型权重
└── ...
```

### 2.2 训练脚本

| 脚本 | 路径 | 说明 |
|------|------|------|
| **V8 Full Forward** | `/home/ubuntu/Sa2VA/scripts/train_dpo_v8_full_forward.py` | **最终版本** - 使用完整LLM forward路径 |
| V7 Hybrid | `/home/ubuntu/Sa2VA/scripts/train_dpo_v7_hybrid.py` | 混合DPO+Dice损失（简化embedding） |
| V6 Hybrid | `/home/ubuntu/Sa2VA/scripts/train_dpo_v6_hybrid.py` | 混合DPO+Dice损失 |

### 2.3 评估脚本

| 脚本 | 路径 | 说明 |
|------|------|------|
| **主评估脚本** | `/home/ubuntu/Sa2VA/evaluate_10_images.py` | 评估10张图片的分割性能 |

### 2.4 数据文件

```
/home/ubuntu/Sa2VA/data/dpo_vessel/
├── dpo_annotations.json          # DPO标注文件
├── images/                       # 原始图像
├── chosen_masks/                 # Chosen masks (Ground Truth)
└── rejected_masks/               # Rejected masks (Baseline预测)
```

**dpo_annotations.json 格式：**
```json
[
  {
    "image": "images/xxx.png",
    "chosen_mask": "chosen_masks/xxx.png",
    "rejected_mask": "rejected_masks/xxx.png"
  },
  ...
]
```

---

## 三、方法论

### 3.1 DPO 原理

DPO 通过最大化以下目标函数来学习偏好：

$$L_{DPO} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$

其中：
- $y_w$: Chosen (偏好) 样本
- $y_l$: Rejected (拒绝) 样本
- $\beta$: KL 散度惩罚系数
- $\pi_\theta$: 当前策略
- $\pi_{ref}$: 参考策略

### 3.2 分割任务的 DPO 适配

在分割任务中，我们将概率定义为基于 Dice 相似度：

```python
log_prob_chosen = log(Dice(pred_mask, gt_mask) + ε)
log_prob_rejected = log(Dice(pred_mask, rejected_mask) + ε)
```

### 3.3 混合损失函数

为防止 Dice 下降，采用混合损失：

$$L_{total} = L_{DPO} + \lambda \cdot L_{Dice}$$

其中 $\lambda = 1.0$（Dice损失权重）

### 3.4 数据筛选策略

**严格筛选标准：**
1. **Chosen = Ground Truth**：不使用模型预测作为chosen
2. **Dice Gap ≥ 0.15**：确保chosen和rejected有足够差异

```python
dice_gap = 1.0 - Dice(rejected_mask, gt_mask)
if dice_gap >= 0.15:
    # 使用该样本
```

### 3.5 模型冻结策略

| 组件 | 状态 | 原因 |
|------|------|------|
| Vision Encoder | ❄️ 冻结 | 防止过拟合 |
| LLM (LoRA) | 🔥 训练 | 学习偏好 |
| text_hidden_fcs | 🔥 训练 | 连接LLM和SAM2 |
| SAM2 Mask Decoder | 🔥 训练 | 优化分割质量 |

---

## 四、训练流程

### 4.1 环境准备

```bash
# 激活环境
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate topo-sarl
```

### 4.2 关键代码修改

**移除 sam2.py 中的 `@torch.no_grad` 装饰器：**

```bash
# 位置：/home/ubuntu/Sa2VA/models/sa2va_vessel_hf/sam2.py
# 行号：1691, 1699, 1708, 3905, 3977

# 将以下装饰器注释掉：
# @torch.no_grad()        -> # REMOVED for training
# @torch.inference_mode() -> # REMOVED for training
```

**在评估脚本中添加 `with torch.no_grad()`：**

```python
# /home/ubuntu/Sa2VA/evaluate_10_images.py
with torch.no_grad():
    result = model.predict_forward(
        image=image,
        text=text,
        tokenizer=tokenizer,
        processor=None,
    )
```

### 4.3 运行训练

```bash
cd /home/ubuntu/Sa2VA

# 运行 V8 训练（推荐）
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_dpo_v8_full_forward.py

# 输出目录
# /home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v8/final/
```

### 4.4 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 1e-5 | 中等学习率 |
| beta | 0.2 | DPO KL惩罚系数 |
| dice_weight | 1.0 | Dice损失权重 |
| lora_r | 16 | LoRA秩 |
| grad_accum | 4 | 梯度累积步数 |
| max_samples | 500 | 最大训练样本数 |

---

## 五、评估流程

### 5.1 运行评估

```bash
cd /home/ubuntu/Sa2VA

# 评估 Baseline
python evaluate_10_images.py

# 评估训练后的模型
# 修改 HF_MODEL_PATH 为训练输出路径
# HF_MODEL_PATH = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v8/final"
python evaluate_10_images.py
```

### 5.2 评估指标

- **Dice Score**: 分割重叠度 (2TP / (2TP + FP + FN))
- **IoU (Jaccard)**: 交并比 (TP / (TP + FP + FN))
- **Precision**: 精确率 (TP / (TP + FP))
- **Recall**: 召回率 (TP / (TP + FN))

---

## 六、实验结果

### 6.1 不同版本对比

| 版本 | 方法 | 学习率 | Mean Dice | 结论 |
|------|------|--------|-----------|------|
| Baseline | - | - | **0.8191** | 基准 |
| V6 | 简化embedding + Hybrid Loss | 1e-6 | 0.8191 | 无变化（梯度被阻断） |
| V7 | 移除装饰器 + Hybrid Loss | 1e-6 | 0.8190 | 无变化 |
| V8 | 完整LLM Forward | 1e-6 | 0.8193 | 微小变化 |
| V8b | 完整LLM Forward | 5e-5 | 0.7978 | ↓ 过拟合 |
| V8c | 完整LLM Forward | 1e-5 | 0.8188 | 基本持平 |

### 6.2 关键发现

1. **梯度阻断问题**：`@torch.no_grad` 装饰器必须移除才能进行训练
2. **训练-推理路径不一致**：训练用 `forward()`，推理用 `generate()`
3. **学习率敏感性**：太大导致遗忘，太小无效果
4. **Baseline已经很强**：Dice 0.82 难以通过DPO显著提升

---

## 七、核心代码解析

### 7.1 完整LLM Forward获取[SEG] Embedding

```python
def _forward_get_seg_embedding(self, pixel_values, input_ids):
    """使用完整LLM forward获取[SEG] embedding"""
    
    # 1. 获取vision embeddings
    vit_embeds = self.model.extract_feature(pixel_values)
    
    # 2. 获取text embeddings
    text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
    
    # 3. 替换IMG_CONTEXT位置为vision embeddings
    input_embeds = text_embeds.clone()
    img_context_mask = (input_ids == self.img_context_token_id)
    if img_context_mask.sum() > 0:
        vit_flat = vit_embeds.reshape(-1, C)
        img_positions = img_context_mask[0].nonzero(as_tuple=True)[0]
        input_embeds[0, img_positions] = vit_flat[:len(img_positions)]
    
    # 4. LLM forward获取hidden states
    outputs = self.model.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    
    # 5. 提取[SEG]位置的hidden state
    hidden_states = outputs.hidden_states[-1]
    seg_mask = (input_ids == self.seg_token_id)
    seg_hidden = hidden_states[seg_mask]
    
    # 6. 通过text_hidden_fcs
    seg_embedding = self.model.text_hidden_fcs(seg_hidden)
    
    return seg_embedding
```

### 7.2 DPO + Dice 混合损失

```python
def train_step(self, sample):
    # 预测mask
    pred_prob = torch.sigmoid(pred_logits)
    
    # Dice损失
    loss_dice = dice_loss(pred_prob, gt_mask)
    
    # DPO损失
    dice_with_gt = compute_dice(pred_prob, gt_mask)
    dice_with_rejected = compute_dice(pred_prob, rejected_mask)
    
    log_prob_chosen = torch.log(dice_with_gt + 1e-8)
    log_prob_rejected = torch.log(dice_with_rejected + 1e-8)
    loss_dpo = -F.logsigmoid(beta * (log_prob_chosen - log_prob_rejected))
    
    # 混合损失
    total_loss = loss_dpo + dice_weight * loss_dice
    
    return total_loss
```

---

## 八、常见问题

### Q1: 为什么Dice没有显著提升？

**A:** 主要原因：
1. Baseline已经很强（0.82），接近任务上限
2. 训练(`forward`)和推理(`generate`)路径不一致
3. DPO更适合偏好学习，不是精确像素级任务

### Q2: 如何确保梯度流动？

**A:** 
1. 移除 `sam2.py` 中的 `@torch.no_grad` 装饰器
2. 在评估时使用 `with torch.no_grad():` 上下文管理器
3. 确保 SAM2 Mask Decoder 的 `requires_grad=True`

### Q3: 推荐的下一步优化方向？

**A:**
1. 使用 XTuner 原生训练框架（统一训练推理路径）
2. 增加高质量DPO数据
3. 考虑直接监督学习替代DPO

---

## 九、附录

### 9.1 完整文件清单

```
训练相关:
├── /home/ubuntu/Sa2VA/scripts/train_dpo_v8_full_forward.py  # 主训练脚本
├── /home/ubuntu/Sa2VA/models/sa2va_vessel_hf/               # 基础模型
├── /home/ubuntu/Sa2VA/data/dpo_vessel/                      # DPO数据
└── /home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v8/          # 输出目录

评估相关:
├── /home/ubuntu/Sa2VA/evaluate_10_images.py                 # 评估脚本
└── /home/ubuntu/Sa2VA/data/dpo_vessel/                      # 测试数据

文档:
└── /home/ubuntu/Sa2VA/docs/DPO_Training_Documentation.md    # 本文档
```

### 9.2 参考文献

1. DPO: Direct Preference Optimization (Rafailov et al., 2023)
2. SAM2: Segment Anything Model 2 (Meta AI, 2024)
3. Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding

---

**文档版本**: 1.0  
**最后更新**: 2024年12月  
**作者**: AI Assistant
