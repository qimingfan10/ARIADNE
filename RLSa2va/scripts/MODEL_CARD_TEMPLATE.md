---
language: en
license: apache-2.0
tags:
- medical-imaging
- vessel-segmentation
- oct
- multimodal
- vision-language
- sa2va
- internvl
- sam2
datasets:
- custom-oct-vessel
metrics:
- iou
- dice
- precision
- recall
library_name: transformers
pipeline_tag: image-segmentation
---

# Sa2VA: Segment Anything to Vessel Analysis

<div align="center">
  <img src="https://img.shields.io/badge/Task-Vessel%20Segmentation-blue" alt="Task"/>
  <img src="https://img.shields.io/badge/Model-Vision--Language-green" alt="Model"/>
  <img src="https://img.shields.io/badge/Data-OCT-orange" alt="Data"/>
</div>

## 模型描述

Sa2VA（Segment Anything to Vessel Analysis）是一个创新的多模态视觉-语言模型，专门用于医学图像中的血管分割任务。该模型将大规模预训练的InternVL-8B与专业分割模型SAM2-Large相结合，通过自然语言交互实现高精度的自动分割。

## 模型详情

- **开发者**: qimingfan10
- **模型类型**: 多模态视觉-语言分割模型
- **架构**: InternVL-8B (Vision-Language) + SAM2-Large (Segmentation)
- **参数量**: ~14B
- **训练数据**: 9,346张OCT视网膜血管图像
- **任务**: 医学图像血管分割

## 性能指标

在OCT视网膜血管数据集上的表现：

| 指标 | 值 |
|------|-----|
| IoU (Intersection over Union) | 0.6725 |
| Dice系数 | 0.8005 |
| Precision | 0.8659 |
| Recall | 0.7539 |
| Accuracy | 0.9784 |

## 使用方法

### 安装依赖

```bash
pip install torch>=2.1.0 transformers>=4.37.0 pillow opencv-python
```

### 基础推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "qimingfan10/sa2va-vessel-hf",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "qimingfan10/sa2va-vessel-hf",
    trust_remote_code=True
)

model.eval()

# 准备图像
image = Image.open("your_oct_image.jpg").convert('RGB')

# 构建prompt
text = "<image>Please segment the blood vessel."

# 推理
with torch.no_grad():
    result = model.predict_forward(
        image=image,
        text=text,
        tokenizer=tokenizer,
        processor=None
    )

# 获取分割掩码
prediction_masks = result['prediction_masks']
pred_mask = prediction_masks[0][0]  # (H, W)

# 保存结果
import cv2
import numpy as np

if isinstance(pred_mask, torch.Tensor):
    pred_mask = pred_mask.cpu().numpy()

# 二值化
pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255

# 保存
cv2.imwrite("segmentation_result.png", pred_mask_binary)
```

### 高级用法：自定义prompt

```python
# 细化分割
text = "<image>Please segment the retinal arteries only."

# 多区域分割
text = "<image>Segment the blood vessel. [SEG] Also segment the optic disc. [SEG]"

# 排除区域
text = "<image>Segment vessels but exclude the fovea region."
```

## 训练详情

### 训练配置

- **优化器**: AdamW
- **学习率**: 2e-5 (warmup 366步)
- **Batch Size**: 16 (4 per GPU × 4 GPUs)
- **Gradient Accumulation**: 2
- **Effective Batch Size**: 32
- **迭代次数**: 12,192
- **训练时间**: ~72小时
- **硬件**: 4× NVIDIA RTX A6000 (48GB)

### 损失函数

```
L_total = L_BCE + L_Dice + 0.5 × L_Language

- L_BCE: Binary Cross-Entropy (像素级监督)
- L_Dice: Dice Loss (处理类别不平衡)
- L_Language: Cross-Entropy (保持语言理解能力)
```

### 参数冻结策略

- Vision Encoder (InternViT-6B): 完全冻结
- LLM前30层: 冻结
- LLM后10层: 部分微调
- Projector (2-layer MLP): 从头训练
- SAM2 Decoder: 完全微调

可训练参数约占15% (~1.2B / 8B)

## 数据集

- **来源**: 多中心临床采集的OCT图像
- **图像数量**: 9,346张
- **分辨率**: 512×512 - 1024×1024
- **标注方式**: 专业医师手工标注多边形mask
- **划分**: 训练90% / 验证5% / 测试5%

## 局限性

1. **计算成本**: 需要高性能GPU（推荐24GB+ VRAM）
2. **推理速度**: 约4.5秒/帧（单GPU），不适合实时应用
3. **数据依赖**: 需要大量标注数据进行训练
4. **泛化能力**: 主要在OCT数据上训练，跨模态泛化有限

## 伦理考量

- 本模型仅用于研究目的，不应直接用于临床诊断
- 医学图像分割结果应由专业医师审核
- 模型可能在未见过的数据分布上表现不佳

## 引用

如果您使用此模型，请引用：

```bibtex
@misc{sa2va2024,
  title={Sa2VA: Segment Anything to Vessel Analysis},
  author={Qiming Fan},
  year={2024},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/qimingfan10/sa2va-vessel-hf}}
}
```

## 许可证

Apache 2.0

## 相关资源

- 📄 [方法论文档](https://github.com/qimingfan10/RLSa2va/blob/main/SA2VA_METHODOLOGY.md)
- 🔧 [技术细节](https://github.com/qimingfan10/RLSa2va/blob/main/SA2VA_TECHNICAL_DETAILS.md)
- 🐙 [GitHub仓库](https://github.com/qimingfan10/RLSa2va)
- 📚 [完整文档](https://github.com/qimingfan10/RLSa2va/blob/main/DOCUMENTATION_INDEX.md)

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: https://github.com/qimingfan10/RLSa2va/issues
- HuggingFace Discussions: https://huggingface.co/qimingfan10/sa2va-vessel-hf/discussions
