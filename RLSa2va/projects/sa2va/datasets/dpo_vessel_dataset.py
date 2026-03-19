"""
DPO (Direct Preference Optimization) 数据集
用于血管分割的偏好学习

数据格式：
{
    "image": "path/to/image.jpg",
    "chosen_mask": "path/to/better_mask.png",      # IoU更高的mask (胜者)
    "rejected_mask": "path/to/worse_mask.png",     # IoU更低的mask (败者)
    "chosen_iou": 0.85,
    "rejected_iou": 0.62,
    "prompt": "<image>Please segment the blood vessels."
}
"""

import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from xtuner.utils import PROMPT_TEMPLATE


class DPOVesselDataset(Dataset):
    """DPO偏好对数据集
    
    每个样本包含：
    - 同一张图像
    - chosen: IoU更高的mask（胜者）
    - rejected: IoU更低的mask（败者）
    """
    
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        tokenizer: dict,
        prompt_template: dict,
        extra_image_processor: dict = None,
        max_length: int = 4096,
        min_iou_gap: float = 0.05,  # 最小IoU差距阈值
        **kwargs
    ):
        self.data_root = data_root
        self.max_length = max_length
        self.min_iou_gap = min_iou_gap
        
        # 加载annotations
        ann_path = os.path.join(data_root, ann_file)
        with open(ann_path, 'r') as f:
            self.annotations = json.load(f)
        
        # 过滤有效的偏好对
        self.samples = self._filter_valid_pairs()
        
        # 初始化tokenizer
        if isinstance(tokenizer, dict):
            self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer)
        else:
            self.tokenizer = tokenizer
            
        self.prompt_template = prompt_template
        self.extra_image_processor = extra_image_processor
        
        print(f"✅ DPO数据集加载完成:")
        print(f"   - 总样本数: {len(self.samples)}")
        print(f"   - 最小IoU差距: {min_iou_gap}")
        
    def _filter_valid_pairs(self) -> List[Dict]:
        """过滤有效的偏好对（IoU差距足够大）"""
        valid_pairs = []
        for item in self.annotations:
            iou_gap = item.get('chosen_iou', 0) - item.get('rejected_iou', 0)
            if iou_gap >= self.min_iou_gap:
                valid_pairs.append(item)
        return valid_pairs
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image_path = os.path.join(self.data_root, sample['image'])
        image = Image.open(image_path).convert('RGB')
        
        # 加载chosen mask (胜者)
        chosen_mask_path = os.path.join(self.data_root, sample['chosen_mask'])
        chosen_mask = np.array(Image.open(chosen_mask_path).convert('L'))
        chosen_mask = (chosen_mask > 127).astype(np.float32)
        
        # 加载rejected mask (败者)
        rejected_mask_path = os.path.join(self.data_root, sample['rejected_mask'])
        rejected_mask = np.array(Image.open(rejected_mask_path).convert('L'))
        rejected_mask = (rejected_mask > 127).astype(np.float32)
        
        # 构建prompt
        prompt = sample.get('prompt', '<image>Please segment the blood vessels.')
        
        # 图像预处理
        if self.extra_image_processor:
            image = self.extra_image_processor(image)
        
        return {
            'image': image,
            'chosen_mask': torch.from_numpy(chosen_mask),
            'rejected_mask': torch.from_numpy(rejected_mask),
            'chosen_iou': sample.get('chosen_iou', 0.0),
            'rejected_iou': sample.get('rejected_iou', 0.0),
            'prompt': prompt,
            'image_path': image_path
        }


def dpo_collect_fn(batch):
    """DPO数据集的collate函数"""
    images = [item['image'] for item in batch]
    chosen_masks = torch.stack([item['chosen_mask'] for item in batch])
    rejected_masks = torch.stack([item['rejected_mask'] for item in batch])
    chosen_ious = torch.tensor([item['chosen_iou'] for item in batch])
    rejected_ious = torch.tensor([item['rejected_iou'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    
    return {
        'images': images,
        'chosen_masks': chosen_masks,
        'rejected_masks': rejected_masks,
        'chosen_ious': chosen_ious,
        'rejected_ious': rejected_ious,
        'prompts': prompts
    }
