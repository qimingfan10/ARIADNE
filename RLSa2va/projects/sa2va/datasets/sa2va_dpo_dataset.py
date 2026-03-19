"""
Sa2VA DPO Dataset
用于DPO训练的数据集，加载chosen和rejected mask对
"""

import os
import random
from typing import Literal, Optional, List
import torch
import numpy as np
from PIL import Image

import mmengine
from .base import Sa2VABaseDataset
from .common import SEG_QUESTIONS


class Sa2VADPODataset(Sa2VABaseDataset):
    """DPO训练数据集，加载chosen/rejected mask对"""
    
    def __init__(self,
                 data_root,
                 ann_file=None,
                 special_tokens=None,
                 prompt_template=None,
                 extra_image_processor=None,
                 tokenizer=None,
                 max_length=2048,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 repeats: int = 1,
                 name: str = 'DPODataset',
                 **kwargs):
        
        # Initialize Sa2VABaseDataset
        Sa2VABaseDataset.__init__(self,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name
        )
        
        self.data_root = data_root
        self.ann_file = os.path.join(data_root, ann_file)
        self.begin_str = '<image>\n'
        
        # Load annotations
        self.data_list = self.load_data_list()
    
    def load_data_list(self) -> List[dict]:
        """Load DPO annotations"""
        annotations = mmengine.load(self.ann_file, file_format='json')
        
        data_list = []
        for item in annotations:
            if 'chosen_mask' not in item or 'rejected_mask' not in item:
                continue
            
            data_info = {
                'image': os.path.join(self.data_root, item['image']),
                'chosen_mask': os.path.join(self.data_root, item['chosen_mask']),
                'rejected_mask': os.path.join(self.data_root, item['rejected_mask']),
                'prompt': item.get('prompt', '<image>Please segment the blood vessels.'),
            }
            data_list.append(data_info)
        
        print(f"Sa2VADPODataset: Loaded {len(data_list)} DPO pairs from {self.ann_file}")
        return data_list
    
    def __len__(self):
        return len(self.data_list) * self.repeats
    
    @property
    def modality_length(self):
        return [100 for _ in range(len(self))]
    
    def _load_mask(self, mask_path):
        """Load mask as tensor"""
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask) > 127
        return torch.from_numpy(mask.astype(np.float32))
    
    def __getitem__(self, index):
        """Get DPO training sample"""
        index = index % len(self.data_list)
        item = self.data_list[index]
        
        # Load image
        image = self._read_image(item['image'])
        if image is None:
            return self.__getitem__((index + 1) % len(self.data_list))
        
        width, height = image.size
        
        # Load masks
        try:
            chosen_mask = self._load_mask(item['chosen_mask'])
            rejected_mask = self._load_mask(item['rejected_mask'])
        except Exception as e:
            print(f"Error loading masks: {e}")
            return self.__getitem__((index + 1) % len(self.data_list))
        
        # Build prompt - use standard segmentation prompt
        question = random.choice(SEG_QUESTIONS)
        prompt = self.begin_str + question
        answer = '[SEG]'
        
        # Process with tokenizer
        result = self._process_single_sample(
            image=image,
            prompt=prompt,
            answer=answer,
            masks=[chosen_mask.numpy()],  # Use chosen mask as primary
            phrases=['blood vessels'],
            width=width,
            height=height,
        )
        
        if result is None:
            return self.__getitem__((index + 1) % len(self.data_list))
        
        # Add DPO-specific data
        result['chosen_mask'] = chosen_mask
        result['rejected_mask'] = rejected_mask
        result['is_dpo'] = True
        
        return result
