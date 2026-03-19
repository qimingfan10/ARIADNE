#!/usr/bin/env python3
"""
Sa2VA 26B 对比学习训练
使用Chosen/Rejected pairs进行对比式优化

核心思想：
1. 对chosen mask计算正向Dice Loss（鼓励模型预测chosen）
2. 对rejected mask计算负向Dice Loss（惩罚模型预测rejected）
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/Sa2VA')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """计算Dice Loss"""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def contrastive_segmentation_loss(
    pred_mask: torch.Tensor, 
    chosen_mask: torch.Tensor, 
    rejected_mask: torch.Tensor,
    margin: float = 0.2
) -> torch.Tensor:
    """
    对比分割损失
    
    鼓励：pred接近chosen
    惩罚：pred接近rejected
    
    Loss = Dice(pred, chosen) + max(0, margin - Dice(pred, rejected))
    """
    # 确保尺寸一致
    if pred_mask.shape != chosen_mask.shape:
        h, w = chosen_mask.shape[-2:]
        pred_mask = F.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0) if pred_mask.dim() == 2 else pred_mask.unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze()
    
    # Sigmoid转换为概率
    pred_prob = torch.sigmoid(pred_mask)
    
    # Chosen loss: 最小化与chosen的差距
    chosen_loss = dice_loss(pred_prob, chosen_mask)
    
    # Rejected loss: 最大化与rejected的差距（使用margin）
    rejected_dice = 1 - dice_loss(pred_prob, rejected_mask)  # 相似度
    rejected_loss = F.relu(rejected_dice - margin)  # 如果相似度太高则惩罚
    
    return chosen_loss + 0.5 * rejected_loss


class Sa2VA26BTrainer:
    """Sa2VA 26B训练器"""
    
    def __init__(
        self,
        model_path: str = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf",
        output_dir: str = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_contrastive",
        lora_r: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 1,
        gradient_accumulation_steps: int = 8,
        max_samples: int = 500,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_samples = max_samples
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🎯 Sa2VA 26B Contrastive Training")
        print("=" * 60)
        
        self._load_model()
        self._load_data()
        self._setup_optimizer()
    
    def _load_model(self):
        """加载模型"""
        print("\n📥 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        
        print("✅ Model loaded!")
        
        # 应用LoRA到language_model
        print("\n🔧 Applying LoRA to language_model...")
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        )
        
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        
        # 冻结视觉编码器
        self.model.vision_model.requires_grad_(False)
        
        # 保持SAM2 decoder可训练（关键部分）
        if hasattr(self.model, 'sam2'):
            for name, param in self.model.sam2.named_parameters():
                # 只训练mask decoder的部分层
                if 'mask_decoder' in name or 'output_upscaling' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    
    def _load_data(self):
        """加载数据"""
        print("\n📊 Loading data...")
        
        data_path = "/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_annotations.json"
        data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        
        with open(data_path) as f:
            self.annotations = json.load(f)
        
        if self.max_samples:
            self.annotations = self.annotations[:self.max_samples]
        
        self.data_root = data_root
        print(f"   Loaded {len(self.annotations)} samples")
    
    def _setup_optimizer(self):
        """设置优化器"""
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        total_steps = len(self.annotations) * self.num_epochs // self.gradient_accumulation_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7
        )
    
    def train_step(self, sample: dict) -> dict:
        """单步训练"""
        # 加载数据
        img_path = os.path.join(self.data_root, sample['image'])
        chosen_path = os.path.join(self.data_root, sample['chosen_mask'])
        rejected_path = os.path.join(self.data_root, sample['rejected_mask'])
        
        if not all(os.path.exists(p) for p in [img_path, chosen_path, rejected_path]):
            return None
        
        image = Image.open(img_path).convert('RGB')
        chosen_mask = torch.from_numpy(
            (np.array(Image.open(chosen_path).convert('L')) > 127).astype(np.float32)
        )
        rejected_mask = torch.from_numpy(
            (np.array(Image.open(rejected_path).convert('L')) > 127).astype(np.float32)
        )
        
        prompt = sample.get('prompt', '<image>Please segment the blood vessels.')
        
        # 前向传播（需要梯度）
        try:
            # 使用模型的chat方法获取预测
            # 这里我们直接调用底层的predict_forward但启用梯度
            with torch.set_grad_enabled(True):
                result = self.model.predict_forward(
                    image=image,
                    text=prompt,
                    tokenizer=self.tokenizer,
                )
            
            if not result.get('prediction_masks'):
                return None
            
            pred_mask = result['prediction_masks'][0][0]
            if isinstance(pred_mask, torch.Tensor):
                pred_mask = pred_mask.float()
            else:
                pred_mask = torch.from_numpy(pred_mask).float()
            
            # 移动到正确的设备
            device = next(self.model.parameters()).device
            pred_mask = pred_mask.to(device)
            chosen_mask = chosen_mask.to(device)
            rejected_mask = rejected_mask.to(device)
            
            # 计算对比损失
            loss = contrastive_segmentation_loss(pred_mask, chosen_mask, rejected_mask)
            
            # 计算指标
            with torch.no_grad():
                pred_binary = (pred_mask > 0.5).float()
                chosen_dice = 1 - dice_loss(pred_binary, chosen_mask)
                rejected_dice = 1 - dice_loss(pred_binary, rejected_mask)
            
            return {
                'loss': loss,
                'chosen_dice': chosen_dice.item(),
                'rejected_dice': rejected_dice.item(),
            }
        
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train(self):
        """训练"""
        print("\n🚀 Starting training...")
        
        self.model.train()
        global_step = 0
        accumulated_loss = 0
        
        for epoch in range(self.num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.num_epochs}")
            
            pbar = tqdm(self.annotations, desc="Training")
            
            for idx, sample in enumerate(pbar):
                result = self.train_step(sample)
                
                if result is None:
                    continue
                
                loss = result['loss']
                
                # 检查loss是否需要梯度
                if not loss.requires_grad:
                    # 如果没有梯度，创建一个需要梯度的dummy loss
                    # 这种情况下我们只能通过其他方式训练
                    continue
                
                # 反向传播
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                
                if (idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        1.0
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    pbar.set_postfix({
                        'loss': f'{accumulated_loss:.4f}',
                        'chosen_dice': f'{result["chosen_dice"]:.4f}',
                        'rejected_dice': f'{result["rejected_dice"]:.4f}',
                    })
                    
                    accumulated_loss = 0
                
                # 定期保存
                if global_step > 0 and global_step % 50 == 0:
                    self._save(f'step_{global_step}')
        
        # 最终保存
        self._save('final')
        
        print("\n" + "=" * 60)
        print("🎉 Training completed!")
        print(f"   Model saved to: {self.output_dir}")
        print("=" * 60)
    
    def _save(self, name: str):
        """保存模型"""
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 Saving to {save_dir}...")
        
        # 合并LoRA
        self.model.language_model = self.model.language_model.merge_and_unload()
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # 重新应用LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)


def main():
    trainer = Sa2VA26BTrainer()
    trainer.train()


if __name__ == '__main__':
    main()
