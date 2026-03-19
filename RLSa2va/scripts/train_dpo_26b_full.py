#!/usr/bin/env python3
"""
Sa2VA 26B DPO训练 - 完整版
基于已有微调好的26B模型 (Dice 0.82) 进行DPO训练

使用简化的DPO损失：直接比较分割mask的质量
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
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import cv2

sys.path.insert(0, '/home/ubuntu/Sa2VA')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader


@dataclass
class DPOConfig:
    """DPO训练配置"""
    # 模型
    model_path: str = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
    output_dir: str = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_vessel"
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # 训练
    learning_rate: float = 2e-6
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    
    # DPO
    beta: float = 0.1  # DPO温度
    
    # 数据
    data_path: str = "/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_annotations.json"
    max_samples: Optional[int] = 500  # 限制样本数


class DPOVesselDataset(Dataset):
    """DPO血管分割数据集"""
    
    def __init__(self, annotations_path: str, data_root: str, max_samples: Optional[int] = None):
        with open(annotations_path) as f:
            self.annotations = json.load(f)
        
        if max_samples:
            self.annotations = self.annotations[:max_samples]
        
        self.data_root = data_root
        
        print(f"Loaded {len(self.annotations)} DPO samples")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 加载图片
        img_path = os.path.join(self.data_root, ann['image'])
        image = Image.open(img_path).convert('RGB')
        
        # 加载chosen和rejected masks
        chosen_path = os.path.join(self.data_root, ann['chosen_mask'])
        rejected_path = os.path.join(self.data_root, ann['rejected_mask'])
        
        chosen_mask = np.array(Image.open(chosen_path).convert('L'))
        rejected_mask = np.array(Image.open(rejected_path).convert('L'))
        
        # 归一化到0-1
        chosen_mask = (chosen_mask > 127).astype(np.float32)
        rejected_mask = (rejected_mask > 127).astype(np.float32)
        
        return {
            'image': image,
            'chosen_mask': torch.from_numpy(chosen_mask),
            'rejected_mask': torch.from_numpy(rejected_mask),
            'chosen_iou': ann['chosen_iou'],
            'rejected_iou': ann['rejected_iou'],
            'prompt': ann.get('prompt', '<image>Please segment the blood vessels.'),
        }


def compute_mask_log_prob(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    计算预测mask和目标mask之间的log概率
    
    使用Binary Cross Entropy的负值作为log probability:
    log p(target|pred) = target * log(pred) + (1-target) * log(1-pred)
    """
    eps = 1e-7
    pred_mask = pred_mask.clamp(eps, 1 - eps)
    
    log_prob = target_mask * torch.log(pred_mask) + (1 - target_mask) * torch.log(1 - pred_mask)
    return log_prob.mean()


def compute_dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """计算Dice分数"""
    pred = (pred > 0.5).astype(float)
    target = (target > 0.5).astype(float)
    intersection = (pred * target).sum()
    return 2 * intersection / (pred.sum() + target.sum() + 1e-8)


class DPOTrainer:
    """DPO训练器"""
    
    def __init__(self, config: DPOConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("=" * 60)
        print("🎯 Sa2VA 26B DPO Trainer")
        print("=" * 60)
        
        # 加载模型
        self._load_model()
        
        # 设置数据
        self._setup_data()
        
        # 设置优化器
        self._setup_optimizer()
    
    def _load_model(self):
        """加载模型并应用LoRA"""
        print("\n📥 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        
        print("✅ Model loaded!")
        
        # 应用LoRA
        print("\n🔧 Applying LoRA...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        )
        
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        
        # 冻结其他部分
        self.model.vision_model.requires_grad_(False)
        if hasattr(self.model, 'sam2'):
            self.model.sam2.requires_grad_(False)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ LoRA applied! Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    
    def _setup_data(self):
        """设置数据"""
        print("\n📊 Loading data...")
        data_root = os.path.dirname(self.config.data_path)
        self.dataset = DPOVesselDataset(
            self.config.data_path, 
            data_root, 
            self.config.max_samples
        )
    
    def _setup_optimizer(self):
        """设置优化器"""
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        
        # 学习率调度器
        total_steps = len(self.dataset) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-7
        )
    
    def get_model_prediction(self, image: Image.Image, prompt: str) -> Optional[np.ndarray]:
        """获取模型预测的mask"""
        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                result = self.model.predict_forward(
                    image=image,
                    text=prompt,
                    tokenizer=self.tokenizer,
                )
            
            if result.get('prediction_masks'):
                pred_mask = result['prediction_masks'][0][0]
                if isinstance(pred_mask, torch.Tensor):
                    pred_mask = pred_mask.cpu().numpy()
                return pred_mask
        except Exception as e:
            print(f"  Prediction error: {e}")
        
        return None
    
    def compute_dpo_loss(
        self, 
        pred_mask: torch.Tensor, 
        chosen_mask: torch.Tensor, 
        rejected_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算DPO损失
        
        DPO Loss = -log σ(β * (log π(chosen) - log π(rejected)))
        
        对于分割任务，log π(mask) 表示模型生成这个mask的log概率
        我们用预测mask和目标mask之间的相似度来近似
        """
        # 确保在同一设备上
        pred_mask = pred_mask.float()
        chosen_mask = chosen_mask.float()
        rejected_mask = rejected_mask.float()
        
        # 调整尺寸
        if pred_mask.shape != chosen_mask.shape:
            h, w = chosen_mask.shape
            pred_mask = F.interpolate(
                pred_mask.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # 计算与chosen和rejected的log概率
        # 这里使用负BCE作为log probability的代理
        chosen_log_prob = compute_mask_log_prob(pred_mask, chosen_mask)
        rejected_log_prob = compute_mask_log_prob(pred_mask, rejected_mask)
        
        # DPO Loss
        logits = self.config.beta * (chosen_log_prob - rejected_log_prob)
        loss = -F.logsigmoid(logits)
        
        # 计算指标
        with torch.no_grad():
            accuracy = (logits > 0).float()
            margin = (chosen_log_prob - rejected_log_prob).item()
        
        metrics = {
            'loss': loss.item(),
            'chosen_log_prob': chosen_log_prob.item(),
            'rejected_log_prob': rejected_log_prob.item(),
            'margin': margin,
            'accuracy': accuracy.item(),
        }
        
        return loss, metrics
    
    def train(self):
        """训练循环"""
        print("\n🚀 Starting DPO training...")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.model.train()
        global_step = 0
        total_loss = 0
        accumulated_loss = 0
        
        all_metrics = []
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            pbar = tqdm(range(len(self.dataset)), desc=f"Training")
            
            for idx in pbar:
                sample = self.dataset[idx]
                
                try:
                    # 获取模型预测
                    pred_mask = self.get_model_prediction(sample['image'], sample['prompt'])
                    
                    if pred_mask is None:
                        continue
                    
                    # 转换为tensor
                    pred_tensor = torch.from_numpy(pred_mask).float()
                    
                    # 确定设备
                    device = next(self.model.parameters()).device
                    pred_tensor = pred_tensor.to(device)
                    chosen_tensor = sample['chosen_mask'].to(device)
                    rejected_tensor = sample['rejected_mask'].to(device)
                    
                    # 计算DPO损失
                    loss, metrics = self.compute_dpo_loss(
                        pred_tensor, 
                        chosen_tensor, 
                        rejected_tensor
                    )
                    
                    # 反向传播
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    accumulated_loss += loss.item()
                    
                    # 梯度累积
                    if (idx + 1) % self.config.gradient_accumulation_steps == 0:
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.config.max_grad_norm
                        )
                        
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        global_step += 1
                        total_loss += accumulated_loss
                        
                        # 更新进度条
                        pbar.set_postfix({
                            'loss': f'{accumulated_loss:.4f}',
                            'margin': f'{metrics["margin"]:.4f}',
                            'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                        })
                        
                        all_metrics.append(metrics)
                        accumulated_loss = 0
                    
                    # 定期保存
                    if global_step > 0 and global_step % 100 == 0:
                        self._save_checkpoint(global_step)
                
                except Exception as e:
                    print(f"\n  Error at step {idx}: {e}")
                    continue
        
        # 最终保存
        self._save_checkpoint(global_step, final=True)
        
        # 打印总结
        print("\n" + "=" * 60)
        print("🎉 Training completed!")
        print("=" * 60)
        print(f"Total steps: {global_step}")
        print(f"Average loss: {total_loss / max(global_step, 1):.4f}")
        print(f"Model saved to: {self.config.output_dir}")
    
    def _save_checkpoint(self, step: int, final: bool = False):
        """保存检查点"""
        save_dir = self.config.output_dir if final else os.path.join(self.config.output_dir, f'step_{step}')
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 Saving checkpoint to {save_dir}...")
        
        # 合并LoRA权重
        self.model.language_model = self.model.language_model.merge_and_unload()
        
        # 保存模型
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # 重新应用LoRA（如果不是最终保存）
        if not final:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias='none',
                task_type=TaskType.CAUSAL_LM,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            )
            self.model.language_model = get_peft_model(self.model.language_model, lora_config)


def main():
    config = DPOConfig()
    
    trainer = DPOTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
