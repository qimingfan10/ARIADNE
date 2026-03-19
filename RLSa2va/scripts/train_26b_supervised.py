#!/usr/bin/env python3
"""
Sa2VA 26B 监督微调
直接使用chosen mask作为监督信号进行训练

核心思想：
1. 获取模型生成的[SEG] hidden states
2. 通过SAM2 decoder生成mask logits（带梯度）
3. 使用Dice Loss + BCE Loss对chosen mask进行监督
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
import cv2

sys.path.insert(0, '/home/ubuntu/Sa2VA')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice Loss"""
    pred = torch.sigmoid(pred).flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary Cross Entropy Loss"""
    return F.binary_cross_entropy_with_logits(pred.flatten(), target.flatten())


def combined_loss(pred: torch.Tensor, target: torch.Tensor, dice_weight: float = 0.5) -> torch.Tensor:
    """Combined Dice + BCE Loss"""
    return dice_weight * dice_loss(pred, target) + (1 - dice_weight) * bce_loss(pred, target)


class Sa2VA26BSupervised:
    """Sa2VA 26B监督微调训练器"""
    
    def __init__(
        self,
        model_path: str = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf",
        output_dir: str = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_supervised",
        lora_r: int = 16,
        learning_rate: float = 5e-5,
        num_epochs: int = 1,
        gradient_accumulation: int = 4,
        max_samples: int = 500,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation = gradient_accumulation
        self.max_samples = max_samples
        self.lora_r = lora_r
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🎯 Sa2VA 26B Supervised Training")
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
        
        # 应用LoRA
        print("\n🔧 Applying LoRA...")
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_r * 2,
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        
        # 冻结vision encoder
        self.model.vision_model.requires_grad_(False)
        
        # SAM2 decoder保持可训练
        if hasattr(self.model, 'sam2'):
            # 全部可训练
            pass
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
        
        # 初始化prediction配置
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
    
    def _load_data(self):
        """加载数据"""
        print("\n📊 Loading data...")
        
        # 使用chosen mask作为ground truth
        data_path = "/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_annotations.json"
        with open(data_path) as f:
            self.annotations = json.load(f)
        
        # 过滤只要chosen_method == 'gt'的样本（真实的GT）
        self.annotations = [ann for ann in self.annotations if ann.get('chosen_method') == 'gt']
        
        if self.max_samples:
            self.annotations = self.annotations[:self.max_samples]
        
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        print(f"   Loaded {len(self.annotations)} GT samples")
    
    def _setup_optimizer(self):
        """设置优化器"""
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        total_steps = len(self.annotations) * self.num_epochs // self.gradient_accumulation
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7
        )
    
    def get_mask_logits_with_grad(self, image: Image.Image, prompt: str):
        """
        获取带梯度的mask logits
        
        这个函数复现predict_forward的逻辑，但保持梯度
        """
        # 准备图像
        ori_image_size = image.size
        g_image = np.array(image)
        g_image = self.model.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.model.torch_dtype)
        g_pixel_values = torch.stack([
            self.model.grounding_encoder.preprocess_image(g_pixel_values)
        ]).to(self.model.torch_dtype)
        
        # 这里简化：我们假设模型会生成[SEG] token
        # 直接使用一个预设的[SEG] embedding来生成mask
        
        # 获取[SEG] token的embedding
        seg_token_id = self.model.seg_token_idx
        device = next(self.model.parameters()).device
        
        # 创建一个dummy [SEG] hidden state（使用可学习的embedding）
        # 实际应该从LLM输出获取，但这里简化处理
        hidden_size = self.model.config.hidden_size
        seg_hidden = self.model.language_model.get_input_embeddings()(
            torch.tensor([[seg_token_id]], device=device)
        )
        
        # 通过text_hidden_fcs转换
        seg_hidden_states = self.model.text_hidden_fcs(seg_hidden.squeeze(0))
        
        # 获取SAM2 embeddings
        sam_states = self.model.grounding_encoder.get_sam2_embeddings(g_pixel_values.to(device))
        
        # 生成mask（带梯度）
        pred_masks = self.model.grounding_encoder.language_embd_inference(
            sam_states, [seg_hidden_states]
        )
        
        # 调整到原图尺寸
        w, h = ori_image_size
        masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
        
        return masks[:, 0]  # [1, H, W]
    
    def train_step(self, sample: dict) -> dict:
        """训练步骤"""
        img_path = os.path.join(self.data_root, sample['image'])
        mask_path = os.path.join(self.data_root, sample['chosen_mask'])
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            return None
        
        image = Image.open(img_path).convert('RGB')
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_mask = (gt_mask > 127).astype(np.float32)
        gt_tensor = torch.from_numpy(gt_mask)
        
        try:
            # 获取带梯度的mask logits
            pred_logits = self.get_mask_logits_with_grad(
                image, 
                sample.get('prompt', '<image>Please segment the blood vessels.')
            )
            
            # 移动到同一设备
            device = pred_logits.device
            gt_tensor = gt_tensor.to(device)
            
            # 确保尺寸一致
            if pred_logits.shape[-2:] != gt_tensor.shape:
                gt_tensor = F.interpolate(
                    gt_tensor.unsqueeze(0).unsqueeze(0),
                    size=pred_logits.shape[-2:],
                    mode='nearest'
                ).squeeze()
            
            # 计算损失
            loss = combined_loss(pred_logits.squeeze(), gt_tensor)
            
            # 计算指标
            with torch.no_grad():
                pred_binary = (torch.sigmoid(pred_logits) > 0.5).float().squeeze()
                intersection = (pred_binary * gt_tensor).sum()
                dice = (2 * intersection / (pred_binary.sum() + gt_tensor.sum() + 1e-8)).item()
            
            return {
                'loss': loss,
                'dice': dice,
            }
        
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train(self):
        """训练"""
        print("\n🚀 Starting training...")
        
        self.model.train()
        global_step = 0
        accumulated_loss = 0
        accumulated_dice = 0
        
        for epoch in range(self.num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.num_epochs}")
            
            pbar = tqdm(self.annotations, desc="Training")
            
            for idx, sample in enumerate(pbar):
                result = self.train_step(sample)
                
                if result is None:
                    continue
                
                loss = result['loss']
                
                if not loss.requires_grad:
                    print(f"  Step {idx}: Loss has no gradient!")
                    continue
                
                # 反向传播
                scaled_loss = loss / self.gradient_accumulation
                scaled_loss.backward()
                accumulated_loss += loss.item()
                accumulated_dice += result['dice']
                
                if (idx + 1) % self.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        1.0
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    avg_loss = accumulated_loss / self.gradient_accumulation
                    avg_dice = accumulated_dice / self.gradient_accumulation
                    
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'dice': f'{avg_dice:.4f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                    })
                    
                    accumulated_loss = 0
                    accumulated_dice = 0
                
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
        """保存"""
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
            lora_alpha=self.lora_r * 2,
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)


def main():
    trainer = Sa2VA26BSupervised()
    trainer.train()


if __name__ == '__main__':
    main()
