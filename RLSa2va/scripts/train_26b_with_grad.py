#!/usr/bin/env python3
"""
Sa2VA 26B 带梯度训练
直接调用SAM2底层函数，绕过@torch.inference_mode()
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
    pred = torch.sigmoid(pred).flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def forward_sam_with_grad(model, g_pixel_values, language_embd, ori_size):
    """
    带梯度的SAM2前向传播
    直接调用_forward_sam_heads，绕过inference_mode
    """
    sam2 = model.grounding_encoder
    
    # 准备图像特征
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # 使用forward_image获取backbone特征
        image_features = sam2.sam2_model.forward_image(g_pixel_values)
        
        # 准备backbone特征
        _, vision_feats, vision_pos_embeds, feat_sizes = sam2.sam2_model._prepare_backbone_features(image_features)
        
        # 获取图像embedding
        B = vision_feats[-1].size(1)
        C = sam2.sam2_model.hidden_dim
        H, W = feat_sizes[-1]
        
        # 直接添加no_mem_embed
        pix_feat = vision_feats[-1] + sam2.sam2_model.no_mem_embed
        pix_feat = pix_feat.permute(1, 2, 0).view(B, C, H, W)
        
        # 调整尺寸
        expected_size = sam2.sam2_model.sam_image_embedding_size
        if H != expected_size or W != expected_size:
            pix_feat = F.interpolate(pix_feat, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
        
        # 准备high_res_features
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
        ]
        if H != expected_size:
            high_res_features = [
                F.interpolate(feat, size=(feat.size(2) * expected_size // H, feat.size(3) * expected_size // W), 
                              mode='bilinear', align_corners=False)
                for feat in high_res_features
            ]
        
        # 调用_forward_sam_heads（带梯度）
        _, _, _, low_res_masks, high_res_masks, obj_ptr, _ = sam2.sam2_model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=False,
            language_embd=language_embd,
        )
    
    # 调整到原图尺寸
    h, w = ori_size
    masks = F.interpolate(low_res_masks, size=(h, w), mode='bilinear', align_corners=False)
    return masks.squeeze(1)  # [B, H, W]


class Sa2VA26BTrainer:
    def __init__(
        self,
        model_path="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf",
        output_dir="/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_with_grad",
        learning_rate=5e-5,
        lora_r=16,
        max_samples=300,
        gradient_accumulation=4,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.lora_r = lora_r
        self.max_samples = max_samples
        self.gradient_accumulation = gradient_accumulation
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🎯 Sa2VA 26B Training with Gradients")
        print("=" * 60)
        
        self._load_model()
        self._load_data()
        self._setup_optimizer()
    
    def _load_model(self):
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
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        
        # 冻结vision encoder
        self.model.vision_model.requires_grad_(False)
        
        # SAM2保持可训练
        if hasattr(self.model, 'sam2'):
            for param in self.model.sam2.parameters():
                param.requires_grad = True
        
        # text_hidden_fcs保持可训练
        if hasattr(self.model, 'text_hidden_fcs'):
            for param in self.model.text_hidden_fcs.parameters():
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
        
        # 初始化
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
    
    def _load_data(self):
        print("\n📊 Loading data...")
        data_path = "/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_annotations.json"
        with open(data_path) as f:
            self.annotations = json.load(f)
        
        # 只用GT样本
        self.annotations = [a for a in self.annotations if a.get('chosen_method') == 'gt'][:self.max_samples]
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        print(f"   Loaded {len(self.annotations)} samples")
    
    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        total_steps = len(self.annotations) // self.gradient_accumulation
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7
        )
    
    def train_step(self, sample):
        img_path = os.path.join(self.data_root, sample['image'])
        mask_path = os.path.join(self.data_root, sample['chosen_mask'])
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            return None
        
        image = Image.open(img_path).convert('RGB')
        gt_mask = (np.array(Image.open(mask_path).convert('L')) > 127).astype(np.float32)
        gt_tensor = torch.from_numpy(gt_mask)
        ori_size = gt_mask.shape
        
        try:
            # 准备图像
            g_image = np.array(image)
            g_image = self.model.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(torch.bfloat16)
            
            device = next(self.model.parameters()).device
            g_pixel_values = self.model.grounding_encoder.preprocess_image(g_pixel_values).unsqueeze(0).to(device)
            
            # 获取[SEG] token embedding
            seg_token_id = self.model.seg_token_idx
            seg_embedding = self.model.language_model.get_input_embeddings()(
                torch.tensor([[seg_token_id]], device=device)
            )
            
            # 通过text_hidden_fcs
            language_embd = self.model.text_hidden_fcs(seg_embedding)  # [1, 1, hidden_dim]
            
            # 带梯度的SAM2前向
            pred_logits = forward_sam_with_grad(
                self.model, g_pixel_values, language_embd, ori_size
            )
            
            # 计算损失
            gt_tensor = gt_tensor.to(pred_logits.device)
            loss = dice_loss(pred_logits.squeeze(), gt_tensor)
            
            # 计算指标
            with torch.no_grad():
                pred_binary = (torch.sigmoid(pred_logits) > 0.5).float().squeeze()
                dice = (2 * (pred_binary * gt_tensor).sum() / (pred_binary.sum() + gt_tensor.sum() + 1e-8)).item()
            
            return {'loss': loss, 'dice': dice}
        
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train(self):
        print("\n🚀 Starting training...")
        
        self.model.train()
        global_step = 0
        acc_loss = 0
        acc_dice = 0
        
        pbar = tqdm(self.annotations, desc="Training")
        
        for idx, sample in enumerate(pbar):
            result = self.train_step(sample)
            
            if result is None:
                continue
            
            loss = result['loss']
            
            if not loss.requires_grad:
                print(f"  Step {idx}: No gradient!")
                continue
            
            scaled_loss = loss / self.gradient_accumulation
            scaled_loss.backward()
            acc_loss += loss.item()
            acc_dice += result['dice']
            
            if (idx + 1) % self.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    1.0
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                global_step += 1
                
                avg_loss = acc_loss / self.gradient_accumulation
                avg_dice = acc_dice / self.gradient_accumulation
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'dice': f'{avg_dice:.4f}',
                })
                
                acc_loss = 0
                acc_dice = 0
            
            if global_step > 0 and global_step % 30 == 0:
                self._save(f'step_{global_step}')
        
        self._save('final')
        print("\n✅ Training completed!")
        print(f"   Model saved to: {self.output_dir}")
    
    def _save(self, name):
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n💾 Saving to {save_dir}...")
        
        # 合并LoRA
        self.model.language_model = self.model.language_model.merge_and_unload()
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # 重新应用LoRA
        lora_config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_r * 2, lora_dropout=0.05, bias='none',
            task_type=TaskType.CAUSAL_LM, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)


def main():
    trainer = Sa2VA26BTrainer()
    trainer.train()


if __name__ == '__main__':
    main()
