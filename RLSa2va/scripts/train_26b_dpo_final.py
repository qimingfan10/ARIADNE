#!/usr/bin/env python3
"""
Sa2VA 26B DPO训练 Final版

关键改进：正确模拟predict_forward的流程，保留梯度
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
sys.path.insert(0, '/home/ubuntu/Sa2VA/models/sa2va_vessel_hf')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def dpo_loss(pred_logits, chosen_mask, rejected_mask, beta=0.1):
    """DPO损失函数"""
    pred_prob = torch.sigmoid(pred_logits).flatten()
    
    # Chosen mask log prob
    chosen_flat = chosen_mask.flatten()
    c_inter = (pred_prob * chosen_flat).sum()
    c_dice = (2. * c_inter + 1.0) / (pred_prob.sum() + chosen_flat.sum() + 1.0)
    log_prob_chosen = torch.log(c_dice + 1e-8)
    
    # Rejected mask log prob
    rejected_flat = rejected_mask.flatten()
    r_inter = (pred_prob * rejected_flat).sum()
    r_dice = (2. * r_inter + 1.0) / (pred_prob.sum() + rejected_flat.sum() + 1.0)
    log_prob_rejected = torch.log(r_dice + 1e-8)
    
    # DPO loss
    logits = beta * (log_prob_chosen - log_prob_rejected)
    loss = -F.logsigmoid(logits)
    
    with torch.no_grad():
        pred_binary = (pred_prob > 0.5).float()
        dice = (2 * (pred_binary * chosen_flat).sum() / (pred_binary.sum() + chosen_flat.sum() + 1e-8)).item()
        prefer = (log_prob_chosen > log_prob_rejected).float().item()
    
    return loss, {'chosen_dice': dice, 'prefer_chosen': prefer}


def forward_sam_with_grad(model, g_pixel_values, language_embd, ori_size):
    """带梯度的SAM2前向"""
    sam2 = model.grounding_encoder
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        image_features = sam2.sam2_model.forward_image(g_pixel_values)
        _, vision_feats, _, feat_sizes = sam2.sam2_model._prepare_backbone_features(image_features)
        
        B = vision_feats[-1].size(1)
        C = sam2.sam2_model.hidden_dim
        H, W = feat_sizes[-1]
        
        pix_feat = vision_feats[-1] + sam2.sam2_model.no_mem_embed
        pix_feat = pix_feat.permute(1, 2, 0).view(B, C, H, W)
        
        expected_size = sam2.sam2_model.sam_image_embedding_size
        if H != expected_size:
            pix_feat = F.interpolate(pix_feat, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
        
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
        
        _, _, _, low_res_masks, _, _, _ = sam2.sam2_model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=False,
            language_embd=language_embd,
        )
    
    h, w = ori_size
    masks = F.interpolate(low_res_masks, size=(h, w), mode='bilinear', align_corners=False)
    return masks.squeeze(1)


class DPOTrainer:
    def __init__(self):
        self.model_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
        self.output_dir = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_final"
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        self.beta = 0.1
        self.lr = 2e-5
        self.lora_r = 16
        self.max_samples = 500
        self.grad_accum = 4
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🎯 Sa2VA 26B DPO Training - Final Version")
        print("=" * 60)
        
        self._load_model()
        self._load_data()
        self._setup_optimizer()
    
    def _load_model(self):
        print("\n📥 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16,
            device_map='auto', trust_remote_code=True,
        )
        
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids('[SEG]')
        print(f"✅ Model loaded! [SEG] id: {self.seg_token_id}")
        
        # LoRA
        print("🔧 Applying LoRA...")
        lora_config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_r * 2, lora_dropout=0.05,
            bias='none', task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        
        # Freeze vision, train SAM2 & text_hidden_fcs
        self.model.vision_model.requires_grad_(False)
        for p in self.model.grounding_encoder.parameters():
            p.requires_grad = True
        for p in self.model.text_hidden_fcs.parameters():
            p.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _load_data(self):
        print("\n📊 Loading data...")
        with open(f"{self.data_root}/dpo_annotations.json") as f:
            data = json.load(f)
        self.annotations = [a for a in data if 'chosen_mask' in a and 'rejected_mask' in a][:self.max_samples]
        print(f"   {len(self.annotations)} samples")
    
    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=0.01,
        )
        total_steps = len(self.annotations) // self.grad_accum
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7
        )
    
    def _get_seg_embedding_simple(self, image):
        """简化版：获取[SEG] embedding"""
        device = next(self.model.parameters()).device
        
        # 获取[SEG] token的基础embedding
        seg_ids = torch.tensor([[self.seg_token_id]], device=device)
        seg_embed = self.model.language_model.get_input_embeddings()(seg_ids)  # [1, 1, hidden]
        
        # 通过text_hidden_fcs
        language_embd = self.model.text_hidden_fcs(seg_embed)
        
        return language_embd
    
    def train_step(self, sample):
        img_path = os.path.join(self.data_root, sample['image'])
        chosen_path = os.path.join(self.data_root, sample['chosen_mask'])
        rejected_path = os.path.join(self.data_root, sample['rejected_mask'])
        
        if not all(os.path.exists(p) for p in [img_path, chosen_path, rejected_path]):
            return None
        
        image = Image.open(img_path).convert('RGB')
        chosen_mask = (np.array(Image.open(chosen_path).convert('L')) > 127).astype(np.float32)
        rejected_mask = (np.array(Image.open(rejected_path).convert('L')) > 127).astype(np.float32)
        
        ori_size = chosen_mask.shape
        
        try:
            # 获取[SEG] embedding
            language_embd = self._get_seg_embedding_simple(image)
            
            # 准备SAM2输入
            device = language_embd.device
            g_image = np.array(image)
            g_image = self.model.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(torch.bfloat16)
            g_pixel_values = self.model.grounding_encoder.preprocess_image(g_pixel_values).unsqueeze(0).to(device)
            
            # SAM2前向
            pred_logits = forward_sam_with_grad(self.model, g_pixel_values, language_embd, ori_size)
            
            # 移动masks到相同设备
            pred_device = pred_logits.device
            chosen_tensor = torch.from_numpy(chosen_mask).to(pred_device)
            rejected_tensor = torch.from_numpy(rejected_mask).to(pred_device)
            
            # DPO loss
            loss, metrics = dpo_loss(pred_logits.squeeze(), chosen_tensor, rejected_tensor, beta=self.beta)
            
            return {'loss': loss, **metrics}
        
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train(self):
        print("\n🚀 Starting DPO training...")
        self.model.train()
        
        acc_loss, acc_dice, acc_prefer, acc_count = 0, 0, 0, 0
        global_step = 0
        
        pbar = tqdm(self.annotations, desc="DPO Training")
        
        for idx, sample in enumerate(pbar):
            result = self.train_step(sample)
            
            if result is None or not result['loss'].requires_grad:
                continue
            
            loss = result['loss'] / self.grad_accum
            loss.backward()
            
            acc_loss += result['loss'].item()
            acc_dice += result['chosen_dice']
            acc_prefer += result['prefer_chosen']
            acc_count += 1
            
            if (idx + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
                
                if acc_count > 0:
                    pbar.set_postfix({
                        'loss': f'{acc_loss/acc_count:.4f}',
                        'dice': f'{acc_dice/acc_count:.4f}',
                        'prefer': f'{acc_prefer/acc_count:.1%}',
                    })
                acc_loss = acc_dice = acc_prefer = acc_count = 0
                
                if global_step % 50 == 0:
                    self._save(f'step_{global_step}')
        
        self._save('final')
        print("\n" + "=" * 60)
        print("🎉 Training completed!")
        print(f"   Saved to: {self.output_dir}")
        print("=" * 60)
    
    def _save(self, name):
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n💾 Saving to {save_dir}...")
        
        # Merge LoRA
        self.model.language_model = self.model.language_model.merge_and_unload()
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Re-apply LoRA
        lora_config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_r * 2, lora_dropout=0.05,
            bias='none', task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)


if __name__ == '__main__':
    trainer = DPOTrainer()
    trainer.train()
