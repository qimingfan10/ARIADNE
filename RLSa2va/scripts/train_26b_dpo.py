#!/usr/bin/env python3
"""
Sa2VA 26B DPO训练
Direct Preference Optimization - 使用chosen/rejected pairs进行偏好学习

DPO Loss = -log σ(β * (log π(chosen) - log π(rejected)))

其中 log π(mask) 使用预测mask与目标mask的负Dice距离作为代理
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


def compute_mask_log_prob(pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    计算预测mask与目标mask的log概率（使用负Dice距离作为代理）
    
    log π(mask) ≈ -DiceLoss(pred, target)
    
    Dice越高（越相似），log概率越高
    """
    pred_prob = torch.sigmoid(pred_logits).flatten()
    target = target_mask.flatten()
    
    intersection = (pred_prob * target).sum()
    dice = (2. * intersection + 1.0) / (pred_prob.sum() + target.sum() + 1.0)
    
    # log概率 = log(dice) ，使用dice作为概率的代理
    # 为了数值稳定性，使用 log(dice + eps)
    log_prob = torch.log(dice + 1e-8)
    
    return log_prob


def dpo_loss(
    pred_logits: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 0.1
) -> tuple:
    """
    DPO损失函数
    
    L = -log σ(β * (log π(chosen) - log π(rejected)))
    
    Args:
        pred_logits: 模型预测的mask logits
        chosen_mask: 优选的mask（GT）
        rejected_mask: 劣选的mask（错误预测）
        beta: 温度参数，控制偏好的强度
    
    Returns:
        loss: DPO损失
        metrics: 包含各项指标的字典
    """
    # 计算log概率
    log_prob_chosen = compute_mask_log_prob(pred_logits, chosen_mask)
    log_prob_rejected = compute_mask_log_prob(pred_logits, rejected_mask)
    
    # DPO loss
    logits = beta * (log_prob_chosen - log_prob_rejected)
    loss = -F.logsigmoid(logits)
    
    # 计算指标
    with torch.no_grad():
        pred_binary = (torch.sigmoid(pred_logits) > 0.5).float().flatten()
        chosen_flat = chosen_mask.flatten()
        rejected_flat = rejected_mask.flatten()
        
        # Dice with chosen
        chosen_inter = (pred_binary * chosen_flat).sum()
        chosen_dice = (2 * chosen_inter / (pred_binary.sum() + chosen_flat.sum() + 1e-8)).item()
        
        # Dice with rejected
        rejected_inter = (pred_binary * rejected_flat).sum()
        rejected_dice = (2 * rejected_inter / (pred_binary.sum() + rejected_flat.sum() + 1e-8)).item()
        
        # 偏好准确率：模型是否更偏好chosen
        prefer_chosen = (log_prob_chosen > log_prob_rejected).float().item()
    
    metrics = {
        'chosen_dice': chosen_dice,
        'rejected_dice': rejected_dice,
        'log_prob_chosen': log_prob_chosen.item(),
        'log_prob_rejected': log_prob_rejected.item(),
        'prefer_chosen': prefer_chosen,
        'margin': (log_prob_chosen - log_prob_rejected).item(),
    }
    
    return loss, metrics


def forward_sam_with_grad(model, g_pixel_values, language_embd, ori_size):
    """
    带梯度的SAM2前向传播
    直接调用_forward_sam_heads，绕过inference_mode
    """
    sam2 = model.grounding_encoder
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # 获取backbone特征
        image_features = sam2.sam2_model.forward_image(g_pixel_values)
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


class Sa2VA_DPO_Trainer:
    """Sa2VA 26B DPO训练器"""
    
    def __init__(
        self,
        model_path="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf",
        output_dir="/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo",
        learning_rate=2e-5,
        lora_r=16,
        beta=0.1,  # DPO温度参数
        max_samples=500,
        gradient_accumulation=4,
        num_epochs=1,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.lora_r = lora_r
        self.beta = beta
        self.max_samples = max_samples
        self.gradient_accumulation = gradient_accumulation
        self.num_epochs = num_epochs
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🎯 Sa2VA 26B DPO Training")
        print(f"   β = {beta}")
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
        
        # 应用LoRA到language_model
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
        
        # SAM2 grounding encoder保持可训练
        for param in self.model.grounding_encoder.parameters():
            param.requires_grad = True
        
        # text_hidden_fcs保持可训练
        for param in self.model.text_hidden_fcs.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
        
        # 初始化
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
    
    def _load_data(self):
        print("\n📊 Loading DPO data (chosen + rejected pairs)...")
        
        data_path = "/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_annotations.json"
        with open(data_path) as f:
            self.annotations = json.load(f)
        
        # 确保有chosen和rejected mask
        valid_annotations = []
        for ann in self.annotations:
            if 'chosen_mask' in ann and 'rejected_mask' in ann:
                valid_annotations.append(ann)
        
        self.annotations = valid_annotations[:self.max_samples]
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        print(f"   Loaded {len(self.annotations)} preference pairs")
    
    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        total_steps = len(self.annotations) * self.num_epochs // self.gradient_accumulation
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7
        )
    
    def train_step(self, sample):
        """DPO训练步骤"""
        img_path = os.path.join(self.data_root, sample['image'])
        chosen_path = os.path.join(self.data_root, sample['chosen_mask'])
        rejected_path = os.path.join(self.data_root, sample['rejected_mask'])
        
        if not all(os.path.exists(p) for p in [img_path, chosen_path, rejected_path]):
            return None
        
        # 加载数据
        image = Image.open(img_path).convert('RGB')
        chosen_mask = (np.array(Image.open(chosen_path).convert('L')) > 127).astype(np.float32)
        rejected_mask = (np.array(Image.open(rejected_path).convert('L')) > 127).astype(np.float32)
        
        chosen_tensor = torch.from_numpy(chosen_mask)
        rejected_tensor = torch.from_numpy(rejected_mask)
        ori_size = chosen_mask.shape
        
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
            language_embd = self.model.text_hidden_fcs(seg_embedding)
            
            # 带梯度的SAM2前向
            pred_logits = forward_sam_with_grad(self.model, g_pixel_values, language_embd, ori_size)
            
            # 移动到同一设备
            chosen_tensor = chosen_tensor.to(pred_logits.device)
            rejected_tensor = rejected_tensor.to(pred_logits.device)
            
            # 计算DPO损失
            loss, metrics = dpo_loss(
                pred_logits.squeeze(),
                chosen_tensor,
                rejected_tensor,
                beta=self.beta
            )
            
            return {'loss': loss, **metrics}
        
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train(self):
        print("\n🚀 Starting DPO training...")
        print(f"   Loss = -log σ(β * (log π(chosen) - log π(rejected)))")
        print(f"   β = {self.beta}")
        
        self.model.train()
        global_step = 0
        
        for epoch in range(self.num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.num_epochs}")
            
            acc_loss = 0
            acc_chosen_dice = 0
            acc_rejected_dice = 0
            acc_prefer = 0
            acc_count = 0
            
            pbar = tqdm(self.annotations, desc="DPO Training")
            
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
                acc_chosen_dice += result['chosen_dice']
                acc_rejected_dice += result['rejected_dice']
                acc_prefer += result['prefer_chosen']
                acc_count += 1
                
                if (idx + 1) % self.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        1.0
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    if acc_count > 0:
                        pbar.set_postfix({
                            'loss': f'{acc_loss/acc_count:.4f}',
                            'chosen': f'{acc_chosen_dice/acc_count:.4f}',
                            'rejected': f'{acc_rejected_dice/acc_count:.4f}',
                            'prefer': f'{acc_prefer/acc_count:.2%}',
                        })
                    
                    acc_loss = 0
                    acc_chosen_dice = 0
                    acc_rejected_dice = 0
                    acc_prefer = 0
                    acc_count = 0
                
                # 定期保存
                if global_step > 0 and global_step % 30 == 0:
                    self._save(f'step_{global_step}')
        
        # 最终保存
        self._save('final')
        
        print("\n" + "=" * 60)
        print("🎉 DPO Training completed!")
        print(f"   Model saved to: {self.output_dir}")
        print("=" * 60)
    
    def _save(self, name):
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n💾 Saving to {save_dir}...")
        
        # 合并LoRA
        self.model.language_model = self.model.language_model.merge_and_unload()
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # 重新应用LoRA以继续训练
        lora_config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_r * 2, lora_dropout=0.05, bias='none',
            task_type=TaskType.CAUSAL_LM, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)


def main():
    trainer = Sa2VA_DPO_Trainer(
        beta=0.1,  # DPO温度参数
        max_samples=500,
        num_epochs=1,
    )
    trainer.train()


if __name__ == '__main__':
    main()
