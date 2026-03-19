#!/usr/bin/env python3
"""
Sa2VA 26B DPO V6 - 混合训练策略
根据新策略：DPO Loss + Dice Loss + 严格数据筛选

核心改动：
1. Chosen = Ground Truth Mask（不是模型预测）
2. Rejected = Baseline预测（需要IoU差距 > margin）
3. Loss = L_DPO + λ * L_Dice(pred, GT)
4. β = 0.2-0.3（防止偏离太远）
5. LR = 1e-6（更保守）
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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def compute_dice(pred, target):
    """计算Dice分数"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + 1e-8) / (pred_flat.sum() + target_flat.sum() + 1e-8)


def dice_loss(pred, target):
    """Dice Loss = 1 - Dice"""
    return 1 - compute_dice(pred, target)


def dpo_loss(log_prob_chosen, log_prob_rejected, beta=0.2):
    """
    DPO Loss: -log σ(β * (log π(chosen) - log π(rejected)))
    """
    logits = beta * (log_prob_chosen - log_prob_rejected)
    return -F.logsigmoid(logits)


class HybridDPOTrainer:
    """
    混合DPO训练器
    Loss = L_DPO + λ * L_Dice
    """
    
    def __init__(self):
        self.model_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
        self.output_dir = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v7"
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        
        # 新策略超参数
        self.beta = 0.2           # 增大beta，防止偏离太远
        self.dice_weight = 1.0    # Dice Loss权重
        self.lr = 1e-6            # 更保守的学习率
        self.lora_r = 16
        self.max_samples = 500
        self.grad_accum = 4
        self.save_steps = 100
        self.margin = 0.15        # Chosen与Rejected的IoU差距要求
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🚀 Hybrid DPO Training V6")
        print("=" * 60)
        print(f"   β = {self.beta}")
        print(f"   Dice weight = {self.dice_weight}")
        print(f"   LR = {self.lr}")
        print(f"   Margin = {self.margin}")
        print("=" * 60)
        
        self._load_model()
        self._load_and_filter_data()
        self._setup_training()
    
    def _load_model(self):
        print("\n📥 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
        self.seg_token_id = self.model.seg_token_idx
        
        print("🔧 Applying LoRA to LLM...")
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_r * 2,
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        
        # 冻结Vision Encoder（新策略要求）
        self.model.vision_model.requires_grad_(False)
        
        # 训练SAM2 Mask Decoder和text_hidden_fcs
        for name, p in self.model.grounding_encoder.named_parameters():
            # 只训练decoder相关部分
            if 'mask_decoder' in name or 'output_upscaling' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        for p in self.model.text_hidden_fcs.parameters():
            p.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _load_and_filter_data(self):
        """
        加载并筛选数据
        新策略：Chosen = GT，Rejected需要与GT有足够差距
        """
        print("\n📊 Loading and filtering data...")
        
        # 加载原始DPO注释
        ann_file = f"{self.data_root}/dpo_annotations.json"
        with open(ann_file) as f:
            annotations = json.load(f)
        
        # 筛选符合条件的样本
        self.filtered_data = []
        skipped = 0
        
        for item in annotations:
            if 'chosen_mask' not in item or 'rejected_mask' not in item:
                skipped += 1
                continue
            
            # 加载mask
            chosen_path = os.path.join(self.data_root, item['chosen_mask'])
            rejected_path = os.path.join(self.data_root, item['rejected_mask'])
            
            if not os.path.exists(chosen_path) or not os.path.exists(rejected_path):
                skipped += 1
                continue
            
            chosen = np.array(Image.open(chosen_path).convert('L')) > 127
            rejected = np.array(Image.open(rejected_path).convert('L')) > 127
            
            # 计算IoU差距
            chosen_dice = 1.0  # GT就是chosen
            
            # 计算rejected与chosen的Dice
            inter = (chosen & rejected).sum()
            rejected_dice = (2 * inter) / (chosen.sum() + rejected.sum() + 1e-8)
            
            # 筛选：差距需要大于margin
            dice_gap = chosen_dice - rejected_dice
            
            if dice_gap >= self.margin:
                self.filtered_data.append({
                    'image': os.path.join(self.data_root, item['image']),
                    'gt_mask': chosen_path,  # Chosen = GT
                    'rejected_mask': rejected_path,
                    'dice_gap': dice_gap,
                })
            else:
                skipped += 1
        
        self.filtered_data = self.filtered_data[:self.max_samples]
        print(f"   Total: {len(annotations)}, Filtered: {len(self.filtered_data)}, Skipped: {skipped}")
        print(f"   Avg Dice Gap: {np.mean([d['dice_gap'] for d in self.filtered_data]):.4f}")
    
    def _setup_training(self):
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=0.01,
        )
        
        total_steps = len(self.filtered_data) // self.grad_accum
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-8
        )
    
    def _get_seg_embedding(self):
        """获取[SEG] token embedding"""
        device = next(self.model.language_model.parameters()).device
        seg_emb = self.model.language_model.get_input_embeddings()(
            torch.tensor([self.seg_token_id], device=device)
        )
        seg_emb = self.model.text_hidden_fcs(seg_emb)
        return seg_emb
    
    def _predict_mask_with_grad(self, image, seg_embedding):
        """使用SAM2预测mask（带梯度）"""
        sam2 = self.model.grounding_encoder
        sam2_device = next(sam2.parameters()).device
        
        # 预处理图像
        g_image = np.array(image)
        g_image = self.model.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
        g_pixel_values = g_pixel_values.to(torch.bfloat16).to(sam2_device)
        g_pixel_values = sam2.preprocess_image(g_pixel_values).unsqueeze(0)
        
        seg_embedding = seg_embedding.to(sam2_device)
        if seg_embedding.dim() == 2:
            seg_embedding = seg_embedding.unsqueeze(1)
        
        # SAM2 forward（使用训练版本的方法）
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = sam2.sam2_model.forward_image(g_pixel_values)
            _, vision_feats, _, feat_sizes = sam2.sam2_model._prepare_backbone_features(feats)
            
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
            
            B = vision_feats[-1].size(1)
            C = sam2.sam2_model.hidden_dim
            H, W = feat_sizes[-1]
            
            pix_feat = vision_feats[-1] + sam2.sam2_model.no_mem_embed
            pix_feat = pix_feat.permute(1, 2, 0).view(B, C, H, W)
            
            expected_size = sam2.sam2_model.sam_image_embedding_size
            if H != expected_size:
                pix_feat = F.interpolate(pix_feat, size=(expected_size, expected_size), 
                                        mode='bilinear', align_corners=False)
                high_res_features = [
                    F.interpolate(feat, 
                                size=(feat.size(2) * expected_size // H, 
                                      feat.size(3) * expected_size // W),
                                mode='bilinear', align_corners=False)
                    for feat in high_res_features
                ]
            
            _, _, _, low_res_masks, _, _, _ = sam2.sam2_model._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=False,
                language_embd=seg_embedding,
            )
        
        h, w = image.size[::-1]
        masks = F.interpolate(low_res_masks, size=(h, w), mode='bilinear', align_corners=False)
        return masks.squeeze(1)
    
    def train_step(self, sample):
        """执行一个训练步骤"""
        if not os.path.exists(sample['image']):
            return None
        
        # 加载数据
        image = Image.open(sample['image']).convert('RGB')
        gt_mask = torch.from_numpy(
            (np.array(Image.open(sample['gt_mask']).convert('L')) > 127).astype(np.float32)
        )
        rejected_mask = torch.from_numpy(
            (np.array(Image.open(sample['rejected_mask']).convert('L')) > 127).astype(np.float32)
        )
        
        try:
            # 获取[SEG] embedding
            seg_emb = self._get_seg_embedding()
            
            # 预测mask
            pred_logits = self._predict_mask_with_grad(image, seg_emb)
            pred_prob = torch.sigmoid(pred_logits.squeeze())
            
            device = pred_prob.device
            gt_mask = gt_mask.to(device)
            rejected_mask = rejected_mask.to(device)
            
            # ============ 计算混合Loss ============
            
            # 1. Dice Loss（与GT对齐）
            loss_dice = dice_loss(pred_prob, gt_mask)
            
            # 2. DPO Loss（偏好学习）
            # 计算与GT和Rejected的相似度作为log probability
            dice_with_gt = compute_dice(pred_prob, gt_mask)
            dice_with_rejected = compute_dice(pred_prob, rejected_mask)
            
            log_prob_chosen = torch.log(dice_with_gt + 1e-8)
            log_prob_rejected = torch.log(dice_with_rejected + 1e-8)
            
            loss_dpo = dpo_loss(log_prob_chosen, log_prob_rejected, beta=self.beta)
            
            # 3. 混合Loss
            total_loss = loss_dpo + self.dice_weight * loss_dice
            
            # 计算指标
            prefer = (dice_with_gt > dice_with_rejected).float().item()
            
            return {
                'loss': total_loss,
                'loss_dpo': loss_dpo.item(),
                'loss_dice': loss_dice.item(),
                'dice': dice_with_gt.item(),
                'prefer': prefer,
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train(self):
        print("\n🚀 Starting Hybrid DPO training...")
        self.model.train()
        
        metrics = {'loss': 0, 'loss_dpo': 0, 'loss_dice': 0, 'dice': 0, 'prefer': 0, 'count': 0}
        global_step = 0
        
        pbar = tqdm(self.filtered_data, desc="Hybrid DPO V7")
        
        for idx, sample in enumerate(pbar):
            result = self.train_step(sample)
            
            if result is None:
                continue
            
            loss = result['loss'] / self.grad_accum
            
            if loss.requires_grad:
                loss.backward()
            
            for k in ['loss', 'loss_dpo', 'loss_dice', 'dice', 'prefer']:
                metrics[k] += result[k] if isinstance(result[k], float) else result[k].item()
            metrics['count'] += 1
            
            if (idx + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
                
                if metrics['count'] > 0:
                    c = metrics['count']
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']/c:.4f}",
                        'dpo': f"{metrics['loss_dpo']/c:.4f}",
                        'dice_l': f"{metrics['loss_dice']/c:.4f}",
                        'dice': f"{metrics['dice']/c:.4f}",
                        'pref': f"{metrics['prefer']/c:.1%}",
                    })
                metrics = {'loss': 0, 'loss_dpo': 0, 'loss_dice': 0, 'dice': 0, 'prefer': 0, 'count': 0}
                
                if global_step % self.save_steps == 0:
                    self._save(f'step_{global_step}')
        
        self._save('final')
        print("\n" + "=" * 60)
        print("🎉 Hybrid DPO V7 Training completed!")
        print(f"   Output: {self.output_dir}")
        print("=" * 60)
    
    def _save(self, name):
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n💾 Saving to {save_dir}...")
        
        self.model.language_model = self.model.language_model.merge_and_unload()
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Re-apply LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_r * 2,
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)


if __name__ == '__main__':
    trainer = HybridDPOTrainer()
    trainer.train()
