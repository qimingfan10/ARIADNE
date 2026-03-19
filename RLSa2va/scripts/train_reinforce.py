#!/usr/bin/env python3
"""
REINFORCE训练：直接用Dice Score作为Reward
解决训练-推理不一致问题
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
import random

sys.path.insert(0, '/home/ubuntu/Sa2VA')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def compute_dice(pred, target):
    """计算Dice系数"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + 1e-8) / (pred_flat.sum() + target_flat.sum() + 1e-8)


class REINFORCETrainer:
    """REINFORCE训练器 - 用Dice作为reward"""
    
    def __init__(self):
        self.model_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
        self.output_dir = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_reinforce"
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        
        self.lr = 5e-7  # RL需要更小的学习率
        self.lora_r = 16
        self.max_samples = 300
        self.grad_accum = 8
        self.baseline_dice = 0.82  # baseline作为reward基线
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🚀 REINFORCE Training with Dice Reward")
        print("=" * 60)
        
        self._load_model()
        self._load_data()
        self._setup_training()
    
    def _load_model(self):
        print("\n📥 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        
        # 初始化模型
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
        self.seg_token_id = self.model.seg_token_idx
        self.img_context_token_id = self.model.img_context_token_id
        
        # 应用LoRA
        print("\n🔧 Applying LoRA to LLM...")
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=32,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        
        # 冻结Vision Encoder
        self.model.vision_model.requires_grad_(False)
        
        # 训练SAM2 Mask Decoder
        for name, p in self.model.grounding_encoder.named_parameters():
            if 'mask_decoder' in name or 'output_upscaling' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        for p in self.model.text_hidden_fcs.parameters():
            p.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _load_data(self):
        """加载训练数据"""
        print("\n📊 Loading training data...")
        
        ann_file = f"{self.data_root}/dpo_annotations.json"
        with open(ann_file) as f:
            annotations = json.load(f)
        
        self.samples = []
        for item in annotations:
            if 'chosen_mask' not in item:
                continue
            
            img_path = os.path.join(self.data_root, item['image'])
            mask_path = os.path.join(self.data_root, item['chosen_mask'])
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append({
                    'image': img_path,
                    'gt_mask': mask_path,
                })
        
        random.shuffle(self.samples)
        self.samples = self.samples[:self.max_samples]
        print(f"   训练样本: {len(self.samples)}")
    
    def _setup_training(self):
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=0.01,
        )
        
        total_steps = len(self.samples) // self.grad_accum
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-9
        )
    
    def _prepare_inputs(self, image):
        """准备输入"""
        img = self.model.transformer(image)
        pixel_values = img.unsqueeze(0).to(self.model.torch_dtype)
        
        g_image = np.array(image)
        g_image = self.model.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
        g_pixel_values = g_pixel_values.to(self.model.torch_dtype)
        g_pixel_values = self.model.grounding_encoder.preprocess_image(g_pixel_values).unsqueeze(0)
        
        text = "<image>Please segment the blood vessel.[SEG]"
        input_text = self.model.template['INSTRUCTION'].format(
            input=text, round=1, bot_name=self.model.bot_name
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        return {
            'pixel_values': pixel_values,
            'g_pixel_values': g_pixel_values,
            'input_ids': input_ids,
            'ori_size': image.size,
        }
    
    def _forward_with_log_prob(self, pixel_values, input_ids):
        """Forward获取[SEG] embedding和log probability"""
        vision_device = next(self.model.vision_model.parameters()).device
        pixel_values = pixel_values.to(vision_device)
        
        vit_embeds = self.model.extract_feature(pixel_values)
        
        llm_device = next(self.model.language_model.parameters()).device
        input_ids = input_ids.to(llm_device)
        
        text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        
        B, N, C = text_embeds.shape
        input_embeds = text_embeds.clone()
        
        img_context_mask = (input_ids == self.img_context_token_id)
        if img_context_mask.sum() > 0:
            vit_flat = vit_embeds.reshape(-1, C).to(llm_device)
            num_img_tokens = img_context_mask.sum().item()
            num_to_replace = min(num_img_tokens, vit_flat.size(0))
            
            img_positions = img_context_mask[0].nonzero(as_tuple=True)[0][:num_to_replace]
            input_embeds[0, img_positions] = vit_flat[:num_to_replace]
        
        attention_mask = torch.ones_like(input_ids).to(llm_device)
        
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 获取logits用于计算log probability
        logits = outputs.logits  # [B, N, vocab_size]
        
        # 计算[SEG] token的log probability
        seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
        if len(seg_positions[0]) == 0:
            return None, None
        
        seg_pos = seg_positions[1][0].item()
        if seg_pos > 0:
            # [SEG] token的预测logits来自前一个位置
            seg_logits = logits[0, seg_pos - 1]  # [vocab_size]
            log_probs = F.log_softmax(seg_logits, dim=-1)
            seg_log_prob = log_probs[self.seg_token_id]
        else:
            seg_log_prob = torch.tensor(0.0, device=llm_device)
        
        # 获取[SEG] embedding
        hidden_states = outputs.hidden_states[-1]
        seg_mask = (input_ids.to(hidden_states.device) == self.seg_token_id)
        seg_hidden = hidden_states[seg_mask]
        seg_embedding = self.model.text_hidden_fcs(seg_hidden.to(llm_device))
        
        return seg_embedding, seg_log_prob
    
    def _forward_sam2(self, g_pixel_values, seg_embedding):
        """SAM2 forward"""
        sam2 = self.model.grounding_encoder
        sam2_device = next(sam2.parameters()).device
        
        g_pixel_values = g_pixel_values.to(sam2_device)
        seg_embedding = seg_embedding.to(sam2_device)
        
        if seg_embedding.dim() == 2:
            seg_embedding = seg_embedding.unsqueeze(1)
        
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
        
        return low_res_masks.squeeze(1)
    
    def train_step(self, sample):
        """REINFORCE训练步"""
        try:
            image = Image.open(sample['image']).convert('RGB')
            gt_mask = np.array(Image.open(sample['gt_mask']).convert('L')) > 127
            gt_mask_tensor = torch.from_numpy(gt_mask.astype(np.float32))
            
            inputs = self._prepare_inputs(image)
            
            # Forward获取embedding和log probability
            seg_embedding, seg_log_prob = self._forward_with_log_prob(
                inputs['pixel_values'],
                inputs['input_ids']
            )
            
            if seg_embedding is None:
                return None
            
            # SAM2预测
            pred_masks = self._forward_sam2(inputs['g_pixel_values'], seg_embedding)
            
            if pred_masks is None:
                return None
            
            # 调整尺寸
            h, w = gt_mask.shape
            pred_logits = F.interpolate(
                pred_masks.float().unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            pred_prob = torch.sigmoid(pred_logits)
            pred_binary = (pred_prob > 0.5).float()
            
            gt_mask_tensor = gt_mask_tensor.to(pred_prob.device)
            
            # 计算Dice作为reward
            dice = compute_dice(pred_binary, gt_mask_tensor)
            
            # Reward = Dice - baseline (advantage)
            advantage = dice - self.baseline_dice
            
            # REINFORCE loss: -log_prob * advantage
            # 如果advantage > 0，我们想增加这个action的概率
            # 如果advantage < 0，我们想减少这个action的概率
            
            # 同时加入Dice loss作为辅助监督
            dice_loss = 1 - compute_dice(pred_prob, gt_mask_tensor)
            
            # 综合loss
            reinforce_loss = -seg_log_prob * advantage.detach()
            total_loss = reinforce_loss + 0.5 * dice_loss
            
            return {
                'loss': total_loss,
                'dice': dice.item(),
                'advantage': advantage.item(),
                'reinforce_loss': reinforce_loss.item(),
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train(self):
        print(f"\n🚀 Starting REINFORCE Training...")
        print(f"   Baseline Dice: {self.baseline_dice}")
        self.model.train()
        
        metrics = {'loss': 0, 'dice': 0, 'adv': 0, 'count': 0}
        global_step = 0
        
        pbar = tqdm(self.samples, desc="REINFORCE")
        
        for idx, sample in enumerate(pbar):
            result = self.train_step(sample)
            
            if result is None:
                continue
            
            loss = result['loss'] / self.grad_accum
            
            if loss.requires_grad:
                loss.backward()
            
            metrics['loss'] += result['loss'].item() if hasattr(result['loss'], 'item') else result['loss']
            metrics['dice'] += result['dice']
            metrics['adv'] += result['advantage']
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
                        'dice': f"{metrics['dice']/c:.4f}",
                        'adv': f"{metrics['adv']/c:.4f}",
                    })
                metrics = {'loss': 0, 'dice': 0, 'adv': 0, 'count': 0}
        
        # 评估最终结果
        self._evaluate()
        
        print("\n" + "=" * 60)
        print("🎉 REINFORCE Training completed!")
        print("=" * 60)
    
    def _evaluate(self):
        """训练后评估"""
        print("\n📊 Evaluating...")
        self.model.eval()
        
        dices = []
        eval_samples = random.sample(self.samples, min(20, len(self.samples)))
        
        for sample in eval_samples:
            try:
                image = Image.open(sample['image']).convert('RGB')
                gt_mask = np.array(Image.open(sample['gt_mask']).convert('L')) > 127
                
                with torch.no_grad():
                    out = self.model.predict_forward(
                        image=image,
                        text='Please segment the blood vessel.',
                        tokenizer=self.tokenizer
                    )
                
                pred = out['prediction_masks'][0]
                h, w = gt_mask.shape
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
                
                inter = ((pred > 127) & gt_mask).sum()
                dice = (2 * inter) / ((pred > 127).sum() + gt_mask.sum() + 1e-8)
                dices.append(dice)
            except:
                pass
        
        if dices:
            print(f"   训练后Dice: {np.mean(dices):.4f}")
        
        self.model.train()


if __name__ == "__main__":
    trainer = REINFORCETrainer()
    trainer.train()
