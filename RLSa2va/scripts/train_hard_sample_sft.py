#!/usr/bin/env python3
"""
困难样本SFT (Hard Example Mining + Supervised Fine-Tuning)
策略：只在Baseline预测较差的样本上进行微调
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
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + 1e-8) / (pred_flat.sum() + target_flat.sum() + 1e-8)


def dice_loss(pred, target):
    return 1 - compute_dice(pred, target)


class HardSampleSFTTrainer:
    """困难样本SFT训练器"""
    
    def __init__(self):
        self.model_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
        self.output_dir = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_hard_sft"
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        
        self.lr = 5e-7  # 使用更小的学习率防止遗忘
        self.lora_r = 8  # 更小的LoRA rank
        self.max_samples = 200
        self.grad_accum = 8
        self.save_steps = 100
        self.dice_threshold = 0.80  # 混合采样：Dice < 0.80的困难样本 + 部分简单样本
        self.mix_easy_ratio = 0.3  # 30%简单样本混合
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🚀 Hard Sample SFT Training")
        print(f"   只在Dice < {self.dice_threshold}的困难样本上训练")
        print("=" * 60)
        
        self._load_model()
        self._load_hard_samples()
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
        
        # 初始化模型属性（关键！）
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
        
        # 获取特殊token ID
        self.seg_token_id = self.model.seg_token_idx
        self.img_context_token_id = self.model.img_context_token_id
        
        # 完全冻结LLM（不使用LoRA，避免embedding漂移）
        print("\n🔧 Freezing LLM completely (no LoRA)...")
        for param in self.model.language_model.parameters():
            param.requires_grad = False
        
        # 冻结Vision Encoder
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.model.mlp1.parameters():
            param.requires_grad = False
        
        # 只训练SAM2 mask decoder和text_hidden_fcs
        for name, param in self.model.grounding_encoder.named_parameters():
            if 'mask_decoder' in name or 'output_upscaling' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for param in self.model.text_hidden_fcs.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _load_hard_samples(self):
        """加载困难样本（rejected mask与GT差距大的样本）"""
        print("\n📊 Loading hard samples...")
        
        ann_file = f"{self.data_root}/dpo_annotations.json"
        with open(ann_file) as f:
            annotations = json.load(f)
        
        self.hard_samples = []
        easy_samples = []
        easy_count = 0
        
        for item in annotations:
            if 'chosen_mask' not in item or 'rejected_mask' not in item:
                continue
            
            chosen_path = os.path.join(self.data_root, item['chosen_mask'])
            rejected_path = os.path.join(self.data_root, item['rejected_mask'])
            
            if not os.path.exists(chosen_path) or not os.path.exists(rejected_path):
                continue
            
            # GT mask
            chosen = np.array(Image.open(chosen_path).convert('L')) > 127
            # Baseline预测
            rejected = np.array(Image.open(rejected_path).convert('L')) > 127
            
            # 计算Baseline的Dice
            inter = (chosen & rejected).sum()
            baseline_dice = (2 * inter) / (chosen.sum() + rejected.sum() + 1e-8)
            
            sample = {
                'image': os.path.join(self.data_root, item['image']),
                'gt_mask': chosen_path,
                'baseline_dice': baseline_dice,
            }
            
            # 分类困难/简单样本
            if baseline_dice < self.dice_threshold:
                self.hard_samples.append(sample)
            else:
                easy_samples.append(sample)
                easy_count += 1
        
        # 混合采样：困难样本 + 30%简单样本
        n_hard = int(self.max_samples * (1 - self.mix_easy_ratio))
        n_easy = int(self.max_samples * self.mix_easy_ratio)
        
        random.shuffle(self.hard_samples)
        random.shuffle(easy_samples)
        
        self.hard_samples = self.hard_samples[:n_hard] + easy_samples[:n_easy]
        random.shuffle(self.hard_samples)
        
        print(f"   困难样本 (Dice < {self.dice_threshold}): {min(n_hard, len(self.hard_samples))}")
        print(f"   简单样本 (混合): {min(n_easy, len(easy_samples))}")
        print(f"   总训练样本: {len(self.hard_samples)}")
        
        if len(self.hard_samples) == 0:
            print("⚠️  没有困难样本！尝试提高阈值...")
            self.dice_threshold = 0.85
            self._load_hard_samples_with_higher_threshold()
    
    def _load_hard_samples_with_higher_threshold(self):
        """使用更高的阈值加载样本"""
        ann_file = f"{self.data_root}/dpo_annotations.json"
        with open(ann_file) as f:
            annotations = json.load(f)
        
        self.hard_samples = []
        
        for item in annotations:
            if 'chosen_mask' not in item or 'rejected_mask' not in item:
                continue
            
            chosen_path = os.path.join(self.data_root, item['chosen_mask'])
            rejected_path = os.path.join(self.data_root, item['rejected_mask'])
            
            if not os.path.exists(chosen_path) or not os.path.exists(rejected_path):
                continue
            
            chosen = np.array(Image.open(chosen_path).convert('L')) > 127
            rejected = np.array(Image.open(rejected_path).convert('L')) > 127
            
            inter = (chosen & rejected).sum()
            baseline_dice = (2 * inter) / (chosen.sum() + rejected.sum() + 1e-8)
            
            if baseline_dice < self.dice_threshold:
                self.hard_samples.append({
                    'image': os.path.join(self.data_root, item['image']),
                    'gt_mask': chosen_path,
                    'baseline_dice': baseline_dice,
                })
        
        self.hard_samples = self.hard_samples[:self.max_samples]
        print(f"   困难样本 (Dice < {self.dice_threshold}): {len(self.hard_samples)}")
    
    def _setup_training(self):
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=0.01,
        )
        
        total_steps = len(self.hard_samples) // self.grad_accum
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-8
        )
    
    def _prepare_inputs(self, image):
        """准备模型输入（与predict_forward一致）"""
        ori_size = image.size
        
        # 使用模型的transformer处理图像
        img = self.model.transformer(image)
        pixel_values = img.unsqueeze(0).to(self.model.torch_dtype)
        
        # SAM2输入
        g_image = np.array(image)
        g_image = self.model.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
        g_pixel_values = g_pixel_values.to(self.model.torch_dtype)
        g_pixel_values = self.model.grounding_encoder.preprocess_image(g_pixel_values).unsqueeze(0)
        
        # 构建输入文本
        text = "<image>Please segment the blood vessel.[SEG]"
        
        # Tokenize
        input_text = self.model.template['INSTRUCTION'].format(
            input=text, round=1, bot_name=self.model.bot_name
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        return {
            'pixel_values': pixel_values,
            'g_pixel_values': g_pixel_values,
            'input_ids': input_ids,
            'ori_size': ori_size,
        }
    
    def _forward_get_seg_embedding(self, pixel_values, input_ids):
        """获取[SEG] embedding (LLM冻结，使用no_grad)"""
        # Vision encoder (冻结)
        with torch.no_grad():
            vision_device = next(self.model.vision_model.parameters()).device
            pixel_values = pixel_values.to(vision_device)
            vit_embeds = self.model.extract_feature(pixel_values)
        
        # LLM forward (冻结)
        with torch.no_grad():
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
        
        hidden_states = outputs.hidden_states[-1]
        hs_device = hidden_states.device
        
        seg_mask = (input_ids.to(hs_device) == self.seg_token_id)
        if seg_mask.sum() == 0:
            return None
        
        seg_hidden = hidden_states[seg_mask]
        seg_embedding = self.model.text_hidden_fcs(seg_hidden.to(llm_device))
        
        return seg_embedding
    
    def _forward_sam2(self, g_pixel_values, seg_embedding, ori_size):
        """SAM2 forward (与V8一致)"""
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
        """单步训练"""
        try:
            image = Image.open(sample['image']).convert('RGB')
            gt_mask = np.array(Image.open(sample['gt_mask']).convert('L')) > 127
            gt_mask = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0)
            
            # 准备输入
            inputs = self._prepare_inputs(image)
            
            # 获取[SEG] embedding
            seg_embedding = self._forward_get_seg_embedding(
                inputs['pixel_values'], 
                inputs['input_ids']
            )
            
            if seg_embedding is None:
                return None
            
            # SAM2预测
            pred_masks = self._forward_sam2(
                inputs['g_pixel_values'],
                seg_embedding,
                inputs['ori_size']
            )
            
            if pred_masks is None:
                return None
            
            # 调整尺寸
            h, w = gt_mask.shape[-2:]
            pred_logits = F.interpolate(
                pred_masks.float().unsqueeze(0), 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            pred_prob = torch.sigmoid(pred_logits)
            gt_mask = gt_mask.to(pred_prob.device)
            
            # 只用Dice Loss进行监督学习
            loss = dice_loss(pred_prob, gt_mask)
            
            dice_score = compute_dice(pred_prob, gt_mask)
            
            return {
                'loss': loss,
                'dice': dice_score.item(),
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train(self):
        print(f"\n🚀 Starting Hard Sample SFT...")
        print(f"   训练样本: {len(self.hard_samples)}")
        self.model.train()
        
        metrics = {'loss': 0, 'dice': 0, 'count': 0}
        global_step = 0
        
        pbar = tqdm(self.hard_samples, desc="Hard SFT")
        
        for idx, sample in enumerate(pbar):
            result = self.train_step(sample)
            
            if result is None:
                continue
            
            loss = result['loss'] / self.grad_accum
            
            if loss.requires_grad:
                loss.backward()
            
            metrics['loss'] += result['loss'].item()
            metrics['dice'] += result['dice']
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
                        'dice': f"{metrics['dice']/c:.4f}",
                    })
                metrics = {'loss': 0, 'dice': 0, 'count': 0}
                
        # 训练完成后直接评估（不保存模型）
        self._evaluate()
        
        print("\n" + "=" * 60)
        print("🎉 Hard Sample SFT completed!")
        print("=" * 60)
    
    def _evaluate(self):
        """训练后评估（使用训练时相同的forward路径）"""
        print("\n📊 Evaluating...")
        self.model.eval()
        
        # 使用merged_vessel_data评估
        eval_root = "/home/ubuntu/Sa2VA/data/merged_vessel_data"
        with open(f"{eval_root}/annotations.json") as f:
            anns = json.load(f)
        
        valid = [a for a in anns if 'mask' in a and len(a['mask']) > 0]
        random.seed(42)
        eval_samples = random.sample(valid, min(20, len(valid)))
        
        dices = []
        for s in tqdm(eval_samples, desc="Eval"):
            try:
                img = Image.open(f"{eval_root}/images/{s['image']}").convert('RGB')
                w, h = img.size
                
                gt = np.zeros((h, w), dtype=np.uint8)
                for m in s['mask']:
                    if len(m) >= 6:
                        pts = np.array(m).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(gt, [pts], 255)
                
                # 使用训练时相同的forward路径
                with torch.no_grad():
                    inputs = self._prepare_inputs(img)
                    seg_embedding = self._forward_get_seg_embedding(
                        inputs['pixel_values'], inputs['input_ids']
                    )
                    if seg_embedding is None:
                        continue
                    pred_masks = self._forward_sam2(
                        inputs['g_pixel_values'], seg_embedding, inputs['ori_size']
                    )
                
                # 调整尺寸
                pred_logits = F.interpolate(
                    pred_masks.float().unsqueeze(0), 
                    size=(h, w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                
                pred_prob = torch.sigmoid(pred_logits).cpu().numpy()
                pred_binary = (pred_prob > 0.5).astype(np.uint8)
                gt_binary = (gt > 127).astype(np.uint8)
                
                inter = (pred_binary & gt_binary).sum()
                dice = (2 * inter) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
                dices.append(dice)
                print(f"  {s['image'][:30]}... Dice={dice:.4f}")
            except Exception as e:
                import traceback
                print(f"  Error: {e}")
                traceback.print_exc()
        
        if dices:
            print(f"\n🎯 平均Dice: {np.mean(dices):.4f} (n={len(dices)})")
            print(f"   Baseline: 0.8191")
            print(f"   变化: {np.mean(dices) - 0.8191:+.4f}")


if __name__ == "__main__":
    trainer = HardSampleSFTTrainer()
    trainer.train()
