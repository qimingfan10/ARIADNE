#!/usr/bin/env python3
"""
Sa2VA V12 - 纯监督学习（基于V8架构）

策略：
1. 从V8 checkpoint开始（Dice ~0.82）
2. 使用纯监督loss（Dice + BCE），不用DPO
3. 7:3混合采样（困难+简单样本）
4. 极低学习率 5e-7
5. 冻结LLM，只训练SAM2 decoder和text_hidden_fcs
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
import random
import cv2

sys.path.insert(0, '/home/ubuntu/Sa2VA')

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def compute_dice(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + 1e-8) / (pred_flat.sum() + target_flat.sum() + 1e-8)


def dice_loss(pred, target):
    return 1 - compute_dice(pred, target)


class SupervisedTrainer:
    """纯监督学习训练器"""
    
    def __init__(self):
        # 从V8 checkpoint开始
        self.model_path = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v8/step_100"
        self.output_dir = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_supervised_v12"
        self.data_root = "/home/ubuntu/Sa2VA/data/merged_vessel_data"
        
        # 训练参数
        self.lr = 5e-7  # 极低学习率
        self.epochs = 2
        self.grad_accum = 4
        self.max_samples = 200
        
        # 混合采样阈值
        self.hard_threshold = 0.75  # Dice < 0.75 为困难样本
        self.easy_threshold = 0.85  # Dice > 0.85 为简单样本
        self.hard_ratio = 0.7  # 70%困难样本
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🚀 Supervised Training V12")
        print(f"   Model: {self.model_path}")
        print(f"   LR: {self.lr}")
        print(f"   Hard:Easy = {int(self.hard_ratio*100)}:{int((1-self.hard_ratio)*100)}")
        print("=" * 60)
        
        self._load_model()
        self._load_and_select_samples()
        self._setup_training()
    
    def _load_model(self):
        print("\n📥 Loading V8 checkpoint...")
        
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
        self.img_context_token_id = self.model.img_context_token_id
        
        # 冻结所有参数
        print("🔧 Freezing all parameters...")
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 只解冻 text_hidden_fcs
        print("🔥 Unfreezing text_hidden_fcs...")
        trainable = 0
        for name, param in self.model.named_parameters():
            if 'text_hidden_fcs' in name:
                param.requires_grad = True
                trainable += param.numel()
        
        # 解冻 SAM2 mask decoder
        print("🔥 Unfreezing SAM2 mask decoder...")
        sam2 = self.model.grounding_encoder.sam2_model
        for name, param in sam2.named_parameters():
            if 'mask_decoder' in name or 'sam_mask_decoder' in name:
                param.requires_grad = True
                trainable += param.numel()
        
        print(f"   Total trainable: {trainable/1e6:.2f}M")
    
    def _get_baseline_dice(self, image_path, gt_mask):
        """获取baseline预测的Dice"""
        try:
            img = Image.open(image_path).convert('RGB')
            w, h = img.size
            
            with torch.no_grad():
                out = self.model.predict_forward(
                    image=img,
                    text='<image>Please segment the blood vessel.',
                    tokenizer=self.tokenizer,
                    processor=None,
                )
            
            pred = out['prediction_masks'][0][0]
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            pred = pred.astype(np.float32)
            if pred.max() <= 1.0:
                pred = pred * 255
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            pred_binary = (pred > 127).astype(np.float32)
            
            gt = gt_mask.astype(np.float32)
            inter = (pred_binary * gt).sum()
            dice = (2 * inter) / (pred_binary.sum() + gt.sum() + 1e-8)
            return dice
        except:
            return 0.8  # 默认中等难度
    
    def _load_and_select_samples(self):
        """加载数据并进行混合采样"""
        print("\n📊 Loading and selecting samples...")
        
        with open(f'{self.data_root}/annotations.json') as f:
            annotations = json.load(f)
        
        valid = [a for a in annotations if 'mask' in a and len(a['mask']) > 0]
        print(f"   Total valid samples: {len(valid)}")
        
        # 计算每个样本的Dice并分类
        print("   Computing baseline Dice for all samples...")
        hard_samples = []
        easy_samples = []
        
        for ann in tqdm(valid[:500], desc="Evaluating"):  # 只评估前500个
            image_path = f'{self.data_root}/images/{ann["image"]}'
            if not os.path.exists(image_path):
                continue
            
            # 加载GT mask
            img = Image.open(image_path)
            w, h = img.size
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            for m in ann['mask']:
                if len(m) >= 6:
                    pts = np.array(m).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(gt_mask, [pts], 255)
            gt_mask = (gt_mask > 127).astype(np.uint8)
            
            dice = self._get_baseline_dice(image_path, gt_mask)
            
            sample = {
                'image': image_path,
                'ann': ann,
                'dice': dice
            }
            
            if dice < self.hard_threshold:
                hard_samples.append(sample)
            elif dice > self.easy_threshold:
                easy_samples.append(sample)
        
        print(f"   Hard samples (Dice<{self.hard_threshold}): {len(hard_samples)}")
        print(f"   Easy samples (Dice>{self.easy_threshold}): {len(easy_samples)}")
        
        # 7:3混合采样
        n_hard = int(self.max_samples * self.hard_ratio)
        n_easy = self.max_samples - n_hard
        
        random.seed(42)
        selected_hard = random.sample(hard_samples, min(n_hard, len(hard_samples)))
        selected_easy = random.sample(easy_samples, min(n_easy, len(easy_samples)))
        
        self.samples = selected_hard + selected_easy
        random.shuffle(self.samples)
        
        print(f"   Selected: {len(selected_hard)} hard + {len(selected_easy)} easy = {len(self.samples)}")
    
    def _setup_training(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        
        total_steps = len(self.samples) * self.epochs // self.grad_accum
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-8
        )
    
    def _prepare_inputs(self, image):
        """准备输入"""
        ori_size = image.size
        
        # Vision encoder输入
        pixel_values = self.model.transformer(image).unsqueeze(0).to(torch.bfloat16)
        
        # SAM2输入
        g_image = np.array(image)
        g_image = self.model.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(torch.bfloat16)
        g_pixel_values = self.model.grounding_encoder.preprocess_image(g_pixel_values).unsqueeze(0)
        
        # 构建输入文本
        text = "<image>Please segment the blood vessel.[SEG]"
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
        """通过完整LLM forward获取[SEG] embedding"""
        vision_device = next(self.model.vision_model.parameters()).device
        pixel_values = pixel_values.to(vision_device)
        
        # Vision encoder (frozen, no grad)
        with torch.no_grad():
            vit_embeds = self.model.extract_feature(pixel_values)
        
        llm_device = next(self.model.language_model.parameters()).device
        input_ids = input_ids.to(llm_device)
        
        # LLM forward (frozen, no grad)
        with torch.no_grad():
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
        
        # text_hidden_fcs有梯度
        seg_embedding = self.model.text_hidden_fcs(seg_hidden.to(llm_device))
        
        return seg_embedding
    
    def _predict_mask_with_embedding(self, g_pixel_values, seg_embedding, ori_size):
        """使用seg_embedding预测mask"""
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
        
        w, h = ori_size
        masks = F.interpolate(low_res_masks, size=(h, w), mode='bilinear', align_corners=False)
        return masks.squeeze(1)
    
    def train_step(self, sample):
        """执行一个训练步骤"""
        try:
            image = Image.open(sample['image']).convert('RGB')
            w, h = image.size
            
            # 加载GT mask
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            for m in sample['ann']['mask']:
                if len(m) >= 6:
                    pts = np.array(m).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(gt_mask, [pts], 255)
            gt_mask = torch.from_numpy((gt_mask > 127).astype(np.float32))
            
            # 准备输入
            inputs = self._prepare_inputs(image)
            
            # 获取seg embedding
            seg_embedding = self._forward_get_seg_embedding(
                inputs['pixel_values'], inputs['input_ids']
            )
            if seg_embedding is None:
                return None
            
            # 预测mask
            pred_logits = self._predict_mask_with_embedding(
                inputs['g_pixel_values'], seg_embedding, inputs['ori_size']
            )
            
            pred_prob = torch.sigmoid(pred_logits.squeeze())
            gt_mask = gt_mask.to(pred_prob.device)
            
            # 纯监督Loss: Dice + BCE
            d_loss = dice_loss(pred_prob, gt_mask)
            bce = F.binary_cross_entropy(pred_prob, gt_mask)
            loss = d_loss + 0.5 * bce
            
            dice = compute_dice((pred_prob > 0.5).float(), gt_mask)
            
            return {'loss': loss, 'dice': dice.item(), 'sample_dice': sample['dice']}
            
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def _evaluate(self):
        """使用predict_forward评估"""
        print("\n📊 Evaluating with predict_forward...")
        
        eval_root = '/home/ubuntu/Sa2VA/data/merged_vessel_data'
        with open(f'{eval_root}/annotations.json') as f:
            anns = json.load(f)
        valid = [a for a in anns if 'mask' in a and len(a['mask']) > 0]
        
        random.seed(42)
        samples = random.sample(valid, 10)
        
        dices = []
        for s in tqdm(samples, desc="Eval"):
            try:
                img = Image.open(f'{eval_root}/images/{s["image"]}').convert('RGB')
                w, h = img.size
                
                gt = np.zeros((h, w), dtype=np.uint8)
                for m in s['mask']:
                    if len(m) >= 6:
                        pts = np.array(m).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(gt, [pts], 255)
                
                with torch.no_grad():
                    out = self.model.predict_forward(
                        image=img,
                        text='<image>Please segment the blood vessel.',
                        tokenizer=self.tokenizer,
                        processor=None,
                    )
                
                pred = out['prediction_masks'][0][0]
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                pred = pred.astype(np.float32)
                if pred.max() <= 1.0:
                    pred = pred * 255
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
                
                pred_binary = (pred > 127).astype(np.uint8)
                gt_binary = (gt > 127).astype(np.uint8)
                
                inter = (pred_binary & gt_binary).sum()
                dice = (2 * inter) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
                dices.append(dice)
            except:
                pass
        
        return np.mean(dices) if dices else 0.0
    
    def train(self):
        """主训练循环"""
        print(f"\n🏋️ Training...")
        print(f"   Samples: {len(self.samples)}")
        print(f"   Epochs: {self.epochs}, LR: {self.lr}")
        
        baseline_dice = 0.8189  # V8 step_100的Dice
        
        for epoch in range(self.epochs):
            print(f"\n📌 Epoch {epoch+1}/{self.epochs}")
            
            self.optimizer.zero_grad()
            epoch_losses = []
            epoch_dices = []
            
            pbar = tqdm(self.samples, desc=f"Epoch {epoch+1}")
            for i, sample in enumerate(pbar):
                result = self.train_step(sample)
                
                if result is not None:
                    loss = result['loss'] / self.grad_accum
                    loss.backward()
                    epoch_losses.append(result['loss'].item())
                    epoch_dices.append(result['dice'])
                
                if (i + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        max_norm=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                if epoch_losses:
                    pbar.set_postfix({
                        'loss': f'{np.mean(epoch_losses[-20:]):.4f}',
                        'dice': f'{np.mean(epoch_dices[-20:]):.4f}'
                    })
            
            if epoch_losses:
                print(f"   Epoch {epoch+1} - Loss: {np.mean(epoch_losses):.4f}, Train Dice: {np.mean(epoch_dices):.4f}")
        
        # 评估
        final_dice = self._evaluate()
        
        print(f"\n🎯 Results:")
        print(f"   Mean Dice: {final_dice:.4f}")
        print(f"   V8 Baseline: {baseline_dice:.4f}")
        print(f"   Change: {final_dice - baseline_dice:+.4f}")
        
        # 保存模型
        if final_dice > baseline_dice:
            print(f"\n💾 Saving improved model...")
            save_path = f"{self.output_dir}/final"
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"   Saved to {save_path}")
        
        print("\n" + "=" * 60)
        print("🎉 Training completed!")
        print("=" * 60)


if __name__ == "__main__":
    trainer = SupervisedTrainer()
    trainer.train()
