#!/usr/bin/env python3
"""
只微调SAM2 Mask Decoder - 最简单的方法
冻结LLM，只训练SAM2的mask decoder
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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def compute_dice(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + 1e-8) / (pred_flat.sum() + target_flat.sum() + 1e-8)


class SAM2DecoderTrainer:
    """只训练SAM2 Mask Decoder"""
    
    def __init__(self):
        self.model_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        
        self.lr = 1e-5
        self.max_samples = 500
        self.grad_accum = 4
        self.epochs = 2
        
        print("=" * 60)
        print("🚀 SAM2 Decoder Only Training")
        print("   冻结LLM，只训练SAM2 mask decoder")
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
        
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
        self.seg_token_id = self.model.seg_token_idx
        self.img_context_token_id = self.model.img_context_token_id
        
        # 完全冻结LLM和Vision
        self.model.vision_model.requires_grad_(False)
        self.model.language_model.requires_grad_(False)
        self.model.mlp1.requires_grad_(False)
        
        # 只训练SAM2 mask decoder
        trainable_count = 0
        for name, p in self.model.grounding_encoder.named_parameters():
            if 'mask_decoder' in name or 'output_upscaling' in name:
                p.requires_grad = True
                trainable_count += p.numel()
            else:
                p.requires_grad = False
        
        # text_hidden_fcs也训练
        for p in self.model.text_hidden_fcs.parameters():
            p.requires_grad = True
            trainable_count += p.numel()
        
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable_count:,} / {total:,} ({100*trainable_count/total:.4f}%)")
    
    def _load_data(self):
        print("\n📊 Loading data...")
        
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
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        
        total_steps = len(self.samples) * self.epochs // self.grad_accum
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7
        )
    
    def _prepare_inputs(self, image):
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
        
        return pixel_values, g_pixel_values, input_ids
    
    def _get_seg_embedding(self, pixel_values, input_ids):
        """获取[SEG] embedding（冻结LLM，不计算梯度）"""
        with torch.no_grad():
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
                num_to_replace = min(img_context_mask.sum().item(), vit_flat.size(0))
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
            seg_mask = (input_ids.to(hidden_states.device) == self.seg_token_id)
            seg_hidden = hidden_states[seg_mask]
        
        # text_hidden_fcs需要梯度
        seg_embedding = self.model.text_hidden_fcs(seg_hidden.to(llm_device))
        return seg_embedding
    
    def _forward_sam2(self, g_pixel_values, seg_embedding):
        """SAM2 forward with gradients"""
        sam2 = self.model.grounding_encoder
        sam2_device = next(sam2.parameters()).device
        
        g_pixel_values = g_pixel_values.to(sam2_device)
        seg_embedding = seg_embedding.to(sam2_device)
        
        if seg_embedding.dim() == 2:
            seg_embedding = seg_embedding.unsqueeze(1)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # SAM2 backbone (冻结)
            with torch.no_grad():
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
            
            # Mask decoder (需要梯度)
            _, _, _, low_res_masks, _, _, _ = sam2.sam2_model._forward_sam_heads(
                backbone_features=pix_feat.detach(),  # detach backbone features
                point_inputs=None,
                mask_inputs=None,
                high_res_features=[f.detach() for f in high_res_features],
                multimask_output=False,
                language_embd=seg_embedding,
            )
        
        return low_res_masks.squeeze(1)
    
    def train_step(self, sample):
        try:
            image = Image.open(sample['image']).convert('RGB')
            gt_mask = np.array(Image.open(sample['gt_mask']).convert('L')) > 127
            gt_mask = torch.from_numpy(gt_mask.astype(np.float32))
            
            pixel_values, g_pixel_values, input_ids = self._prepare_inputs(image)
            
            seg_embedding = self._get_seg_embedding(pixel_values, input_ids)
            if seg_embedding is None:
                return None
            
            pred_masks = self._forward_sam2(g_pixel_values, seg_embedding)
            if pred_masks is None:
                return None
            
            h, w = gt_mask.shape
            pred_logits = F.interpolate(
                pred_masks.float().unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            pred_prob = torch.sigmoid(pred_logits)
            gt_mask = gt_mask.to(pred_prob.device)
            
            # Dice + BCE loss
            dice_loss = 1 - compute_dice(pred_prob, gt_mask)
            bce_loss = F.binary_cross_entropy(pred_prob, gt_mask)
            loss = dice_loss + 0.5 * bce_loss
            
            dice = compute_dice((pred_prob > 0.5).float(), gt_mask)
            
            return {'loss': loss, 'dice': dice.item()}
            
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train(self):
        print(f"\n🚀 Training SAM2 Decoder...")
        
        for epoch in range(self.epochs):
            print(f"\n📌 Epoch {epoch+1}/{self.epochs}")
            self.model.train()
            
            metrics = {'loss': 0, 'dice': 0, 'count': 0}
            random.shuffle(self.samples)
            
            pbar = tqdm(self.samples, desc=f"Epoch {epoch+1}")
            
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
                    
                    if metrics['count'] > 0:
                        c = metrics['count']
                        pbar.set_postfix({
                            'loss': f"{metrics['loss']/c:.4f}",
                            'dice': f"{metrics['dice']/c:.4f}",
                        })
        
        # 评估
        self._evaluate()
        
        print("\n" + "=" * 60)
        print("🎉 SAM2 Decoder Training completed!")
        print("=" * 60)
    
    def _evaluate(self):
        print("\n📊 Evaluating on test set...")
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
                
                with torch.no_grad():
                    out = self.model.predict_forward(
                        image=img,
                        text='Please segment the blood vessel.',
                        tokenizer=self.tokenizer
                    )
                
                pred = out['prediction_masks'][0]
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
                
                inter = ((pred > 127) & (gt > 127)).sum()
                dice = (2 * inter) / ((pred > 127).sum() + (gt > 127).sum() + 1e-8)
                dices.append(dice)
            except Exception as e:
                pass
        
        if dices:
            print(f"\n🎯 评估结果: Dice = {np.mean(dices):.4f} (n={len(dices)})")


if __name__ == "__main__":
    trainer = SAM2DecoderTrainer()
    trainer.train()
