#!/usr/bin/env python3
"""
Sa2VA 26B DPO V8 - 使用完整LLM Forward路径

关键改进：
1. 使用model.forward()而非简化embedding，确保训练路径与推理路径一致
2. 从LLM hidden states中提取[SEG] embedding
3. DPO + Dice混合损失
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
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + 1e-8) / (pred_flat.sum() + target_flat.sum() + 1e-8)


def dice_loss(pred, target):
    return 1 - compute_dice(pred, target)


def dpo_loss(log_prob_chosen, log_prob_rejected, beta=0.2):
    logits = beta * (log_prob_chosen - log_prob_rejected)
    return -F.logsigmoid(logits)


class FullForwardDPOTrainer:
    """使用完整LLM forward的DPO训练器"""
    
    def __init__(self):
        self.model_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
        self.output_dir = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v8"
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        
        self.beta = 0.2
        self.dice_weight = 1.0
        self.lr = 1e-5  # 中等学习率
        self.lora_r = 16
        self.max_samples = 500
        self.grad_accum = 4
        self.save_steps = 100
        self.margin = 0.15
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🚀 Full Forward DPO V8")
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
        self.img_context_token_id = self.model.img_context_token_id
        
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
    
    def _load_and_filter_data(self):
        print("\n📊 Loading data...")
        
        ann_file = f"{self.data_root}/dpo_annotations.json"
        with open(ann_file) as f:
            annotations = json.load(f)
        
        self.filtered_data = []
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
            rejected_dice = (2 * inter) / (chosen.sum() + rejected.sum() + 1e-8)
            dice_gap = 1.0 - rejected_dice
            
            if dice_gap >= self.margin:
                self.filtered_data.append({
                    'image': os.path.join(self.data_root, item['image']),
                    'gt_mask': chosen_path,
                    'rejected_mask': rejected_path,
                    'dice_gap': dice_gap,
                })
        
        self.filtered_data = self.filtered_data[:self.max_samples]
        print(f"   {len(self.filtered_data)} DPO pairs")
    
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
    
    def _prepare_inputs(self, image):
        """准备模型输入（与predict_forward一致）"""
        # 处理图像
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
        """
        通过完整LLM forward获取[SEG] embedding
        这是关键：确保训练路径与推理路径一致
        """
        # 获取vision model所在的设备
        vision_device = next(self.model.vision_model.parameters()).device
        
        pixel_values = pixel_values.to(vision_device)
        
        # 1. 获取vision embeddings（使用extract_feature保持一致）
        vit_embeds = self.model.extract_feature(pixel_values)
        
        # 获取language model所在的设备（可能不同）
        llm_device = next(self.model.language_model.parameters()).device
        input_ids = input_ids.to(llm_device)
        
        # 2. 获取text embeddings
        text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        
        # 3. 替换IMG_CONTEXT位置为vision embeddings
        B, N, C = text_embeds.shape
        input_embeds = text_embeds.clone()
        
        # 将img_context_mask移动到正确的设备
        img_context_mask = (input_ids == self.img_context_token_id)
        if img_context_mask.sum() > 0:
            vit_flat = vit_embeds.reshape(-1, C).to(llm_device)
            num_img_tokens = img_context_mask.sum().item()
            num_to_replace = min(num_img_tokens, vit_flat.size(0))
            
            img_positions = img_context_mask[0].nonzero(as_tuple=True)[0][:num_to_replace]
            input_embeds[0, img_positions] = vit_flat[:num_to_replace]
        
        # 4. LLM forward（获取hidden states）
        attention_mask = torch.ones_like(input_ids).to(llm_device)
        
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 5. 提取[SEG]位置的hidden state
        hidden_states = outputs.hidden_states[-1]  # 最后一层
        hs_device = hidden_states.device
        
        # 确保seg_mask在与hidden_states相同的设备上
        seg_mask = (input_ids.to(hs_device) == self.seg_token_id)
        if seg_mask.sum() == 0:
            return None
        
        seg_hidden = hidden_states[seg_mask]  # [num_seg, hidden_dim]
        
        # 6. 通过text_hidden_fcs
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
        if not os.path.exists(sample['image']):
            return None
        
        image = Image.open(sample['image']).convert('RGB')
        gt_mask = torch.from_numpy(
            (np.array(Image.open(sample['gt_mask']).convert('L')) > 127).astype(np.float32)
        )
        rejected_mask = torch.from_numpy(
            (np.array(Image.open(sample['rejected_mask']).convert('L')) > 127).astype(np.float32)
        )
        
        try:
            # 准备输入
            inputs = self._prepare_inputs(image)
            
            # 使用完整LLM forward获取[SEG] embedding
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
            
            device = pred_prob.device
            gt_mask = gt_mask.to(device)
            rejected_mask = rejected_mask.to(device)
            
            # 计算Loss
            loss_dice = dice_loss(pred_prob, gt_mask)
            
            dice_with_gt = compute_dice(pred_prob, gt_mask)
            dice_with_rejected = compute_dice(pred_prob, rejected_mask)
            
            log_prob_chosen = torch.log(dice_with_gt + 1e-8)
            log_prob_rejected = torch.log(dice_with_rejected + 1e-8)
            
            loss_dpo = dpo_loss(log_prob_chosen, log_prob_rejected, beta=self.beta)
            
            total_loss = loss_dpo + self.dice_weight * loss_dice
            
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
        print("\n🚀 Starting Full Forward DPO V8...")
        self.model.train()
        
        metrics = {'loss': 0, 'loss_dpo': 0, 'loss_dice': 0, 'dice': 0, 'prefer': 0, 'count': 0}
        global_step = 0
        
        pbar = tqdm(self.filtered_data, desc="DPO V8")
        
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
                        'dice': f"{metrics['dice']/c:.4f}",
                        'pref': f"{metrics['prefer']/c:.1%}",
                    })
                metrics = {'loss': 0, 'loss_dpo': 0, 'loss_dice': 0, 'dice': 0, 'prefer': 0, 'count': 0}
                
                if global_step % self.save_steps == 0:
                    self._save(f'step_{global_step}')
        
        self._save('final')
        print("\n" + "=" * 60)
        print("🎉 DPO V8 Training completed!")
        print(f"   Output: {self.output_dir}")
        print("=" * 60)
    
    def _save(self, name):
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n💾 Saving to {save_dir}...")
        
        self.model.language_model = self.model.language_model.merge_and_unload()
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
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
    trainer = FullForwardDPOTrainer()
    trainer.train()
