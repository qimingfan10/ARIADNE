#!/usr/bin/env python3
"""
Sa2VA 26B DPO训练 V4 - 使用完整推理路径

关键改进：使用模型的language_embd_inference方法（已移除@torch.no_grad()装饰器）
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
    
    # Chosen mask log prob (Dice similarity)
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


class DPOTrainerV4:
    def __init__(self):
        self.model_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
        self.output_dir = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v4"
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        self.beta = 0.1
        self.lr = 2e-5
        self.lora_r = 16
        self.max_samples = 500
        self.grad_accum = 4
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🎯 Sa2VA 26B DPO Training V4 - Full Inference Path")
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
        self.seg_token_id = self.model.seg_token_idx
        print(f"✅ Model loaded! [SEG] id: {self.seg_token_id}")
        
        # LoRA on language model
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
    
    def _get_seg_hidden_state_full(self, image):
        """
        使用完整的LLM forward获取[SEG] token的hidden state
        """
        device = next(self.model.parameters()).device
        
        # 准备图像
        img = self.model.transformer(image)
        pixel_values = img.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
        
        # 构建输入文本
        num_image_tokens = self.model.patch_token
        image_token_str = f'{self.model.IMG_START_TOKEN}{self.model.IMG_CONTEXT_TOKEN * num_image_tokens}{self.model.IMG_END_TOKEN}\n'
        text = "Please segment the blood vessels in this image. [SEG]"
        
        full_text = self.model.template['INSTRUCTION'].format(
            input=image_token_str + text, round=1, bot_name=self.model.bot_name
        )
        
        # Tokenize
        input_ids = self.tokenizer.encode(full_text)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        
        # 找[SEG]位置
        seg_positions = (input_ids[0] == self.seg_token_id).nonzero(as_tuple=True)[0]
        if len(seg_positions) == 0:
            return None
        seg_pos = seg_positions[-1].item()
        
        # 获取vision embeddings
        with torch.no_grad():
            vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(torch.bfloat16)
        
        # 获取text embeddings - 确保在同一设备
        text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        text_device = text_embeds.device
        vit_embeds = vit_embeds.to(text_device)  # 移动到相同设备
        
        # 替换IMG_CONTEXT位置
        B, N, C = text_embeds.shape
        text_embeds_flat = text_embeds.reshape(B * N, C).clone()
        input_ids_flat = input_ids.reshape(-1).to(text_device)
        img_positions = (input_ids_flat == self.model.img_context_token_id)
        
        if img_positions.sum() > 0:
            vit_flat = vit_embeds.reshape(-1, C)
            num_to_replace = min(img_positions.sum().item(), vit_flat.size(0))
            img_indices = img_positions.nonzero(as_tuple=True)[0][:num_to_replace]
            text_embeds_flat[img_indices] = vit_flat[:num_to_replace]
        
        input_embeds = text_embeds_flat.reshape(B, N, C)
        
        # Forward through LLM
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=True,
                return_dict=True,
            )
        
        # 提取[SEG]的hidden state
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
            seg_hidden = hidden_states[0, seg_pos, :]
            # 通过text_hidden_fcs
            seg_hidden = seg_hidden.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            language_embd = self.model.text_hidden_fcs(seg_hidden)  # [1, 1, 256]
            return language_embd
        
        return None
    
    def _predict_mask_with_grad(self, image, language_embd):
        """使用SAM2预测mask（带梯度）- 直接调用内部方法"""
        ori_size = image.size[::-1]  # (H, W)
        
        # 获取SAM2模块所在设备
        sam2 = self.model.grounding_encoder
        sam2_device = next(sam2.parameters()).device
        
        # 准备SAM2输入
        g_image = np.array(image)
        g_image = self.model.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(torch.bfloat16)
        g_pixel_values = sam2.preprocess_image(g_pixel_values).unsqueeze(0).to(sam2_device)
        
        # 确保language_embd在正确设备上
        language_embd = language_embd.to(sam2_device)
        
        # 直接使用SAM2的forward_sam_heads（绕过inference_state）
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 获取image features
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
        
        # Resize到原始尺寸
        h, w = ori_size
        masks = F.interpolate(low_res_masks, size=(h, w), mode='bilinear', align_corners=False)
        return masks.squeeze(1)  # [1, H, W]
    
    def train_step(self, sample):
        img_path = os.path.join(self.data_root, sample['image'])
        chosen_path = os.path.join(self.data_root, sample['chosen_mask'])
        rejected_path = os.path.join(self.data_root, sample['rejected_mask'])
        
        if not all(os.path.exists(p) for p in [img_path, chosen_path, rejected_path]):
            return None
        
        image = Image.open(img_path).convert('RGB')
        chosen_mask = (np.array(Image.open(chosen_path).convert('L')) > 127).astype(np.float32)
        rejected_mask = (np.array(Image.open(rejected_path).convert('L')) > 127).astype(np.float32)
        
        try:
            # 获取[SEG]的hidden state（完整LLM forward）
            language_embd = self._get_seg_hidden_state_full(image)
            
            if language_embd is None:
                return None
            
            # 预测mask（带梯度）
            pred_logits = self._predict_mask_with_grad(image, language_embd)
            
            # 移动masks到相同设备
            pred_device = pred_logits.device
            chosen_tensor = torch.from_numpy(chosen_mask).to(pred_device)
            rejected_tensor = torch.from_numpy(rejected_mask).to(pred_device)
            
            # DPO loss
            loss, metrics = dpo_loss(pred_logits.squeeze(), chosen_tensor, rejected_tensor, beta=self.beta)
            
            return {'loss': loss, **metrics}
        
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train(self):
        print("\n🚀 Starting DPO V4 training...")
        self.model.train()
        
        acc_loss, acc_dice, acc_prefer, acc_count = 0, 0, 0, 0
        global_step = 0
        
        pbar = tqdm(self.annotations, desc="DPO V4 Training")
        
        for idx, sample in enumerate(pbar):
            result = self.train_step(sample)
            
            if result is None:
                continue
            
            if not result['loss'].requires_grad:
                print(f"  Warning: Loss has no gradient at step {idx}")
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
        print("🎉 DPO V4 Training completed!")
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
        
        # Re-apply LoRA for continued training
        lora_config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_r * 2, lora_dropout=0.05,
            bias='none', task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)


if __name__ == '__main__':
    trainer = DPOTrainerV4()
    trainer.train()
