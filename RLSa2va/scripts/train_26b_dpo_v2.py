#!/usr/bin/env python3
"""
Sa2VA 26B DPO训练 V2 - 使用完整LLM前向传播

关键改进：使用完整的LLM forward获取上下文相关的[SEG] hidden states
而不是简化的固定embedding方法
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
    """计算预测mask与目标mask的log概率"""
    pred_prob = torch.sigmoid(pred_logits).flatten()
    target = target_mask.flatten()
    
    intersection = (pred_prob * target).sum()
    dice = (2. * intersection + 1.0) / (pred_prob.sum() + target.sum() + 1.0)
    log_prob = torch.log(dice + 1e-8)
    
    return log_prob


def dpo_loss(pred_logits, chosen_mask, rejected_mask, beta=0.1):
    """DPO损失函数"""
    log_prob_chosen = compute_mask_log_prob(pred_logits, chosen_mask)
    log_prob_rejected = compute_mask_log_prob(pred_logits, rejected_mask)
    
    logits = beta * (log_prob_chosen - log_prob_rejected)
    loss = -F.logsigmoid(logits)
    
    with torch.no_grad():
        pred_binary = (torch.sigmoid(pred_logits) > 0.5).float().flatten()
        chosen_flat = chosen_mask.flatten()
        chosen_inter = (pred_binary * chosen_flat).sum()
        chosen_dice = (2 * chosen_inter / (pred_binary.sum() + chosen_flat.sum() + 1e-8)).item()
        prefer_chosen = (log_prob_chosen > log_prob_rejected).float().item()
    
    return loss, {'chosen_dice': chosen_dice, 'prefer_chosen': prefer_chosen}


def get_seg_hidden_states(hidden_states, seg_positions):
    """从hidden states中提取[SEG] token的embedding"""
    # hidden_states: [batch, seq_len, hidden_dim]
    # seg_positions: list of positions
    batch_size = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)
    
    seg_embeddings = []
    for b in range(batch_size):
        if seg_positions[b] is not None and seg_positions[b] < hidden_states.size(1):
            seg_emb = hidden_states[b, seg_positions[b], :]  # [hidden_dim]
        else:
            # 如果没有找到[SEG]，使用最后一个token
            seg_emb = hidden_states[b, -1, :]
        seg_embeddings.append(seg_emb)
    
    return torch.stack(seg_embeddings, dim=0)  # [batch, hidden_dim]


def forward_sam_with_grad(model, g_pixel_values, language_embd, ori_size):
    """带梯度的SAM2前向传播"""
    sam2 = model.grounding_encoder
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        image_features = sam2.sam2_model.forward_image(g_pixel_values)
        _, vision_feats, vision_pos_embeds, feat_sizes = sam2.sam2_model._prepare_backbone_features(image_features)
        
        B = vision_feats[-1].size(1)
        C = sam2.sam2_model.hidden_dim
        H, W = feat_sizes[-1]
        
        pix_feat = vision_feats[-1] + sam2.sam2_model.no_mem_embed
        pix_feat = pix_feat.permute(1, 2, 0).view(B, C, H, W)
        
        expected_size = sam2.sam2_model.sam_image_embedding_size
        if H != expected_size or W != expected_size:
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
        
        _, _, _, low_res_masks, high_res_masks, obj_ptr, _ = sam2.sam2_model._forward_sam_heads(
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


class Sa2VA_DPO_Trainer_V2:
    """Sa2VA 26B DPO训练器 V2 - 使用完整LLM前向传播"""
    
    def __init__(
        self,
        model_path="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf",
        output_dir="/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v2",
        learning_rate=2e-5,
        lora_r=16,
        beta=0.1,
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
        print("🎯 Sa2VA 26B DPO Training V2")
        print("   使用完整LLM前向传播获取上下文相关的[SEG] embedding")
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
        
        # 初始化模型（设置seg_token_idx等）
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
        
        # 获取[SEG] token id
        self.seg_token = '[SEG]'
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids('[SEG]')
        print(f"   [SEG] token id: {self.seg_token_id}")
        
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
        
        # SAM2和text_hidden_fcs保持可训练
        for param in self.model.grounding_encoder.parameters():
            param.requires_grad = True
        for param in self.model.text_hidden_fcs.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    
    def _load_data(self):
        print("\n📊 Loading DPO data...")
        
        data_path = "/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_annotations.json"
        with open(data_path) as f:
            self.annotations = json.load(f)
        
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
    
    def _prepare_inputs(self, image, text):
        """准备模型输入，返回input_ids和image处理结果"""
        device = next(self.model.parameters()).device
        
        # 使用模型的transformer处理图像
        pixel_values = self.model.transformer(image).unsqueeze(0)
        pixel_values = pixel_values.to(dtype=torch.bfloat16, device=device)
        
        # 构建prompt - 包含[SEG] token
        prompt = f"Please segment the blood vessels in this image. {self.seg_token}"
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids']
        input_ids = input_ids.to(next(self.model.parameters()).device)
        
        # 找[SEG] token位置
        seg_positions = []
        for i in range(input_ids.size(0)):
            pos = (input_ids[i] == self.seg_token_id).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                seg_positions.append(pos[-1].item())  # 取最后一个[SEG]
            else:
                seg_positions.append(None)
        
        return input_ids, pixel_values, seg_positions
    
    def train_step(self, sample):
        """DPO训练步骤 - 使用完整LLM前向传播"""
        img_path = os.path.join(self.data_root, sample['image'])
        chosen_path = os.path.join(self.data_root, sample['chosen_mask'])
        rejected_path = os.path.join(self.data_root, sample['rejected_mask'])
        
        if not all(os.path.exists(p) for p in [img_path, chosen_path, rejected_path]):
            return None
        
        # 加载数据
        image = Image.open(img_path).convert('RGB')
        chosen_mask = (np.array(Image.open(chosen_path).convert('L')) > 127).astype(np.float32)
        rejected_mask = (np.array(Image.open(rejected_path).convert('L')) > 127).astype(np.float32)
        
        ori_size = chosen_mask.shape
        device = next(self.model.parameters()).device
        
        chosen_tensor = torch.from_numpy(chosen_mask).to(device)
        rejected_tensor = torch.from_numpy(rejected_mask).to(device)
        
        try:
            # 1. 准备输入
            input_ids, pixel_values, seg_positions = self._prepare_inputs(image, "")
            
            # 2. 获取vision embeddings
            with torch.no_grad():
                vision_outputs = self.model.vision_model(pixel_values)
                vision_embeds = vision_outputs.last_hidden_state
                vision_embeds = self.model.mlp1(vision_embeds)
            
            # 3. 获取text embeddings
            text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            
            # 4. 替换<image>位置为vision embeddings
            # 找到img_context_token的位置
            img_token_id = self.model.img_context_token_id if hasattr(self.model, 'img_context_token_id') else None
            
            # 简化处理：直接拼接vision和text embeddings
            # [vision_embeds, text_embeds]
            batch_size = text_embeds.size(0)
            vision_seq_len = vision_embeds.size(1)
            
            # 调整seg_positions以考虑vision embedding的偏移
            adjusted_seg_positions = []
            for pos in seg_positions:
                if pos is not None:
                    adjusted_seg_positions.append(pos + vision_seq_len)
                else:
                    adjusted_seg_positions.append(None)
            
            # 拼接embeddings
            combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
            
            # 5. LLM前向传播获取hidden states
            outputs = self.model.language_model(
                inputs_embeds=combined_embeds,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # 获取最后一层的hidden states
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
            
            # 6. 提取[SEG] token的hidden state
            seg_hidden = get_seg_hidden_states(hidden_states, adjusted_seg_positions)  # [batch, hidden_dim]
            
            # 7. 通过text_hidden_fcs
            seg_hidden = seg_hidden.unsqueeze(1)  # [batch, 1, hidden_dim]
            language_embd = self.model.text_hidden_fcs(seg_hidden)  # [batch, 1, sam_dim]
            
            # 8. 准备SAM2输入图像
            g_image = np.array(image)
            g_image = self.model.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(torch.bfloat16)
            g_pixel_values = self.model.grounding_encoder.preprocess_image(g_pixel_values).unsqueeze(0).to(device)
            
            # 9. SAM2前向传播获取mask
            pred_logits = forward_sam_with_grad(self.model, g_pixel_values, language_embd, ori_size)
            
            # 10. 计算DPO损失
            loss, metrics = dpo_loss(pred_logits.squeeze(), chosen_tensor, rejected_tensor, beta=self.beta)
            
            return {'loss': loss, **metrics}
        
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train(self):
        print("\n🚀 Starting DPO training V2...")
        print("   使用完整LLM forward获取上下文相关的[SEG] hidden states")
        
        self.model.train()
        global_step = 0
        
        for epoch in range(self.num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.num_epochs}")
            
            acc_loss = 0
            acc_dice = 0
            acc_prefer = 0
            acc_count = 0
            
            pbar = tqdm(self.annotations, desc="DPO Training V2")
            
            for idx, sample in enumerate(pbar):
                result = self.train_step(sample)
                
                if result is None:
                    continue
                
                loss = result['loss']
                
                if not loss.requires_grad:
                    continue
                
                scaled_loss = loss / self.gradient_accumulation
                scaled_loss.backward()
                
                acc_loss += loss.item()
                acc_dice += result['chosen_dice']
                acc_prefer += result['prefer_chosen']
                acc_count += 1
                
                if (idx + 1) % self.gradient_accumulation == 0:
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
                            'prefer': f'{acc_prefer/acc_count:.2%}',
                        })
                    
                    acc_loss = 0
                    acc_dice = 0
                    acc_prefer = 0
                    acc_count = 0
                
                if global_step > 0 and global_step % 30 == 0:
                    self._save(f'step_{global_step}')
        
        self._save('final')
        
        print("\n" + "=" * 60)
        print("🎉 DPO Training V2 completed!")
        print(f"   Model saved to: {self.output_dir}")
        print("=" * 60)
    
    def _save(self, name):
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n💾 Saving to {save_dir}...")
        
        self.model.language_model = self.model.language_model.merge_and_unload()
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        lora_config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_r * 2, lora_dropout=0.05, bias='none',
            task_type=TaskType.CAUSAL_LM, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)


def main():
    trainer = Sa2VA_DPO_Trainer_V2(
        beta=0.1,
        max_samples=500,
        num_epochs=1,
    )
    trainer.train()


if __name__ == '__main__':
    main()
