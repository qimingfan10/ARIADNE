#!/usr/bin/env python3
"""
使用XTuner/MMEngine框架进行Sa2VA 26B DPO训练
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/Sa2VA')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 设置环境
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def dice_score(pred, target):
    """计算Dice分数"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)


def dpo_loss_from_dice(pred_logits, chosen_mask, rejected_mask, beta=0.1):
    """基于Dice相似度的DPO损失"""
    pred_prob = torch.sigmoid(pred_logits).flatten()
    
    chosen_flat = chosen_mask.flatten().to(pred_prob.device).to(pred_prob.dtype)
    rejected_flat = rejected_mask.flatten().to(pred_prob.device).to(pred_prob.dtype)
    
    # Dice with chosen
    c_dice = dice_score(pred_prob, chosen_flat)
    log_prob_chosen = torch.log(c_dice + 1e-8)
    
    # Dice with rejected  
    r_dice = dice_score(pred_prob, rejected_flat)
    log_prob_rejected = torch.log(r_dice + 1e-8)
    
    # DPO loss
    logits = beta * (log_prob_chosen - log_prob_rejected)
    loss = -F.logsigmoid(logits)
    
    return loss, c_dice.item(), (log_prob_chosen > log_prob_rejected).float().item()


class XTunerDPOTrainer:
    def __init__(self):
        self.model_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
        self.output_dir = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_xtuner"
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        
        self.beta = 0.1
        self.lr = 1e-5
        self.lora_r = 16
        self.max_samples = 500
        self.grad_accum = 4
        self.save_steps = 100
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🎯 XTuner DPO Training for Sa2VA 26B")
        print("=" * 60)
        
        self._load_model()
        self._load_data()
        self._setup_training()
    
    def _load_model(self):
        print("\n📥 Loading Sa2VA 26B model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        
        # 初始化模型
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
        self.seg_token_id = self.model.seg_token_idx
        
        print(f"✅ Model loaded, [SEG] token id: {self.seg_token_id}")
        
        # 应用LoRA到language_model
        print("🔧 Applying LoRA to language model...")
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_r * 2,
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        
        # 冻结vision，训练SAM2和text_hidden_fcs
        self.model.vision_model.requires_grad_(False)
        
        for p in self.model.grounding_encoder.parameters():
            p.requires_grad = True
        for p in self.model.text_hidden_fcs.parameters():
            p.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _load_data(self):
        print("\n📊 Loading DPO data...")
        with open(f"{self.data_root}/dpo_annotations.json") as f:
            data = json.load(f)
        
        self.annotations = []
        for item in data:
            if 'chosen_mask' in item and 'rejected_mask' in item:
                self.annotations.append(item)
        
        self.annotations = self.annotations[:self.max_samples]
        print(f"   Loaded {len(self.annotations)} DPO pairs")
    
    def _setup_training(self):
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=0.01,
        )
        
        total_steps = len(self.annotations) // self.grad_accum
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7
        )
    
    def _predict_mask(self, image, prompt="Please segment the blood vessels."):
        """使用模型预测mask"""
        result = self.model.predict_forward(
            image=image,
            text=prompt,
            tokenizer=self.tokenizer,
        )
        
        if result and 'prediction_masks' in result and len(result['prediction_masks']) > 0:
            # prediction_masks是list，每个元素是一个mask
            mask = result['prediction_masks'][0]
            if isinstance(mask, np.ndarray):
                return torch.from_numpy(mask).float()
            return mask.float()
        return None
    
    def _get_seg_embedding_simplified(self, image):
        """
        获取简化的[SEG] embedding
        注意：这是简化版本，不使用完整的LLM forward
        """
        # 获取[SEG] token的embedding
        seg_embedding = self.model.language_model.get_input_embeddings()(
            torch.tensor([self.seg_token_id]).to(next(self.model.language_model.parameters()).device)
        )
        
        # 通过text_hidden_fcs
        seg_embedding = self.model.text_hidden_fcs(seg_embedding)
        
        return seg_embedding
    
    def _predict_mask_with_grad(self, image, seg_embedding):
        """使用seg_embedding预测mask（带梯度）"""
        # 准备SAM2输入
        sam2 = self.model.grounding_encoder
        sam2_device = next(sam2.parameters()).device
        
        g_image = np.array(image)
        g_image = self.model.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
        g_pixel_values = g_pixel_values.to(torch.bfloat16).to(sam2_device)
        g_pixel_values = sam2.preprocess_image(g_pixel_values).unsqueeze(0)
        
        # 确保embedding在正确设备
        seg_embedding = seg_embedding.to(sam2_device)
        if seg_embedding.dim() == 2:
            seg_embedding = seg_embedding.unsqueeze(1)
        
        # 获取SAM2特征
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = sam2.sam2_model.forward_image(g_pixel_values)
            _, vision_feats, _, feat_sizes = sam2.sam2_model._prepare_backbone_features(feats)
        
        # 准备高分辨率特征
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
            pix_feat = F.interpolate(pix_feat, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
            high_res_features = [
                F.interpolate(feat, size=(feat.size(2) * expected_size // H, feat.size(3) * expected_size // W),
                              mode='bilinear', align_corners=False)
                for feat in high_res_features
            ]
        
        # 预测mask
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, _, low_res_masks, _, _, _ = sam2.sam2_model._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=False,
                language_embd=seg_embedding,
            )
        
        # Resize到原始尺寸
        h, w = image.size[::-1]
        masks = F.interpolate(low_res_masks, size=(h, w), mode='bilinear', align_corners=False)
        return masks.squeeze(1)
    
    def train_step(self, sample):
        """执行一个训练步骤"""
        img_path = os.path.join(self.data_root, sample['image'])
        chosen_path = os.path.join(self.data_root, sample['chosen_mask'])
        rejected_path = os.path.join(self.data_root, sample['rejected_mask'])
        
        if not all(os.path.exists(p) for p in [img_path, chosen_path, rejected_path]):
            return None
        
        # 加载数据
        image = Image.open(img_path).convert('RGB')
        chosen_mask = torch.from_numpy(
            (np.array(Image.open(chosen_path).convert('L')) > 127).astype(np.float32)
        )
        rejected_mask = torch.from_numpy(
            (np.array(Image.open(rejected_path).convert('L')) > 127).astype(np.float32)
        )
        
        try:
            # 获取[SEG] embedding
            seg_embedding = self._get_seg_embedding_simplified(image)
            
            # 预测mask（带梯度）
            pred_logits = self._predict_mask_with_grad(image, seg_embedding)
            
            # 计算DPO损失
            loss, dice, prefer = dpo_loss_from_dice(
                pred_logits.squeeze(), chosen_mask, rejected_mask, beta=self.beta
            )
            
            return {'loss': loss, 'dice': dice, 'prefer': prefer}
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train(self):
        print("\n🚀 Starting XTuner DPO training...")
        self.model.train()
        
        acc_loss, acc_dice, acc_prefer, acc_count = 0, 0, 0, 0
        global_step = 0
        
        pbar = tqdm(self.annotations, desc="XTuner DPO")
        
        for idx, sample in enumerate(pbar):
            result = self.train_step(sample)
            
            if result is None:
                continue
            
            loss = result['loss'] / self.grad_accum
            
            if loss.requires_grad:
                loss.backward()
            
            acc_loss += result['loss'].item()
            acc_dice += result['dice']
            acc_prefer += result['prefer']
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
                
                if global_step % self.save_steps == 0:
                    self._save(f'step_{global_step}')
        
        self._save('final')
        print("\n" + "=" * 60)
        print("🎉 XTuner DPO Training completed!")
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
            r=self.lora_r,
            lora_alpha=self.lora_r * 2,
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)


if __name__ == '__main__':
    trainer = XTunerDPOTrainer()
    trainer.train()
