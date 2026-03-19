#!/usr/bin/env python3
"""
Sa2VA 26B 安全DPO训练 - 只训练LLM LoRA，不修改SAM2

关键：不修改SAM2组件，保持模型推理能力
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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def dice_loss(pred_prob, target):
    """Dice损失"""
    pred_flat = pred_prob.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)
    return 1 - dice, dice.item()


class SafeDPOTrainer:
    """安全DPO训练器 - 只训练LLM LoRA"""
    
    def __init__(self):
        self.model_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
        self.output_dir = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_safe"
        self.data_root = "/home/ubuntu/Sa2VA/data/dpo_vessel"
        
        self.lr = 5e-6
        self.lora_r = 16
        self.max_samples = 500
        self.grad_accum = 4
        self.save_steps = 100
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("🔒 Safe DPO Training - LLM LoRA Only")
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
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        
        self.model.preparing_for_generation(tokenizer=self.tokenizer)
        print(f"✅ Model loaded")
        
        # 只对LLM应用LoRA
        print("🔧 Applying LoRA to LLM only...")
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_r * 2,
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        
        # 冻结所有其他组件
        self.model.vision_model.requires_grad_(False)
        self.model.grounding_encoder.requires_grad_(False)
        self.model.text_hidden_fcs.requires_grad_(False)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print("   Only LLM LoRA is trainable, SAM2 is frozen")
    
    def _load_data(self):
        print("\n📊 Loading data...")
        with open(f"{self.data_root}/dpo_annotations.json") as f:
            data = json.load(f)
        
        self.annotations = [
            a for a in data 
            if 'chosen_mask' in a and 'rejected_mask' in a
        ][:self.max_samples]
        print(f"   {len(self.annotations)} DPO pairs")
    
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
    
    def _predict_and_compute_loss(self, image, chosen_mask, rejected_mask):
        """
        使用模型推理并计算DPO风格的损失
        
        注意：由于SAM2使用@torch.no_grad，我们无法直接对mask预测进行梯度更新
        但我们可以通过LLM的生成质量间接优化
        """
        # 使用模型预测
        with torch.enable_grad():
            result = self.model.predict_forward(
                image=image,
                text="Please segment the blood vessels.",
                tokenizer=self.tokenizer,
            )
        
        if not result or 'prediction_masks' not in result or len(result['prediction_masks']) == 0:
            return None, 0, 0
        
        pred_mask = result['prediction_masks'][0]
        if isinstance(pred_mask, np.ndarray):
            pred_mask = torch.from_numpy(pred_mask)
        
        # 计算与chosen和rejected的Dice相似度
        pred_binary = (pred_mask > 0.5).float()
        
        chosen_t = torch.from_numpy(chosen_mask).float()
        rejected_t = torch.from_numpy(rejected_mask).float()
        
        # Dice with chosen
        c_inter = (pred_binary * chosen_t).sum()
        c_dice = (2. * c_inter) / (pred_binary.sum() + chosen_t.sum() + 1e-8)
        
        # Dice with rejected
        r_inter = (pred_binary * rejected_t).sum()
        r_dice = (2. * r_inter) / (pred_binary.sum() + rejected_t.sum() + 1e-8)
        
        prefer = (c_dice > r_dice).float().item()
        
        # 由于无法获取梯度，我们使用监督学习方式
        # 计算与chosen mask的损失
        if 'probability_maps' in result and len(result['probability_maps']) > 0:
            prob_map = result['probability_maps'][0]
            if isinstance(prob_map, np.ndarray):
                prob_map = torch.from_numpy(prob_map)
            
            # BCE损失
            chosen_t = chosen_t.to(prob_map.device)
            loss = F.binary_cross_entropy(prob_map.float(), chosen_t, reduction='mean')
            return loss, c_dice.item(), prefer
        
        return None, c_dice.item(), prefer
    
    def train_step(self, sample):
        """训练步骤"""
        img_path = os.path.join(self.data_root, sample['image'])
        chosen_path = os.path.join(self.data_root, sample['chosen_mask'])
        rejected_path = os.path.join(self.data_root, sample['rejected_mask'])
        
        if not all(os.path.exists(p) for p in [img_path, chosen_path, rejected_path]):
            return None
        
        image = Image.open(img_path).convert('RGB')
        chosen_mask = (np.array(Image.open(chosen_path).convert('L')) > 127).astype(np.float32)
        rejected_mask = (np.array(Image.open(rejected_path).convert('L')) > 127).astype(np.float32)
        
        try:
            loss, dice, prefer = self._predict_and_compute_loss(image, chosen_mask, rejected_mask)
            
            if loss is None:
                return None
            
            return {'loss': loss, 'dice': dice, 'prefer': prefer}
            
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train(self):
        print("\n🚀 Starting Safe DPO training...")
        self.model.train()
        
        acc_loss, acc_dice, acc_prefer, acc_count = 0, 0, 0, 0
        global_step = 0
        
        pbar = tqdm(self.annotations, desc="Safe DPO")
        
        for idx, sample in enumerate(pbar):
            result = self.train_step(sample)
            
            if result is None:
                continue
            
            loss = result['loss'] / self.grad_accum
            
            if hasattr(loss, 'backward'):
                loss.backward()
            
            acc_loss += result['loss'].item() if hasattr(result['loss'], 'item') else result['loss']
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
        print("🎉 Safe DPO Training completed!")
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
    trainer = SafeDPOTrainer()
    trainer.train()
