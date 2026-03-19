#!/usr/bin/env python3
"""
LoRA SFT训练脚本 - 简化版
"""
import os
import sys
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image

from combo_loss import ComboLoss, calculate_metrics


class VesselDataset(Dataset):
    def __init__(self, data_root, split='train', train_ratio=0.8):
        self.data_root = data_root
        images_dir = os.path.join(data_root, 'images')
        self.masks_dir = os.path.join(data_root, 'masks')
        
        all_images = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        n_train = int(len(all_images) * train_ratio)
        
        if split == 'train':
            self.image_files = all_images[:n_train]
        else:
            self.image_files = all_images[n_train:]
        
        print(f"{split.capitalize()} set: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        img_name = os.path.basename(image_path)
        mask_name = img_name.replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        mask = np.array(Image.open(mask_path).convert('L')).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        return {'image': image, 'mask': mask, 'name': img_name}


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(args.output_dir, f'sft_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Output: {self.output_dir}")
    
    def setup_model(self):
        print("\nLoading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path, trust_remote_code=True
        )
        
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.args.target_modules.split(','),
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("✅ Model loaded")
    
    def collate_fn(self, batch):
        # 自定义collate：保持PIL Image原样
        return batch[0]  # batch_size=1，直接返回第一个元素
    
    def setup_data(self):
        print("\nPreparing data...")
        train_dataset = VesselDataset(self.args.data_root, 'train', self.args.train_ratio)
        val_dataset = VesselDataset(self.args.data_root, 'val', self.args.train_ratio)
        
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=self.collate_fn)
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
    
    def setup_training(self):
        print("\nSetting up training...")
        self.criterion = ComboLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        
        total_steps = len(self.train_loader) * self.args.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps
        )
        print(f"Optimizer: AdamW (LR={self.args.lr})")
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                image = batch['image']
                gt_mask = torch.from_numpy(batch['mask']).to(self.device).float()
                
                result = self.model.predict_forward(
                    image=image,
                    text='<image>\nPlease segment the blood vessel.',
                    tokenizer=self.tokenizer,
                    return_tensors=True  # 返回tensor以保持梯度
                )
                
                if 'probability_maps' in result and len(result['probability_maps']) > 0:
                    pred_prob = result['probability_maps'][0][0]
                    
                    # 确保pred_prob有梯度
                    if not pred_prob.requires_grad:
                        # 如果没有梯度，手动设置requires_grad（虽然这不是最佳方案）
                        pred_prob = pred_prob.detach().requires_grad_(True)
                    
                    pred_prob = pred_prob.to(self.device)
                    
                    if pred_prob.shape != gt_mask.shape:
                        pred_prob = F.interpolate(
                            pred_prob.unsqueeze(0).unsqueeze(0),
                            size=gt_mask.shape, mode='bilinear'
                        ).squeeze()
                    
                    # 注意：pred_prob已经是sigmoid后的概率，需要转回logits
                    # logit = log(p / (1-p))
                    pred_prob_clamped = torch.clamp(pred_prob, 1e-7, 1-1e-7)
                    pred_logits = torch.log(pred_prob_clamped / (1 - pred_prob_clamped))
                    
                    self.optimizer.zero_grad()
                    loss, dice_score, _ = self.criterion(pred_logits, gt_mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    epoch_loss += loss.item()
                    epoch_dice += dice_score
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'dice': f"{dice_score:.4f}"
                    })
            except Exception as e:
                print(f"\nBatch {batch_idx} error: {e}")
                continue
        
        n = len(self.train_loader)
        return epoch_loss/n, epoch_dice/n
    
    def validate(self):
        self.model.eval()
        val_dice = []
        val_recall = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                    image = batch['image']
                    gt_mask = torch.from_numpy(batch['mask']).to(self.device).float()
                    
                    result = self.model.predict_forward(
                        image=image,
                        text='<image>\nPlease segment the blood vessel.',
                        tokenizer=self.tokenizer
                    )
                    
                    if 'probability_maps' in result and len(result['probability_maps']) > 0:
                        pred_prob = torch.from_numpy(result['probability_maps'][0][0]).to(self.device)
                        
                        if pred_prob.shape != gt_mask.shape:
                            pred_prob = F.interpolate(
                                pred_prob.unsqueeze(0).unsqueeze(0),
                                size=gt_mask.shape, mode='bilinear'
                            ).squeeze()
                        
                        metrics = calculate_metrics(pred_prob.unsqueeze(0), gt_mask.unsqueeze(0))
                        val_dice.append(metrics['dice'])
                        val_recall.append(metrics['recall'])
                except:
                    continue
        
        return np.mean(val_dice) if val_dice else 0.0, np.mean(val_recall) if val_recall else 0.0
    
    def train(self):
        print("\n" + "="*80)
        print("Start Training")
        print("="*80)
        
        best_dice = 0.0
        
        for epoch in range(self.args.epochs):
            train_loss, train_dice = self.train_epoch(epoch)
            val_dice, val_recall = self.validate()
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"  Val   - Dice: {val_dice:.4f}, Recall: {val_recall:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                save_path = os.path.join(self.output_dir, 'best_model')
                self.model.save_pretrained(save_path)
                print(f"  ✅ Best model saved (Dice: {best_dice:.4f})")
        
        print("\n" + "="*80)
        print(f"Training complete! Best Dice: {best_dice:.4f}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/ubuntu/Sa2VA/models/sa2va_vessel_hf')
    parser.add_argument('--data_root', default='/home/ubuntu/Sa2VA/Segment_DATA_Merged_512')
    parser.add_argument('--output_dir', default='./output_sft')
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--target_modules', default='q_proj,k_proj,v_proj,o_proj')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--gpu', default='3')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    trainer = Trainer(args)
    trainer.setup_model()
    trainer.setup_data()
    trainer.setup_training()
    trainer.train()


if __name__ == '__main__':
    main()
