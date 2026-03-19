#!/usr/bin/env python3
"""
最终版LoRA SFT训练 - 修复梯度回传问题
通过移除generate的@torch.no_grad()，确保梯度能够正确回传
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
        self.output_dir = os.path.join(args.output_dir, f'sft_final_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Output: {self.output_dir}")
    
    def collate_fn(self, batch):
        return batch[0]
    
    def setup_model(self):
        print("\n" + "="*80)
        print("Loading Model")
        print("="*80)
        
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
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("✅ Model loaded with LoRA")
    
    def setup_data(self):
        print("\n" + "="*80)
        print("Preparing Data")
        print("="*80)
        
        train_dataset = VesselDataset(self.args.data_root, 'train', self.args.train_ratio)
        val_dataset = VesselDataset(self.args.data_root, 'val', self.args.train_ratio)
        
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=self.collate_fn)
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
    
    def setup_training(self):
        print("\n" + "="*80)
        print("Setup Training")
        print("="*80)
        
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
        valid_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                image = batch['image']
                gt_mask = torch.from_numpy(batch['mask']).to(self.device).float()
                
                # 现在generate没有@torch.no_grad()，梯度可以回传
                result = self.model.predict_forward(
                    image=image,
                    text='<image>\nPlease segment the blood vessel.',
                    tokenizer=self.tokenizer,
                    return_tensors=True
                )
                
                if 'probability_maps' in result and len(result['probability_maps']) > 0:
                    pred_prob = result['probability_maps'][0][0].to(self.device)
                    
                    if pred_prob.shape != gt_mask.shape:
                        pred_prob = F.interpolate(
                            pred_prob.unsqueeze(0).unsqueeze(0),
                            size=gt_mask.shape, mode='bilinear'
                        ).squeeze()
                    
                    # 转换为logits
                    pred_prob_clamped = torch.clamp(pred_prob, 1e-7, 1-1e-7)
                    pred_logits = torch.log(pred_prob_clamped / (1 - pred_prob_clamped))
                    
                    # 计算loss
                    self.optimizer.zero_grad()
                    loss, dice_score, _ = self.criterion(pred_logits, gt_mask)
                    
                    # 检查梯度
                    if loss.requires_grad:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                        
                        epoch_loss += loss.item()
                        epoch_dice += dice_score
                        valid_batches += 1
                        
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'dice': f"{dice_score:.4f}"
                        })
                    else:
                        print(f"\n⚠️ Batch {batch_idx}: Loss has no gradient!")
                        
            except Exception as e:
                print(f"\nBatch {batch_idx} error: {e}")
                continue
        
        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
        avg_dice = epoch_dice / valid_batches if valid_batches > 0 else 0
        return avg_loss, avg_dice
    
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
        print("START TRAINING")
        print("="*80)
        
        best_dice = 0.0
        history = []
        
        for epoch in range(self.args.epochs):
            train_loss, train_dice = self.train_epoch(epoch)
            val_dice, val_recall = self.validate()
            
            result = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_dice': float(train_dice),
                'val_dice': float(val_dice),
                'val_recall': float(val_recall)
            }
            history.append(result)
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"  Val   - Dice: {val_dice:.4f}, Recall: {val_recall:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                save_path = os.path.join(self.output_dir, 'best_model')
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"  ✅ Best model saved (Dice: {best_dice:.4f})")
            
            # 保存历史
            with open(os.path.join(self.output_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)
        
        print("\n" + "="*80)
        print(f"TRAINING COMPLETE! Best Dice: {best_dice:.4f}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/ubuntu/Sa2VA/models/sa2va_vessel_hf')
    parser.add_argument('--data_root', default='/home/ubuntu/Sa2VA/Segment_DATA_Merged_512')
    parser.add_argument('--output_dir', default='./output_sft')
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
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
