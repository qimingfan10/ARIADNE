"""
LoRA SFT训练脚本 - 使用ComboLoss进行监督微调
替代PPO强化学习方案
"""
import os
import sys
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image

sys.path.insert(0, '/home/ubuntu/Sa2VA/lora_sft_training')
from combo_loss import ComboLoss, calculate_metrics


class VesselSegmentationDataset(Dataset):
    """血管分割数据集"""
    def __init__(self, data_root, split='train', train_ratio=0.8):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, 'images')
        self.masks_dir = os.path.join(data_root, 'masks')
        
        # 获取所有图像
        import glob
        all_images = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))
        
        # 划分训练/验证集
        n_train = int(len(all_images) * train_ratio)
        if split == 'train':
            self.image_files = all_images[:n_train]
        else:
            self.image_files = all_images[n_train:]
        
        print(f"{split.capitalize()} set: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 加载mask
        img_name = os.path.basename(image_path)
        mask_name = img_name.replace('.jpg', '_mask.png').replace('.JPG', '_mask.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask).astype(np.float32) / 255.0  # 归一化到[0, 1]
        mask = (mask > 0.5).astype(np.float32)  # 二值化
        
        return {
            'image': image,
            'mask': mask,
            'image_name': img_name
        }


class LoRASFTTrainer:
    """LoRA SFT训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(args.output_dir, f'lora_sft_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        print(f"输出目录: {self.output_dir}")
        
    def setup_model(self):
        """设置模型和LoRA"""
        print("\n加载模型...")
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=True
        )
        
        # 配置LoRA - 更激进的参数
        lora_config = LoraConfig(
            r=self.args.lora_rank,  # 64或128
            lora_alpha=self.args.lora_alpha,  # 128或256
            target_modules=self.args.target_modules.split(','),
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # 启用梯度检查点（节省显存）
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("✅ 梯度检查点已启用")
        
        print("✅ 模型加载完成")
        
    def setup_data(self):
        """设置数据加载器"""
        print("\n准备数据...")
        
        # 创建数据集
        train_dataset = VesselSegmentationDataset(
            self.args.data_root,
            split='train',
            train_ratio=self.args.train_ratio
        )
        
        val_dataset = VesselSegmentationDataset(
            self.args.data_root,
            split='val',
            train_ratio=self.args.train_ratio
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        print(f"训练样本: {len(train_dataset)}")
        print(f"验证样本: {len(val_dataset)}")
        print(f"训练批次: {len(self.train_loader)}")
        
    def setup_training(self):
        """设置训练组件"""
        print("\n设置训练组件...")
        
        # 损失函数
        self.criterion = ComboLoss(
            weight_dice=self.args.weight_dice,
            weight_focal=self.args.weight_focal,
            weight_bce=self.args.weight_bce
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # 学习率调度器 - Cosine Annealing
        total_steps = len(self.train_loader) * self.args.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),  # 10% warmup
            num_training_steps=total_steps
        )
        
        print(f"优化器: AdamW (LR={self.args.learning_rate})")
        print(f"调度器: Cosine Annealing with Warmup")
        print(f"总步数: {total_steps}")
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        epoch_loss = 0
        epoch_dice = 0
        epoch_metrics = {'dice': [], 'recall': [], 'precision': []}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                image = batch['image']
                gt_mask = batch['mask'].to(self.device)  # (B, H, W)
                
                # 推理获取预测mask的logits
                result = self.model.predict_forward(
                    image=image[0] if len(image) == 1 else image,
                    text='<image>\nPlease segment the blood vessel.',
                    tokenizer=self.tokenizer
                )
                
                # 获取概率图
                if 'probability_maps' in result and len(result['probability_maps']) > 0:
                    pred_prob = result['probability_maps'][0][0]  # (H, W)
                    pred_prob = torch.from_numpy(pred_prob).to(self.device)
                    
                    # 转换回logits (inverse sigmoid)
                    pred_prob = torch.clamp(pred_prob, 1e-7, 1 - 1e-7)
                    pred_logits = torch.log(pred_prob / (1 - pred_prob))
                    
                    # 调整尺寸匹配GT
                    if pred_logits.shape != gt_mask.shape:
                        pred_logits = F.interpolate(
                            pred_logits.unsqueeze(0).unsqueeze(0),
                            size=gt_mask.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    
                    # 计算loss
                    self.optimizer.zero_grad()
                    loss, dice_score, loss_dict = self.criterion(pred_logits, gt_mask)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    # 统计
                    epoch_loss += loss.item()
                    epoch_dice += dice_score
                    
                    # 计算详细指标
                    metrics = calculate_metrics(torch.sigmoid(pred_logits).unsqueeze(0), gt_mask.unsqueeze(0))
                    for k in ['dice', 'recall', 'precision']:
                        epoch_metrics[k].append(metrics[k])
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'dice': f"{dice_score:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                else:
                    print(f"⚠️ 批次{batch_idx}未返回概率图")
                    
            except Exception as e:
                print(f"❌ 批次{batch_idx}训练失败: {e}")
                continue
        
        # 计算epoch平均指标
        n_batches = len(self.train_loader)
        avg_loss = epoch_loss / n_batches
        avg_dice = epoch_dice / n_batches
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return avg_loss, avg_dice, avg_metrics
    
    def validate(self):
        """验证"""
        self.model.eval()
        
        val_metrics = {'dice': [], 'recall': [], 'precision': []}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                    image = batch['image']
                    gt_mask = batch['mask'].to(self.device)
                    
                    result = self.model.predict_forward(
                        image=image[0] if len(image) == 1 else image,
                        text='<image>\nPlease segment the blood vessel.',
                        tokenizer=self.tokenizer
                    )
                    
                    if 'probability_maps' in result and len(result['probability_maps']) > 0:
                        pred_prob = result['probability_maps'][0][0]
                        pred_prob = torch.from_numpy(pred_prob).to(self.device)
                        
                        # 调整尺寸
                        if pred_prob.shape != gt_mask.shape:
                            pred_prob = F.interpolate(
                                pred_prob.unsqueeze(0).unsqueeze(0),
                                size=gt_mask.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                        
                        # 计算指标
                        metrics = calculate_metrics(pred_prob.unsqueeze(0), gt_mask.unsqueeze(0))
                        for k in ['dice', 'recall', 'precision']:
                            val_metrics[k].append(metrics[k])
                
                except Exception as e:
                    continue
        
        # 平均指标
        avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in val_metrics.items()}
        return avg_metrics
    
    def train(self):
        """主训练循环"""
        print("\n" + "="*80)
        print("开始训练")
        print("="*80)
        
        best_dice = 0.0
        training_log = []
        
        for epoch in range(self.args.num_epochs):
            # 训练
            train_loss, train_dice, train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate()
            
            # 日志
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'train_recall': train_metrics['recall'],
                'train_precision': train_metrics['precision'],
                'val_dice': val_metrics['dice'],
                'val_recall': val_metrics['recall'],
                'val_precision': val_metrics['precision']
            }
            training_log.append(log_entry)
            
            # 打印
            print(f"\nEpoch {epoch+1}/{self.args.num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, "
                  f"Recall: {train_metrics['recall']:.4f}, Precision: {train_metrics['precision']:.4f}")
            print(f"  Val   - Dice: {val_metrics['dice']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}, Precision: {val_metrics['precision']:.4f}")
            
            # 保存最佳模型
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ✅ 最佳模型已保存 (Dice: {best_dice:.4f})")
            
            # 定期保存
            if (epoch + 1) % self.args.save_every == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
        
        # 保存训练日志
        with open(os.path.join(self.output_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2)
        
        print("\n" + "="*80)
        print(f"训练完成！最佳验证Dice: {best_dice:.4f}")
        print(f"模型保存在: {self.output_dir}")
        print("="*80)
        
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        suffix = 'best' if is_best else f'epoch_{epoch+1}'
        save_path = os.path.join(self.output_dir, f'checkpoint_{suffix}')
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存训练状态
        state = {
            'epoch': epoch + 1,
            'metrics': metrics,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, os.path.join(save_path, 'training_state.pt'))


def parse_args():
    parser = argparse.ArgumentParser(description='LoRA SFT Training')
    
    # 模型
    parser.add_argument('--model_path', type=str, 
                        default='/home/ubuntu/Sa2VA/models/sa2va_vessel_hf',
                        help='Sa2VA模型路径')
    
    # 数据
    parser.add_argument('--data_root', type=str,
                        default='/home/ubuntu/Sa2VA/Segment_DATA_Merged_512',
                        help='数据集根目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    
    # LoRA配置
    parser.add_argument('--lora_rank', type=int, default=64,
                        help='LoRA rank (推荐64或128)')
    parser.add_argument('--lora_alpha', type=int, default=128,
                        help='LoRA alpha (通常是rank的2倍)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    parser.add_argument('--target_modules', type=str,
                        default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
                        help='LoRA目标模块（逗号分隔）')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪')
    
    # Loss权重
    parser.add_argument('--weight_dice', type=float, default=1.0,
                        help='Dice Loss权重')
    parser.add_argument('--weight_focal', type=float, default=1.0,
                        help='Focal Loss权重')
    parser.add_argument('--weight_bce', type=float, default=0.5,
                        help='BCE Loss权重')
    
    # 其他
    parser.add_argument('--output_dir', type=str, default='./output_sft',
                        help='输出目录')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='启用梯度检查点（节省显存）')
    parser.add_argument('--save_every', type=int, default=5,
                        help='每N个epoch保存一次')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU编号')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 创建训练器
    trainer = LoRASFTTrainer(args)
    
    # 设置模型
    trainer.setup_model()
    
    # 设置数据
    trainer.setup_data()
    
    # 设置训练组件
    trainer.setup_training()
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
