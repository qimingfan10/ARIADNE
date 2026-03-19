"""
训练Reward Network
第一步：生成训练数据并训练奖励网络
"""
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import cv2

sys.path.insert(0, '/home/ubuntu/Sa2VA')
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl_reward_network.models import LightweightRewardNetwork, calculate_dice_score


class RewardDataset(Dataset):
    """Reward Network训练数据集"""
    
    def __init__(self, data_root, model, tokenizer, max_samples=None, device='cuda'):
        self.data_root = data_root
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 加载数据
        with open(os.path.join(data_root, 'annotations.json')) as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        print(f"生成Reward训练数据集，共{len(self.data)}个样本...")
        self.samples = self._generate_samples()
    
    def _generate_samples(self):
        """生成训练样本：使用Sa2VA预测并计算Dice作为标签"""
        samples = []
        
        for idx, item in enumerate(tqdm(self.data)):
            try:
                # 加载图像
                img_path = os.path.join(self.data_root, 'images', item['image'])
                image = Image.open(img_path).convert('RGB')
                image_np = np.array(image)
                h, w = image_np.shape[:2]
                
                # 创建GT mask
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                for mask_coords in item['mask']:
                    if len(mask_coords) >= 6:
                        points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(gt_mask, [points], 255)
                
                # 使用Sa2VA预测
                result = self.model.predict_forward(
                    image=image,
                    text="<image>Please segment the blood vessel.",
                    tokenizer=self.tokenizer,
                    processor=None,
                )
                
                if '[SEG]' in result.get('prediction', '') and 'prediction_masks' in result:
                    pred_masks = result['prediction_masks']
                    if len(pred_masks) > 0:
                        pred_mask = pred_masks[0][0]
                        
                        if isinstance(pred_mask, torch.Tensor):
                            pred_mask = pred_mask.cpu().numpy()
                        
                        if pred_mask.shape != (h, w):
                            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        if pred_mask.max() <= 1.0:
                            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                        else:
                            pred_mask = (pred_mask > 127).astype(np.uint8) * 255
                        
                        # 计算Dice作为质量标签
                        pred_flat = (pred_mask > 127).flatten().astype(int)
                        gt_flat = (gt_mask > 127).flatten().astype(int)
                        
                        intersection = np.sum(pred_flat * gt_flat)
                        union = np.sum(pred_flat) + np.sum(gt_flat)
                        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
                        
                        samples.append({
                            'image': image_np,
                            'mask': pred_mask,
                            'quality': float(dice)
                        })
                
            except Exception as e:
                print(f"处理样本{idx}失败: {e}")
                continue
        
        print(f"成功生成{len(samples)}个训练样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 归一化图像
        image = torch.from_numpy(sample['image']).float() / 255.0  # [H,W,3]
        image = image.permute(2, 0, 1)  # [3,H,W]
        
        # 归一化mask
        mask = torch.from_numpy(sample['mask']).float() / 255.0  # [H,W]
        mask = mask.unsqueeze(0)  # [1,H,W]
        
        # 质量标签
        quality = torch.tensor([sample['quality']], dtype=torch.float32)
        
        # Resize到固定大小
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(256, 256), mode='bilinear'
        ).squeeze(0)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(256, 256), mode='nearest'
        ).squeeze(0)
        
        return image, mask, quality


def train_reward_network(args):
    """训练Reward Network"""
    print("=" * 80)
    print("实验三：Reward Network训练 - 步骤1/2")
    print("=" * 80)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"reward_net_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载Sa2VA模型（用于生成训练数据）
    print("\n加载Sa2VA模型...")
    sa2va_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",  # 让transformers自动分配
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    sa2va_model.eval()
    print("✅ Sa2VA模型加载成功")
    
    # 生成训练数据
    print("\n生成Reward训练数据...")
    train_dataset = RewardDataset(
        args.data_root,
        sa2va_model,
        tokenizer,
        max_samples=args.max_samples,
        device=device
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # 创建Reward Network
    print("\n初始化Reward Network...")
    reward_net = LightweightRewardNetwork(input_channels=4).to(device)
    print(f"✅ Reward Network初始化完成，参数量: {sum(p.numel() for p in reward_net.parameters())}")
    
    # 优化器和损失
    optimizer = optim.Adam(reward_net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # 训练
    print("\n" + "=" * 80)
    print("开始训练Reward Network...")
    print("=" * 80)
    
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        reward_net.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (images, masks, qualities) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            qualities = qualities.to(device)
            
            # 前向传播
            pred_qualities = reward_net(images, masks)
            loss = criterion(pred_qualities, qualities)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({'loss': loss.item()})
            writer.add_scalar('Loss/train', loss.item(), global_step)
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: 平均损失 = {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': reward_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(output_dir, 'best_reward_net.pth'))
            print(f"  ✅ 保存最佳模型 (loss={best_loss:.4f})")
    
    # 保存最终模型
    torch.save(reward_net.state_dict(), os.path.join(output_dir, 'final_reward_net.pth'))
    
    writer.close()
    print(f"\n✅ 训练完成！模型保存至: {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="训练Reward Network - 实验三步骤1")
    
    parser.add_argument('--model_path', type=str, 
                        default='/home/ubuntu/Sa2VA/models/sa2va_vessel_hf')
    parser.add_argument('--data_root', type=str,
                        default='/home/ubuntu/Sa2VA/data/merged_vessel_data')
    parser.add_argument('--output_dir', type=str,
                        default='/home/ubuntu/Sa2VA/rl_reward_network/outputs')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='最大样本数（快速测试）')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=1,
                        help='使用哪个GPU (0-3)')
    
    args = parser.parse_args()
    train_reward_network(args)


if __name__ == '__main__':
    main()
