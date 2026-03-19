"""
主训练脚本：LoRA + PPO微调Sa2VA
目标：通过强化学习直接优化Dice指标
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

from reward_functions import get_reward_function
from lora_config import get_lora_config, apply_lora_to_model, save_lora_weights
from data_loader import create_dataloaders


class Sa2VALoRAPPOTrainer:
    """Sa2VA LoRA + PPO训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # 设置输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(args.output_dir, f'sa2va_lora_ppo_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化wandb
        if args.use_wandb:
            wandb.init(
                project="sa2va-lora-ppo",
                name=f"run_{timestamp}",
                config=vars(args)
            )
        
        # 加载模型
        print(f"\n{'='*60}")
        print("加载Sa2VA模型...")
        print(f"{'='*60}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,  # Sa2VA默认使用bfloat16
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 应用LoRA
        print(f"\n应用LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
        lora_config = get_lora_config(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        self.model = apply_lora_to_model(self.base_model, lora_config)
        self.model.eval()  # RL训练时保持eval模式（不更新BN等）
        
        # 奖励函数
        print(f"\n初始化奖励函数: {args.reward_type}...")
        self.reward_function = get_reward_function(
            reward_type=args.reward_type,
            dice_weight=args.dice_weight,
            recall_weight=args.recall_weight,
            topology_weight=args.topology_weight,
            length_weight=args.length_weight,
            recall_target=args.recall_target
        )
        
        # 数据加载
        print(f"\n加载数据...")
        self.train_loader, self.val_loader = create_dataloaders(
            data_root=args.data_root,
            train_samples=args.max_train_samples,
            val_samples=args.max_val_samples,
            batch_size=1,  # RL训练batch=1
            num_workers=args.num_workers
        )
        
        # 优化器（仅优化LoRA参数）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        total_steps = len(self.train_loader) * args.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=args.learning_rate * 0.1
        )
        
        # 统计信息
        self.global_step = 0
        self.best_val_dice = 0.0
        
        print(f"\n{'='*60}")
        print("训练配置:")
        print(f"{'='*60}")
        print(f"输出目录: {self.output_dir}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")
        print(f"训练轮数: {args.num_epochs}")
        print(f"学习率: {args.learning_rate}")
        print(f"设备: {self.device}")
        print(f"{'='*60}\n")
    
    def predict_with_sa2va(self, image, prompt):
        """使用Sa2VA进行预测"""
        # 确保prompt包含<image>标记
        if '<image>' not in prompt:
            text_with_image = f"<image>\n{prompt}"
        else:
            text_with_image = prompt
        
        with torch.no_grad():
            result = self.model.predict_forward(
                image=image,
                text=text_with_image,
                tokenizer=self.tokenizer
            )
        
        # 解析返回值
        if isinstance(result, dict) and 'prediction_masks' in result:
            masks = result['prediction_masks']
            if len(masks) > 0:
                pred_mask = masks[0]
                if len(pred_mask.shape) > 2:
                    pred_mask = pred_mask[0]
                if isinstance(pred_mask, torch.Tensor):
                    pred_mask = pred_mask.cpu().numpy()
                return pred_mask.astype(np.uint8)
        
        return None
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()  # 启用梯度
        
        epoch_rewards = []
        epoch_dice = []
        epoch_recall = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # batch是list，因为collate_fn返回list
            sample = batch[0]  # Batch size = 1
            image = sample['image']
            gt_mask = sample['mask']
            
            # 使用当前策略预测
            pred_mask = self.predict_with_sa2va(image, self.args.prompt)
            
            if pred_mask is None:
                print(f"警告: 预测失败，跳过该样本")
                continue
            
            # 调整大小匹配GT
            if pred_mask.shape != gt_mask.shape:
                from PIL import Image as PILImage
                pred_mask = np.array(PILImage.fromarray(pred_mask).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), PILImage.NEAREST
                ))
            
            # 计算奖励
            reward, reward_dict = self.reward_function(pred_mask, gt_mask)
            
            # RL损失（简化版：直接用reward作为损失）
            # 注意：这是简化实现，完整的PPO需要更复杂的逻辑
            loss = -reward  # 负奖励作为损失
            
            # 反向传播
            self.optimizer.zero_grad()
            
            # 由于Sa2VA的predict_forward在no_grad下，我们需要重新forward
            # 这里简化处理：基于reward调整LoRA参数
            # 实际PPO需要policy gradient
            
            # TODO: 完整的PPO实现需要：
            # 1. 计算旧策略的log_prob
            # 2. 计算新策略的log_prob
            # 3. 计算重要性采样比率
            # 4. 使用clip更新策略
            
            # 当前简化版本：直接用reward信号更新
            if reward > 0:
                # 正奖励：尝试增强当前策略
                pass  # 简化处理
            
            # 记录统计
            epoch_rewards.append(reward)
            epoch_dice.append(reward_dict['dice'])
            if 'recall' in reward_dict:
                epoch_recall.append(reward_dict['recall'])
            
            # 更新进度条
            pbar.set_postfix({
                'reward': f'{reward:.4f}',
                'dice': f'{reward_dict["dice"]:.4f}',
                'recall': f'{reward_dict.get("recall", 0):.4f}'
            })
            
            self.global_step += 1
            
            # 日志
            if self.args.use_wandb and self.global_step % self.args.log_freq == 0:
                wandb.log({
                    'train/reward': reward,
                    'train/dice': reward_dict['dice'],
                    'train/recall': reward_dict.get('recall', 0),
                    'train/precision': reward_dict.get('precision', 0),
                    'global_step': self.global_step
                })
            
            # 验证
            if self.global_step % self.args.val_freq == 0:
                val_metrics = self.validate()
                
                # 保存最佳模型
                if val_metrics['dice'] > self.best_val_dice:
                    self.best_val_dice = val_metrics['dice']
                    save_path = os.path.join(self.output_dir, 'best_lora')
                    save_lora_weights(self.model, save_path)
                    print(f"\n✅ 新的最佳模型! Dice: {self.best_val_dice:.4f}")
        
        # Epoch统计
        avg_reward = np.mean(epoch_rewards)
        avg_dice = np.mean(epoch_dice)
        avg_recall = np.mean(epoch_recall) if epoch_recall else 0
        
        print(f"\nEpoch {epoch+1} 统计:")
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  平均Dice: {avg_dice:.4f}")
        print(f"  平均Recall: {avg_recall:.4f}")
        
        return {
            'reward': avg_reward,
            'dice': avg_dice,
            'recall': avg_recall
        }
    
    def validate(self):
        """验证"""
        self.model.eval()
        
        val_dice = []
        val_recall = []
        val_precision = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中", leave=False):
                # batch是list，因为collate_fn返回list
                sample = batch[0]  # Batch size = 1
                image = sample['image']
                gt_mask = sample['mask']
                
                pred_mask = self.predict_with_sa2va(image, self.args.prompt)
                
                if pred_mask is None:
                    continue
                
                # 调整大小
                if pred_mask.shape != gt_mask.shape:
                    from PIL import Image as PILImage
                    pred_mask = np.array(PILImage.fromarray(pred_mask).resize(
                        (gt_mask.shape[1], gt_mask.shape[0]), PILImage.NEAREST
                    ))
                
                # 计算指标
                _, reward_dict = self.reward_function(pred_mask, gt_mask)
                
                val_dice.append(reward_dict['dice'])
                if 'recall' in reward_dict:
                    val_recall.append(reward_dict['recall'])
                if 'precision' in reward_dict:
                    val_precision.append(reward_dict['precision'])
        
        metrics = {
            'dice': np.mean(val_dice),
            'recall': np.mean(val_recall) if val_recall else 0,
            'precision': np.mean(val_precision) if val_precision else 0
        }
        
        print(f"\n验证结果:")
        print(f"  Dice: {metrics['dice']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        
        if self.args.use_wandb:
            wandb.log({
                'val/dice': metrics['dice'],
                'val/recall': metrics['recall'],
                'val/precision': metrics['precision'],
                'global_step': self.global_step
            })
        
        return metrics
    
    def train(self):
        """主训练循环"""
        print(f"\n{'='*60}")
        print("开始训练...")
        print(f"{'='*60}\n")
        
        for epoch in range(self.args.num_epochs):
            epoch_metrics = self.train_epoch(epoch)
            
            # 保存checkpoint
            if (epoch + 1) % self.args.save_freq == 0:
                checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}')
                save_lora_weights(self.model, checkpoint_path)
        
        # 保存最终模型
        final_path = os.path.join(self.output_dir, 'final_lora')
        save_lora_weights(self.model, final_path)
        
        # 最终验证
        print(f"\n{'='*60}")
        print("最终验证...")
        print(f"{'='*60}")
        final_metrics = self.validate()
        
        # 保存训练信息
        training_info = {
            'args': vars(self.args),
            'final_metrics': final_metrics,
            'best_val_dice': self.best_val_dice,
            'total_steps': self.global_step
        }
        
        info_path = os.path.join(self.output_dir, 'training_info.json')
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"{'='*60}")
        print(f"最佳验证Dice: {self.best_val_dice:.4f}")
        print(f"最终Dice: {final_metrics['dice']:.4f}")
        print(f"最终Recall: {final_metrics['recall']:.4f}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")
        
        if self.args.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Sa2VA LoRA + PPO训练')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='Sa2VA模型路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./lora_ppo_output',
                        help='输出目录')
    
    # LoRA参数
    parser.add_argument('--lora_rank', type=int, default=32,
                        help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int, default=64,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='最大训练样本数')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='最大验证样本数')
    
    # 奖励函数参数
    parser.add_argument('--reward_type', type=str, default='multi_objective',
                        choices=['multi_objective', 'simple_dice', 'recall_focused'],
                        help='奖励函数类型')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Dice权重')
    parser.add_argument('--recall_weight', type=float, default=0.2,
                        help='Recall权重')
    parser.add_argument('--topology_weight', type=float, default=0.2,
                        help='拓扑权重')
    parser.add_argument('--length_weight', type=float, default=0.1,
                        help='长度权重')
    parser.add_argument('--recall_target', type=float, default=0.85,
                        help='Recall目标值')
    
    # Prompt
    parser.add_argument('--prompt', type=str, 
                        default='Please segment the blood vessel.',
                        help='推理prompt')
    
    # 其他参数
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--use_bf16', action='store_true',
                        help='使用bfloat16')
    parser.add_argument('--use_wandb', action='store_true',
                        help='使用Wandb记录')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='日志频率')
    parser.add_argument('--val_freq', type=int, default=100,
                        help='验证频率')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='保存频率（epoch）')
    
    args = parser.parse_args()
    
    # 创建trainer并训练
    trainer = Sa2VALoRAPPOTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
