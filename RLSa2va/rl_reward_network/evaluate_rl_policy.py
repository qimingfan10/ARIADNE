"""
评估训练好的RL策略
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import cv2

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from stable_baselines3 import PPO

from env.sa2va_finetune_env import Sa2VAFinetuneEnv
from models.reward_network import LightweightRewardNetwork


def calculate_metrics(pred_mask, gt_mask):
    """计算评估指标"""
    # 二值化
    pred = (pred_mask > 0).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)
    
    # 计算指标
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    
    # Dice
    dice = (2.0 * intersection) / (union + 1e-8)
    
    # IoU
    iou = intersection / (pred.sum() + gt.sum() - intersection + 1e-8)
    
    # Precision
    precision = intersection / (pred.sum() + 1e-8)
    
    # Recall
    recall = intersection / (gt.sum() + 1e-8)
    
    # Accuracy
    correct = ((pred == gt).sum())
    total = pred.size
    accuracy = correct / total
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy)
    }


def load_dataset(data_root, max_samples=None):
    """加载数据集"""
    from PIL import Image, ImageDraw
    
    # 加载annotations
    annotations_path = os.path.join(data_root, 'annotations.json')
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # 限制样本数
    if max_samples is not None:
        annotations = annotations[:max_samples]
    
    # 加载数据
    dataset = []
    images_dir = os.path.join(data_root, 'images')
    
    for ann in tqdm(annotations, desc="加载数据集"):
        try:
            # 加载图像
            image_path = os.path.join(images_dir, ann['image'])
            image = Image.open(image_path).convert('RGB')
            
            # 从坐标点生成mask
            width, height = image.size
            mask_img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask_img)
            
            # 绘制多边形mask
            if 'mask' in ann and len(ann['mask']) > 0:
                polygons = ann['mask']
                for polygon in polygons:
                    if len(polygon) >= 6:
                        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                        draw.polygon(points, fill=255)
            
            # 转换为numpy数组
            mask = np.array(mask_img)
            
            dataset.append({
                'image': image,
                'mask': mask,
                'image_path': ann['image']
            })
            
        except Exception as e:
            print(f"加载失败 {ann.get('image', 'unknown')}: {e}")
            continue
    
    print(f"✅ 加载{len(dataset)}个样本")
    return dataset


def evaluate_rl_policy(args):
    """评估RL策略"""
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载Reward Network
    print(f"\n加载Reward Network...")
    reward_network = LightweightRewardNetwork(input_channels=4).to(device)
    checkpoint = torch.load(args.reward_net_path, map_location=device)
    reward_network.load_state_dict(checkpoint['model_state_dict'])
    reward_network.eval()
    print("✅ Reward Network加载成功")
    
    # 加载Sa2VA模型
    print(f"\n加载Sa2VA模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    sa2va_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    sa2va_model.eval()
    print("✅ Sa2VA模型加载成功")
    
    # 加载数据集
    print(f"\n加载数据集...")
    dataset = load_dataset(args.data_root, max_samples=args.max_samples)
    
    # 创建环境
    print(f"\n创建评估环境...")
    env = Sa2VAFinetuneEnv(
        sa2va_model=sa2va_model,
        tokenizer=tokenizer,
        reward_network=reward_network,
        dataset=dataset,
        device=device,
        max_steps=1
    )
    
    # 加载RL策略
    print(f"\n加载RL策略: {args.policy_path}")
    model = PPO.load(args.policy_path, env=env)
    print("✅ RL策略加载成功")
    
    # 评估
    print(f"\n开始评估...")
    results = []
    
    for i in tqdm(range(len(dataset)), desc="评估中"):
        obs, info = env.reset()
        
        # 使用RL策略选择action
        action, _states = model.predict(obs, deterministic=True)
        
        # 执行action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 获取最佳mask
        pred_mask = env.best_mask
        gt_mask = env.current_gt_mask
        
        # 计算指标
        metrics = calculate_metrics(pred_mask, gt_mask)
        metrics['reward_net_score'] = info['reward_net_score']
        metrics['action'] = int(action)
        metrics['prompt'] = info['prompt']
        metrics['image_path'] = dataset[i]['image_path']
        
        results.append(metrics)
    
    # 统计平均指标
    avg_metrics = {
        'dice': np.mean([r['dice'] for r in results]),
        'iou': np.mean([r['iou'] for r in results]),
        'precision': np.mean([r['precision'] for r in results]),
        'recall': np.mean([r['recall'] for r in results]),
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'reward_net_score': np.mean([r['reward_net_score'] for r in results])
    }
    
    # 统计action分布
    action_counts = {}
    for r in results:
        action = r['action']
        action_counts[action] = action_counts.get(action, 0) + 1
    
    # 保存结果
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'average_metrics': avg_metrics,
            'action_distribution': action_counts,
            'per_sample_results': results
        }, f, indent=2)
    
    print(f"\n✅ 评估完成！")
    print(f"\n{'='*60}")
    print("平均指标:")
    print(f"{'='*60}")
    for key, value in avg_metrics.items():
        print(f"{key:20s}: {value:.4f}")
    
    print(f"\n{'='*60}")
    print("Action分布:")
    print(f"{'='*60}")
    for action, count in sorted(action_counts.items()):
        percentage = count / len(results) * 100
        print(f"Action {action:2d}: {count:3d}次 ({percentage:5.1f}%)")
    
    print(f"\n结果已保存至: {results_file}")
    
    return avg_metrics, results


def main():
    parser = argparse.ArgumentParser(description='评估RL策略')
    
    parser.add_argument('--policy_path', type=str, required=True,
                        help='RL策略路径')
    parser.add_argument('--model_path', type=str, 
                        default='/home/ubuntu/Sa2VA/models/sa2va_vessel_hf',
                        help='Sa2VA模型路径')
    parser.add_argument('--reward_net_path', type=str,
                        default='/home/ubuntu/Sa2VA/rl_reward_network/outputs/reward_net_20251129_132402/best_reward_net.pth',
                        help='Reward Network路径')
    parser.add_argument('--data_root', type=str,
                        default='/home/ubuntu/Sa2VA/data/merged_vessel_data',
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str,
                        default='./evaluation_output',
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='最大评估样本数')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("实验三步骤2：RL策略评估")
    print("="*60)
    print(f"策略路径: {args.policy_path}")
    print(f"Sa2VA模型: {args.model_path}")
    print(f"Reward Network: {args.reward_net_path}")
    print(f"数据集: {args.data_root}")
    print(f"评估样本数: {args.max_samples}")
    print("="*60 + "\n")
    
    avg_metrics, results = evaluate_rl_policy(args)
    
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)


if __name__ == '__main__':
    main()
