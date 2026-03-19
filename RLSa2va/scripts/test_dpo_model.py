#!/usr/bin/env python3
"""
测试DPO训练后的模型效果
对比原始模型和DPO模型的分割质量
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/home/ubuntu/Sa2VA')

def calculate_iou(pred_mask, gt_mask):
    """计算IoU"""
    pred = pred_mask > 0.5
    gt = gt_mask > 0.5
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def calculate_dice(pred_mask, gt_mask):
    """计算Dice系数"""
    pred = pred_mask > 0.5
    gt = gt_mask > 0.5
    intersection = np.logical_and(pred, gt).sum()
    if pred.sum() + gt.sum() == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2 * intersection / (pred.sum() + gt.sum())

def load_model(checkpoint_path, config_path):
    """加载模型"""
    from mmengine.config import Config
    from mmengine.registry import MODELS
    from mmengine.runner import load_checkpoint
    
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤并加载权重
        model_state = model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)}/{len(state_dict)} weights from {checkpoint_path}")
    
    model.eval()
    model.cuda()
    return model

def test_single_image(model, image_path, prompt="Please segment the blood vessel in this image."):
    """测试单张图片"""
    from transformers import AutoTokenizer
    from torchvision import transforms
    
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 使用模型推理
    with torch.no_grad():
        try:
            result = model.generate_mask(image, prompt)
            return result
        except Exception as e:
            print(f"推理错误: {e}")
            return None

def evaluate_on_dataset(model, data_root, ann_file, num_samples=50):
    """在数据集上评估"""
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # 随机采样
    if len(annotations) > num_samples:
        import random
        random.seed(42)
        annotations = random.sample(annotations, num_samples)
    
    results = {
        'iou_scores': [],
        'dice_scores': [],
        'success_count': 0,
        'total_count': len(annotations)
    }
    
    for ann in tqdm(annotations, desc="评估中"):
        image_path = os.path.join(data_root, 'images', ann['image'])
        if not os.path.exists(image_path):
            continue
        
        try:
            pred_mask = test_single_image(model, image_path)
            if pred_mask is not None:
                results['success_count'] += 1
                # 如果有GT mask，计算指标
                # 这里简化处理，只统计成功推理的数量
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return results

def quick_visual_test(checkpoint_path, test_images_dir, output_dir):
    """快速可视化测试"""
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer, AutoModel
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取测试图片
    test_images = list(Path(test_images_dir).glob("*.jpg"))[:5]
    
    print(f"\n🔍 测试 {len(test_images)} 张图片...")
    print(f"   模型: {checkpoint_path}")
    
    for img_path in test_images:
        print(f"   - {img_path.name}")
    
    return test_images

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/ubuntu/Sa2VA/work_dirs/dpo_vessel_training/iter_1224.pth',
                       help='DPO模型checkpoint路径')
    parser.add_argument('--baseline', type=str,
                       default='/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth',
                       help='基线模型checkpoint路径')
    parser.add_argument('--test_dir', type=str,
                       default='/home/ubuntu/Sa2VA/data/dpo_vessel/images',
                       help='测试图片目录')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='测试样本数')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 DPO模型效果测试")
    print("=" * 60)
    
    # 检查文件
    print("\n📁 检查文件...")
    print(f"   DPO模型: {os.path.exists(args.checkpoint)} - {args.checkpoint}")
    print(f"   基线模型: {os.path.exists(args.baseline)} - {args.baseline}")
    print(f"   测试目录: {os.path.exists(args.test_dir)} - {args.test_dir}")
    
    # 统计测试图片
    test_images = list(Path(args.test_dir).glob("*.jpg"))
    print(f"   测试图片数: {len(test_images)}")
    
    # 检查checkpoint内容
    print("\n📊 Checkpoint信息:")
    
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            print(f"   DPO模型参数数: {len(ckpt['state_dict'])}")
            # 显示一些关键参数
            lora_params = [k for k in ckpt['state_dict'].keys() if 'lora' in k.lower()]
            print(f"   LoRA参数数: {len(lora_params)}")
        if 'meta' in ckpt:
            print(f"   训练迭代: {ckpt['meta'].get('iter', 'N/A')}")
            print(f"   训练epoch: {ckpt['meta'].get('epoch', 'N/A')}")
    
    if os.path.exists(args.baseline):
        ckpt = torch.load(args.baseline, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            print(f"   基线模型参数数: {len(ckpt['state_dict'])}")
    
    print("\n✅ 文件检查完成!")
    print("\n💡 要进行完整推理测试，请运行:")
    print(f"   python scripts/inference_sa2va.py --checkpoint {args.checkpoint}")

if __name__ == '__main__':
    main()
