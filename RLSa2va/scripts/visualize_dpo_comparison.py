#!/usr/bin/env python3
"""
可视化对比DPO训练前后的分割效果
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '/home/ubuntu/Sa2VA')

def polygon_to_mask(polygons, height, width):
    """将多边形转换为mask"""
    from PIL import Image, ImageDraw
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for polygon in polygons:
        if len(polygon) >= 6:
            # 转换为(x,y)元组列表
            points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=255)
    
    return np.array(mask)

def calculate_metrics(pred_mask, gt_mask):
    """计算IoU和Dice"""
    pred = pred_mask > 127
    gt = gt_mask > 127
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    iou = intersection / union if union > 0 else 0
    dice = 2 * intersection / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0
    
    return iou, dice

def load_dpo_annotations():
    """加载DPO数据集"""
    ann_path = '/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_annotations.json'
    with open(ann_path, 'r') as f:
        return json.load(f)

def visualize_samples(num_samples=6):
    """可视化DPO样本对比"""
    annotations = load_dpo_annotations()
    
    # 按IoU差异排序，选择差异最大的样本
    sorted_anns = sorted(annotations, key=lambda x: x['chosen_iou'] - x['rejected_iou'], reverse=True)
    
    # 选择前num_samples个
    selected = sorted_anns[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, ann in enumerate(selected):
        # 加载图片
        img_path = os.path.join('/home/ubuntu/Sa2VA/data/dpo_vessel', ann['image'])
        if not os.path.exists(img_path):
            continue
            
        image = Image.open(img_path).convert('RGB')
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # 加载masks
        chosen_path = os.path.join('/home/ubuntu/Sa2VA/data/dpo_vessel', ann['chosen_mask'])
        rejected_path = os.path.join('/home/ubuntu/Sa2VA/data/dpo_vessel', ann['rejected_mask'])
        gt_path = os.path.join('/home/ubuntu/Sa2VA/data/dpo_vessel', ann['gt_mask'])
        
        chosen_mask = np.array(Image.open(chosen_path).convert('L')) if os.path.exists(chosen_path) else np.zeros((h, w))
        rejected_mask = np.array(Image.open(rejected_path).convert('L')) if os.path.exists(rejected_path) else np.zeros((h, w))
        gt_mask = np.array(Image.open(gt_path).convert('L')) if os.path.exists(gt_path) else np.zeros((h, w))
        
        # 绘制
        axes[i, 0].imshow(img_array)
        axes[i, 0].set_title(f'原图', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title(f'GT Mask', fontsize=10)
        axes[i, 1].axis('off')
        
        # Chosen mask (绿色overlay)
        overlay_chosen = img_array.copy()
        overlay_chosen[chosen_mask > 127] = [0, 255, 0]
        axes[i, 2].imshow(overlay_chosen)
        axes[i, 2].set_title(f'Chosen (IoU: {ann["chosen_iou"]:.3f})', fontsize=10, color='green')
        axes[i, 2].axis('off')
        
        # Rejected mask (红色overlay)
        overlay_rejected = img_array.copy()
        overlay_rejected[rejected_mask > 127] = [255, 0, 0]
        axes[i, 3].imshow(overlay_rejected)
        axes[i, 3].set_title(f'Rejected (IoU: {ann["rejected_iou"]:.3f})', fontsize=10, color='red')
        axes[i, 3].axis('off')
    
    plt.suptitle('DPO训练数据: Chosen vs Rejected 对比\n(绿色=Chosen/更好, 红色=Rejected/更差)', fontsize=14)
    plt.tight_layout()
    
    output_path = '/home/ubuntu/Sa2VA/work_dirs/dpo_vessel_training/dpo_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图已保存: {output_path}")
    plt.close()

def analyze_dpo_data():
    """分析DPO数据统计"""
    annotations = load_dpo_annotations()
    
    chosen_ious = [ann['chosen_iou'] for ann in annotations]
    rejected_ious = [ann['rejected_iou'] for ann in annotations]
    iou_diffs = [c - r for c, r in zip(chosen_ious, rejected_ious)]
    
    print("\n" + "=" * 60)
    print("📊 DPO数据集统计")
    print("=" * 60)
    print(f"样本总数: {len(annotations)}")
    print(f"\nChosen IoU:")
    print(f"  - 平均: {np.mean(chosen_ious):.4f}")
    print(f"  - 最小: {np.min(chosen_ious):.4f}")
    print(f"  - 最大: {np.max(chosen_ious):.4f}")
    print(f"\nRejected IoU:")
    print(f"  - 平均: {np.mean(rejected_ious):.4f}")
    print(f"  - 最小: {np.min(rejected_ious):.4f}")
    print(f"  - 最大: {np.max(rejected_ious):.4f}")
    print(f"\nIoU差异 (Chosen - Rejected):")
    print(f"  - 平均: {np.mean(iou_diffs):.4f}")
    print(f"  - 最小: {np.min(iou_diffs):.4f}")
    print(f"  - 最大: {np.max(iou_diffs):.4f}")
    
    # 统计DPO训练效果
    print("\n" + "=" * 60)
    print("📈 训练效果分析")
    print("=" * 60)
    
    # 加载训练日志
    log_path = '/home/ubuntu/dpo_training.log'
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # 提取loss信息
        losses = []
        for line in lines:
            if 'loss:' in line and 'Iter(train)' in line:
                try:
                    # 解析loss值
                    parts = line.split('loss:')[1].split()[0]
                    losses.append(float(parts))
                except:
                    pass
        
        if losses:
            print(f"训练Loss变化:")
            print(f"  - 初始: {losses[0]:.4f}")
            print(f"  - 最终: {losses[-1]:.4f}")
            print(f"  - 下降: {losses[0] - losses[-1]:.4f} ({(1 - losses[-1]/losses[0])*100:.1f}%)")
    
    # 绘制IoU分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(chosen_ious, bins=30, alpha=0.7, color='green', label='Chosen')
    axes[0].hist(rejected_ious, bins=30, alpha=0.7, color='red', label='Rejected')
    axes[0].set_xlabel('IoU')
    axes[0].set_ylabel('Count')
    axes[0].set_title('IoU分布对比')
    axes[0].legend()
    
    axes[1].hist(iou_diffs, bins=30, alpha=0.7, color='blue')
    axes[1].axvline(x=0, color='red', linestyle='--')
    axes[1].set_xlabel('IoU差异 (Chosen - Rejected)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('IoU差异分布')
    
    # Loss曲线
    if losses:
        axes[2].plot(losses, color='blue')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('训练Loss曲线')
    
    plt.tight_layout()
    output_path = '/home/ubuntu/Sa2VA/work_dirs/dpo_vessel_training/dpo_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 统计图已保存: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("🔍 DPO模型效果分析")
    print("=" * 60)
    
    # 分析数据统计
    analyze_dpo_data()
    
    # 可视化样本对比
    print("\n📸 生成可视化对比...")
    visualize_samples(num_samples=4)
    
    print("\n" + "=" * 60)
    print("✅ 分析完成!")
