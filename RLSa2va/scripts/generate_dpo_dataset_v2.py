#!/usr/bin/env python3
"""
生成DPO偏好对数据集 V2

从annotations.json读取mask多边形，生成偏好对
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


def polygon_to_mask(polygon_points: List[List[float]], image_size: Tuple[int, int]) -> np.ndarray:
    """
    将多边形坐标转换为二值mask
    
    Args:
        polygon_points: [[x1,y1,x2,y2,...], [x1,y1,...], ...]  多个多边形
        image_size: (width, height)
    """
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for polygon in polygon_points:
        if len(polygon) < 6:  # 至少需要3个点
            continue
        # 转换为[(x1,y1), (x2,y2), ...]格式
        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon)-1, 2)]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    
    return np.array(mask) / 255.0


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算IoU"""
    pred_binary = (pred > 0.5).astype(bool)
    gt_binary = (gt > 0.5).astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def generate_perturbed_masks(gt_mask: np.ndarray, num_samples: int = 5) -> List[Tuple[np.ndarray, dict]]:
    """
    从GT生成扰动的masks作为训练数据
    
    策略：
    1. 膨胀/腐蚀 - 模拟过分割/欠分割
    2. 随机噪声 - 模拟预测噪声
    3. 部分遮挡 - 模拟漏检
    """
    from scipy import ndimage
    
    masks = []
    gt_binary = (gt_mask > 0.5).astype(np.float32)
    
    # 原始GT（作为最佳）
    masks.append((gt_binary.copy(), {'method': 'gt', 'quality': 'best'}))
    
    # 策略1: 轻微膨胀（过分割，IoU稍低）
    try:
        dilated_1 = ndimage.binary_dilation(gt_binary, iterations=1).astype(np.float32)
        masks.append((dilated_1, {'method': 'dilation_1', 'quality': 'good'}))
    except:
        pass
    
    # 策略2: 更多膨胀（更多过分割）
    try:
        dilated_3 = ndimage.binary_dilation(gt_binary, iterations=3).astype(np.float32)
        masks.append((dilated_3, {'method': 'dilation_3', 'quality': 'medium'}))
    except:
        pass
    
    # 策略3: 轻微腐蚀（欠分割）
    try:
        eroded_1 = ndimage.binary_erosion(gt_binary, iterations=1).astype(np.float32)
        if eroded_1.sum() > 0:  # 确保不是全黑
            masks.append((eroded_1, {'method': 'erosion_1', 'quality': 'good'}))
    except:
        pass
    
    # 策略4: 更多腐蚀
    try:
        eroded_3 = ndimage.binary_erosion(gt_binary, iterations=3).astype(np.float32)
        if eroded_3.sum() > 0:
            masks.append((eroded_3, {'method': 'erosion_3', 'quality': 'medium'}))
    except:
        pass
    
    # 策略5: 添加随机噪声
    try:
        noise = np.random.random(gt_binary.shape) < 0.05
        noisy = np.logical_xor(gt_binary.astype(bool), noise).astype(np.float32)
        masks.append((noisy, {'method': 'noise', 'quality': 'poor'}))
    except:
        pass
    
    # 策略6: 随机遮挡（模拟漏检）
    try:
        occluded = gt_binary.copy()
        h, w = occluded.shape
        # 随机遮挡一个区域
        cx, cy = np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4)
        radius = min(h, w) // 8
        y, x = np.ogrid[:h, :w]
        mask_circle = (x - cx)**2 + (y - cy)**2 <= radius**2
        occluded[mask_circle] = 0
        if occluded.sum() > 0:
            masks.append((occluded, {'method': 'occlusion', 'quality': 'poor'}))
    except:
        pass
    
    # 策略7: 边界模糊
    try:
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(gt_binary.astype(float), sigma=2)
        blurred = (blurred > 0.3).astype(np.float32)
        masks.append((blurred, {'method': 'blur', 'quality': 'medium'}))
    except:
        pass
    
    return masks[:num_samples]


def generate_dpo_dataset(
    data_root: str,
    ann_file: str,
    output_dir: str,
    num_samples: int = 5,
    min_iou_gap: float = 0.05
):
    """生成DPO数据集"""
    
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    # 加载annotations
    ann_path = os.path.join(data_root, ann_file)
    with open(ann_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"📊 加载了 {len(annotations)} 个样本")
    
    all_pairs = []
    
    for idx, item in enumerate(tqdm(annotations, desc="生成偏好对")):
        image_name = item['image']
        mask_polygons = item['mask']
        
        # 加载图像获取尺寸
        image_path = os.path.join(data_root, 'images', image_name)
        if not os.path.exists(image_path):
            continue
        
        image = Image.open(image_path)
        image_size = image.size  # (width, height)
        
        # 从多边形生成GT mask
        gt_mask = polygon_to_mask(mask_polygons, image_size)
        
        if gt_mask.sum() == 0:
            continue  # 跳过空mask
        
        # 生成扰动masks
        perturbed_masks = generate_perturbed_masks(gt_mask, num_samples)
        
        if len(perturbed_masks) < 2:
            continue
        
        # 计算每个mask的IoU
        mask_scores = []
        for i, (mask, meta) in enumerate(perturbed_masks):
            iou = compute_iou(mask, gt_mask)
            mask_scores.append({
                'mask': mask,
                'iou': iou,
                'meta': meta,
                'index': i
            })
        
        # 按IoU排序
        mask_scores.sort(key=lambda x: x['iou'], reverse=True)
        
        # 构建偏好对
        image_id = Path(image_name).stem
        
        for i in range(len(mask_scores)):
            for j in range(i + 1, len(mask_scores)):
                chosen = mask_scores[i]
                rejected = mask_scores[j]
                
                iou_gap = chosen['iou'] - rejected['iou']
                if iou_gap < min_iou_gap:
                    continue
                
                # 保存masks
                chosen_filename = f"{image_id}_chosen_{i}_{j}.png"
                rejected_filename = f"{image_id}_rejected_{i}_{j}.png"
                
                chosen_path = os.path.join(masks_dir, chosen_filename)
                rejected_path = os.path.join(masks_dir, rejected_filename)
                
                Image.fromarray((chosen['mask'] * 255).astype(np.uint8)).save(chosen_path)
                Image.fromarray((rejected['mask'] * 255).astype(np.uint8)).save(rejected_path)
                
                pair = {
                    'image': f"images/{image_name}",
                    'chosen_mask': f"masks/{chosen_filename}",
                    'rejected_mask': f"masks/{rejected_filename}",
                    'chosen_iou': chosen['iou'],
                    'rejected_iou': rejected['iou'],
                    'iou_gap': iou_gap,
                    'chosen_method': chosen['meta']['method'],
                    'rejected_method': rejected['meta']['method'],
                    'prompt': '<image>Please segment the blood vessels.'
                }
                all_pairs.append(pair)
    
    # 保存annotations
    output_ann_path = os.path.join(output_dir, 'dpo_annotations.json')
    with open(output_ann_path, 'w') as f:
        json.dump(all_pairs, f, indent=2)
    
    # 创建软链接到images目录
    images_link = os.path.join(output_dir, 'images')
    if not os.path.exists(images_link):
        os.symlink(os.path.join(data_root, 'images'), images_link)
    
    print(f"\n{'='*60}")
    print(f"📊 DPO数据集生成完成!")
    print(f"{'='*60}")
    print(f"  - 总偏好对数: {len(all_pairs)}")
    if all_pairs:
        print(f"  - 平均IoU差距: {np.mean([p['iou_gap'] for p in all_pairs]):.4f}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - Annotations: {output_ann_path}")
    
    return all_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成DPO偏好对数据集 V2')
    parser.add_argument('--data_root', type=str, 
                        default='/home/ubuntu/Sa2VA/data/merged_vessel_data',
                        help='数据根目录（包含images/和annotations.json）')
    parser.add_argument('--ann_file', type=str, default='annotations.json',
                        help='Annotations文件名')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/ubuntu/Sa2VA/data/dpo_vessel',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=5, 
                        help='每张图像生成的扰动mask数量')
    parser.add_argument('--min_iou_gap', type=float, default=0.05, 
                        help='最小IoU差距阈值')
    
    args = parser.parse_args()
    
    generate_dpo_dataset(
        data_root=args.data_root,
        ann_file=args.ann_file,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        min_iou_gap=args.min_iou_gap
    )
