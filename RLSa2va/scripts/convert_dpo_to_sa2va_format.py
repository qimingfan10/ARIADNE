#!/usr/bin/env python3
"""
将DPO偏好对数据集转换为Sa2VA训练格式

从DPO数据集中提取chosen masks，转换为Sa2VA可用的格式
"""

import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def mask_to_polygon(mask_path: str) -> list:
    """将二值mask转换为多边形坐标"""
    from skimage import measure
    
    mask = np.array(Image.open(mask_path).convert('L'))
    mask = (mask > 127).astype(np.uint8)
    
    polygons = []
    contours = measure.find_contours(mask, 0.5)
    
    for contour in contours:
        if len(contour) < 10:  # 忽略太小的轮廓
            continue
        # 简化轮廓点数
        step = max(1, len(contour) // 100)
        simplified = contour[::step]
        
        # 转换为[x1, y1, x2, y2, ...]格式
        polygon = []
        for point in simplified:
            polygon.extend([float(point[1]), float(point[0])])  # x, y
        
        if len(polygon) >= 6:  # 至少3个点
            polygons.append(polygon)
    
    return polygons


def convert_dpo_to_sa2va(
    dpo_data_root: str,
    dpo_ann_file: str,
    output_ann_file: str,
    use_chosen_only: bool = True
):
    """
    将DPO数据集转换为Sa2VA格式
    
    Args:
        dpo_data_root: DPO数据集根目录
        dpo_ann_file: DPO annotations文件
        output_ann_file: 输出的Sa2VA格式annotations
        use_chosen_only: 只使用chosen masks
    """
    
    # 加载DPO annotations
    with open(dpo_ann_file, 'r') as f:
        dpo_annotations = json.load(f)
    
    print(f"📊 加载了 {len(dpo_annotations)} 个DPO偏好对")
    
    # 去重：每个图像只保留IoU最高的chosen mask
    image_to_best = {}
    
    for item in dpo_annotations:
        image_name = item['image']
        chosen_iou = item['chosen_iou']
        
        if image_name not in image_to_best or chosen_iou > image_to_best[image_name]['chosen_iou']:
            image_to_best[image_name] = item
    
    print(f"📊 去重后: {len(image_to_best)} 个唯一图像")
    
    # 转换为Sa2VA格式
    sa2va_annotations = []
    
    for image_name, item in tqdm(image_to_best.items(), desc="转换格式"):
        chosen_mask_path = os.path.join(dpo_data_root, item['chosen_mask'])
        
        if not os.path.exists(chosen_mask_path):
            print(f"⚠️ 找不到mask: {chosen_mask_path}")
            continue
        
        # 转换mask为多边形
        try:
            polygons = mask_to_polygon(chosen_mask_path)
        except Exception as e:
            print(f"⚠️ 转换失败 {chosen_mask_path}: {e}")
            continue
        
        if not polygons:
            continue
        
        # 提取图像文件名（去掉images/前缀）
        if image_name.startswith('images/'):
            image_filename = image_name[7:]
        else:
            image_filename = image_name
        
        # 创建Sa2VA格式的annotation
        # text必须是列表，每个polygon对应一个text
        text_list = ["blood vessel"] * len(polygons)
        
        sa2va_item = {
            "image": image_filename,
            "mask": polygons,
            "text": text_list,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nPlease segment the blood vessel in this image."
                },
                {
                    "from": "gpt",
                    "value": "Sure, [SEG]."
                }
            ]
        }
        
        sa2va_annotations.append(sa2va_item)
    
    # 保存
    with open(output_ann_file, 'w') as f:
        json.dump(sa2va_annotations, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 转换完成!")
    print(f"  - 输出样本数: {len(sa2va_annotations)}")
    print(f"  - 输出文件: {output_ann_file}")
    
    return sa2va_annotations


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpo_root', type=str, 
                        default='/home/ubuntu/Sa2VA/data/dpo_vessel')
    parser.add_argument('--dpo_ann', type=str,
                        default='/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_annotations.json')
    parser.add_argument('--output', type=str,
                        default='/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_chosen_annotations.json')
    
    args = parser.parse_args()
    
    convert_dpo_to_sa2va(
        dpo_data_root=args.dpo_root,
        dpo_ann_file=args.dpo_ann,
        output_ann_file=args.output
    )
