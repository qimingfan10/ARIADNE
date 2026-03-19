#!/usr/bin/env python3
"""
生成DPO偏好对数据集

工作流程：
1. 加载Sa2VA模型
2. 对每张图像，使用不同的采样策略生成多个mask
3. 计算每个mask与GT的IoU
4. 构建偏好对：IoU高的为chosen，IoU低的为rejected
5. 保存为DPO数据集格式
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/home/ubuntu/Sa2VA')


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算IoU (Intersection over Union)"""
    pred_binary = (pred > 0.5).astype(bool)
    gt_binary = (gt > 0.5).astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算Dice系数"""
    pred_binary = (pred > 0.5).astype(bool)
    gt_binary = (gt > 0.5).astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    
    if pred_binary.sum() + gt_binary.sum() == 0:
        return 0.0
    return float(2 * intersection) / float(pred_binary.sum() + gt_binary.sum())


class PreferencePairGenerator:
    """偏好对生成器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        num_samples_per_image: int = 5,
        temperature_range: Tuple[float, float] = (0.3, 1.0),
        min_iou_gap: float = 0.05
    ):
        self.device = device
        self.num_samples = num_samples_per_image
        self.temp_range = temperature_range
        self.min_iou_gap = min_iou_gap
        
        # 加载Sa2VA模型
        print(f"📦 加载Sa2VA模型: {model_path}")
        from transformers import AutoModel, AutoTokenizer
        
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print("✅ 模型加载完成")
    
    @torch.no_grad()
    def generate_masks(self, image: Image.Image, prompt: str) -> List[Tuple[np.ndarray, dict]]:
        """
        使用不同采样策略生成多个mask
        
        Returns:
            List of (mask, metadata) tuples
        """
        masks = []
        
        # 策略1: 不同温度采样
        temperatures = np.linspace(self.temp_range[0], self.temp_range[1], self.num_samples)
        
        for temp in temperatures:
            try:
                # 调用Sa2VA生成mask
                result = self.model.chat(
                    self.tokenizer,
                    pixel_values=image,
                    question=prompt,
                    generation_config={
                        'temperature': temp,
                        'do_sample': temp > 0.1,
                        'max_new_tokens': 512
                    }
                )
                
                # 提取mask
                if hasattr(result, 'masks') and len(result.masks) > 0:
                    mask = result.masks[0].cpu().numpy()
                    masks.append((mask, {'temperature': temp, 'method': 'temperature_sampling'}))
                    
            except Exception as e:
                print(f"⚠️ 生成失败 (temp={temp}): {e}")
                continue
        
        # 策略2: 添加噪声扰动
        try:
            # 基础预测
            base_result = self.model.chat(
                self.tokenizer,
                pixel_values=image,
                question=prompt,
                generation_config={'temperature': 0.0, 'do_sample': False}
            )
            
            if hasattr(base_result, 'masks') and len(base_result.masks) > 0:
                base_mask = base_result.masks[0].cpu().numpy()
                
                # 添加形态学扰动
                from scipy import ndimage
                
                # 膨胀
                dilated = ndimage.binary_dilation(base_mask > 0.5, iterations=2).astype(np.float32)
                masks.append((dilated, {'method': 'dilation'}))
                
                # 腐蚀
                eroded = ndimage.binary_erosion(base_mask > 0.5, iterations=2).astype(np.float32)
                masks.append((eroded, {'method': 'erosion'}))
                
        except Exception as e:
            print(f"⚠️ 扰动生成失败: {e}")
        
        return masks
    
    def build_preference_pairs(
        self,
        image_path: str,
        gt_path: str,
        output_dir: str,
        image_id: str
    ) -> List[dict]:
        """
        为单张图像构建偏好对
        
        Returns:
            List of preference pair annotations
        """
        # 加载图像和GT
        image = Image.open(image_path).convert('RGB')
        gt = np.array(Image.open(gt_path).convert('L'))
        gt = (gt > 127).astype(np.float32)
        
        # 生成多个mask
        prompt = "<image>Please segment the blood vessels."
        masks_with_meta = self.generate_masks(image, prompt)
        
        if len(masks_with_meta) < 2:
            return []
        
        # 计算每个mask的IoU
        mask_scores = []
        for i, (mask, meta) in enumerate(masks_with_meta):
            # 确保mask尺寸与GT一致
            if mask.shape != gt.shape:
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((gt.shape[1], gt.shape[0]), PILImage.NEAREST)
                mask = np.array(mask_pil) / 255.0
            
            iou = compute_iou(mask, gt)
            dice = compute_dice(mask, gt)
            mask_scores.append({
                'mask': mask,
                'iou': iou,
                'dice': dice,
                'meta': meta,
                'index': i
            })
        
        # 按IoU排序
        mask_scores.sort(key=lambda x: x['iou'], reverse=True)
        
        # 构建偏好对
        pairs = []
        masks_dir = os.path.join(output_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        for i in range(len(mask_scores)):
            for j in range(i + 1, len(mask_scores)):
                chosen = mask_scores[i]
                rejected = mask_scores[j]
                
                # 检查IoU差距是否足够大
                iou_gap = chosen['iou'] - rejected['iou']
                if iou_gap < self.min_iou_gap:
                    continue
                
                # 保存masks
                chosen_filename = f"{image_id}_chosen_{i}_{j}.png"
                rejected_filename = f"{image_id}_rejected_{i}_{j}.png"
                
                chosen_path = os.path.join(masks_dir, chosen_filename)
                rejected_path = os.path.join(masks_dir, rejected_filename)
                
                Image.fromarray((chosen['mask'] * 255).astype(np.uint8)).save(chosen_path)
                Image.fromarray((rejected['mask'] * 255).astype(np.uint8)).save(rejected_path)
                
                # 创建annotation
                pair = {
                    'image': os.path.relpath(image_path, output_dir),
                    'chosen_mask': os.path.relpath(chosen_path, output_dir),
                    'rejected_mask': os.path.relpath(rejected_path, output_dir),
                    'chosen_iou': chosen['iou'],
                    'rejected_iou': rejected['iou'],
                    'chosen_dice': chosen['dice'],
                    'rejected_dice': rejected['dice'],
                    'iou_gap': iou_gap,
                    'chosen_meta': chosen['meta'],
                    'rejected_meta': rejected['meta'],
                    'prompt': prompt
                }
                pairs.append(pair)
        
        return pairs


def generate_from_existing_data(
    images_dir: str,
    gt_dir: str,
    output_dir: str,
    model_path: str,
    num_samples: int = 5,
    min_iou_gap: float = 0.05
):
    """从现有数据生成DPO数据集"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化生成器
    generator = PreferencePairGenerator(
        model_path=model_path,
        num_samples_per_image=num_samples,
        min_iou_gap=min_iou_gap
    )
    
    # 收集所有图像
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(images_dir).glob(ext))
    
    print(f"📊 找到 {len(image_files)} 张图像")
    
    # 生成偏好对
    all_pairs = []
    
    for img_path in tqdm(image_files, desc="生成偏好对"):
        # 找到对应的GT
        gt_path = Path(gt_dir) / img_path.name
        if not gt_path.exists():
            # 尝试其他命名格式
            gt_path = Path(gt_dir) / f"{img_path.stem}_mask.png"
        if not gt_path.exists():
            gt_path = Path(gt_dir) / f"{img_path.stem}.png"
        
        if not gt_path.exists():
            print(f"⚠️ 找不到GT: {img_path.name}")
            continue
        
        # 生成偏好对
        pairs = generator.build_preference_pairs(
            image_path=str(img_path),
            gt_path=str(gt_path),
            output_dir=output_dir,
            image_id=img_path.stem
        )
        
        all_pairs.extend(pairs)
        print(f"  ✅ {img_path.name}: {len(pairs)} 对")
    
    # 保存annotations
    ann_path = os.path.join(output_dir, 'dpo_annotations.json')
    with open(ann_path, 'w') as f:
        json.dump(all_pairs, f, indent=2)
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"📊 DPO数据集生成完成!")
    print(f"{'='*60}")
    print(f"  - 总偏好对数: {len(all_pairs)}")
    print(f"  - 平均IoU差距: {np.mean([p['iou_gap'] for p in all_pairs]):.4f}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - Annotations: {ann_path}")
    
    return all_pairs


def generate_from_predictions(
    predictions_dir: str,
    gt_dir: str,
    output_dir: str,
    min_iou_gap: float = 0.05
):
    """
    从已有的多个预测结果生成DPO数据集
    
    predictions_dir 结构:
    ├── method_1/
    │   ├── image_001.png
    │   └── ...
    ├── method_2/
    │   └── ...
    └── method_3/
        └── ...
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有预测方法
    methods = [d for d in Path(predictions_dir).iterdir() if d.is_dir()]
    print(f"📊 找到 {len(methods)} 种预测方法: {[m.name for m in methods]}")
    
    # 收集所有图像ID
    all_image_ids = set()
    for method_dir in methods:
        for f in method_dir.glob('*.png'):
            all_image_ids.add(f.stem)
    
    print(f"📊 共 {len(all_image_ids)} 张图像")
    
    all_pairs = []
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    for image_id in tqdm(all_image_ids, desc="构建偏好对"):
        # 加载GT
        gt_path = Path(gt_dir) / f"{image_id}.png"
        if not gt_path.exists():
            gt_path = Path(gt_dir) / f"{image_id}_mask.png"
        if not gt_path.exists():
            continue
            
        gt = np.array(Image.open(gt_path).convert('L'))
        gt = (gt > 127).astype(np.float32)
        
        # 收集所有方法的预测和IoU
        predictions = []
        for method_dir in methods:
            pred_path = method_dir / f"{image_id}.png"
            if not pred_path.exists():
                continue
            
            pred = np.array(Image.open(pred_path).convert('L'))
            pred = (pred > 127).astype(np.float32)
            
            # 调整尺寸
            if pred.shape != gt.shape:
                pred_pil = Image.fromarray((pred * 255).astype(np.uint8))
                pred_pil = pred_pil.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
                pred = np.array(pred_pil) / 255.0
            
            iou = compute_iou(pred, gt)
            predictions.append({
                'mask': pred,
                'iou': iou,
                'method': method_dir.name,
                'path': str(pred_path)
            })
        
        if len(predictions) < 2:
            continue
        
        # 按IoU排序
        predictions.sort(key=lambda x: x['iou'], reverse=True)
        
        # 构建偏好对
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                chosen = predictions[i]
                rejected = predictions[j]
                
                iou_gap = chosen['iou'] - rejected['iou']
                if iou_gap < min_iou_gap:
                    continue
                
                # 保存或引用masks
                chosen_rel = os.path.relpath(chosen['path'], output_dir)
                rejected_rel = os.path.relpath(rejected['path'], output_dir)
                
                # 找到原图
                image_path = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    candidate = Path(gt_dir).parent / 'images' / f"{image_id}{ext}"
                    if candidate.exists():
                        image_path = str(candidate)
                        break
                
                if image_path is None:
                    continue
                
                pair = {
                    'image': os.path.relpath(image_path, output_dir),
                    'chosen_mask': chosen_rel,
                    'rejected_mask': rejected_rel,
                    'chosen_iou': chosen['iou'],
                    'rejected_iou': rejected['iou'],
                    'iou_gap': iou_gap,
                    'chosen_method': chosen['method'],
                    'rejected_method': rejected['method'],
                    'prompt': '<image>Please segment the blood vessels.'
                }
                all_pairs.append(pair)
    
    # 保存
    ann_path = os.path.join(output_dir, 'dpo_annotations.json')
    with open(ann_path, 'w') as f:
        json.dump(all_pairs, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 DPO数据集生成完成!")
    print(f"  - 总偏好对数: {len(all_pairs)}")
    print(f"  - 输出: {ann_path}")
    
    return all_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成DPO偏好对数据集')
    parser.add_argument('--mode', choices=['generate', 'from_predictions'], default='generate',
                        help='生成模式')
    parser.add_argument('--images_dir', type=str, required=True, help='图像目录')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--model_path', type=str, 
                        default='/home/ubuntu/Sa2VA/models/sa2va_vessel_hf',
                        help='Sa2VA模型路径')
    parser.add_argument('--num_samples', type=int, default=5, help='每张图像采样数')
    parser.add_argument('--min_iou_gap', type=float, default=0.05, help='最小IoU差距')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        generate_from_existing_data(
            images_dir=args.images_dir,
            gt_dir=args.gt_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            num_samples=args.num_samples,
            min_iou_gap=args.min_iou_gap
        )
    else:
        generate_from_predictions(
            predictions_dir=args.images_dir,  # 这里是predictions目录
            gt_dir=args.gt_dir,
            output_dir=args.output_dir,
            min_iou_gap=args.min_iou_gap
        )
