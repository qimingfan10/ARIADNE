#!/usr/bin/env python3
"""
使用HuggingFace格式模型评估血管分割效果
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/Sa2VA')

def calculate_iou(pred_mask, gt_mask):
    """计算IoU"""
    pred = (pred_mask > 0.5).astype(float)
    gt = (gt_mask > 127).astype(float)
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return intersection / (union + 1e-8)

def calculate_dice(pred_mask, gt_mask):
    """计算Dice"""
    pred = (pred_mask > 0.5).astype(float)
    gt = (gt_mask > 127).astype(float)
    intersection = (pred * gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

def load_model(model_path):
    """加载HuggingFace模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"📥 加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 单GPU加载
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()
    model.eval()
    
    print(f"   ✅ 模型加载完成")
    return model, tokenizer

def inference(model, tokenizer, image_path, prompt="<image>\nPlease segment the blood vessel in this image."):
    """推理"""
    image = Image.open(image_path).convert('RGB')
    
    with torch.no_grad():
        result = model.predict_forward(
            image=image,
            text=prompt,
            tokenizer=tokenizer,
        )
    
    return {
        'text': result.get('prediction', ''),
        'masks': result.get('prediction_masks', []),
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192_hf')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--output_dir', default='/home/ubuntu/Sa2VA/work_dirs/eval_results')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 血管分割模型评估")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, tokenizer = load_model(args.model_path)
    
    # 加载DPO测试数据
    ann_path = '/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_annotations.json'
    with open(ann_path) as f:
        annotations = json.load(f)
    
    # 随机采样
    import random
    random.seed(42)
    samples = random.sample(annotations, min(args.num_samples, len(annotations)))
    
    print(f"\n📸 评估 {len(samples)} 张图片...")
    
    results = []
    seg_success = 0
    
    for i, ann in enumerate(tqdm(samples, desc="推理中")):
        img_path = os.path.join('/home/ubuntu/Sa2VA/data/dpo_vessel', ann['image'])
        if not os.path.exists(img_path):
            continue
        
        try:
            result = inference(model, tokenizer, img_path)
            has_seg = '[SEG]' in result['text']
            has_mask = len(result['masks']) > 0
            
            if has_seg and has_mask:
                seg_success += 1
                
                # 计算与chosen/rejected的IoU对比
                pred_mask = result['masks'][0]
                if len(pred_mask.shape) == 3:
                    pred_mask = pred_mask[0]
                if isinstance(pred_mask, torch.Tensor):
                    pred_mask = pred_mask.cpu().numpy()
                
                # 加载chosen mask作为参考
                chosen_path = os.path.join('/home/ubuntu/Sa2VA/data/dpo_vessel', ann['chosen_mask'])
                if os.path.exists(chosen_path):
                    chosen_mask = np.array(Image.open(chosen_path).convert('L'))
                    # 需要resize pred_mask到相同尺寸
                    pred_resized = np.array(Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(
                        (chosen_mask.shape[1], chosen_mask.shape[0]), Image.NEAREST)) / 255.0
                    iou_with_chosen = calculate_iou(pred_resized, chosen_mask)
                else:
                    iou_with_chosen = None
            else:
                iou_with_chosen = None
            
            results.append({
                'image': ann['image'],
                'has_seg': has_seg,
                'has_mask': has_mask,
                'text': result['text'][:100],
                'chosen_iou': ann['chosen_iou'],
                'rejected_iou': ann['rejected_iou'],
                'pred_iou_with_chosen': iou_with_chosen,
            })
            
        except Exception as e:
            print(f"\n   错误 [{Path(img_path).name}]: {e}")
            results.append({
                'image': ann['image'],
                'has_seg': False,
                'has_mask': False,
                'error': str(e),
            })
    
    # 统计结果
    print(f"\n" + "=" * 60)
    print("📊 评估结果")
    print("=" * 60)
    
    total = len(results)
    seg_count = sum(1 for r in results if r.get('has_seg', False))
    mask_count = sum(1 for r in results if r.get('has_mask', False))
    
    print(f"总样本数: {total}")
    print(f"生成[SEG]: {seg_count}/{total} ({seg_count/total*100:.1f}%)")
    print(f"生成Mask: {mask_count}/{total} ({mask_count/total*100:.1f}%)")
    
    # IoU统计
    valid_ious = [r['pred_iou_with_chosen'] for r in results if r.get('pred_iou_with_chosen') is not None]
    if valid_ious:
        print(f"\n与Chosen Mask的IoU:")
        print(f"  平均: {np.mean(valid_ious):.4f}")
        print(f"  最小: {np.min(valid_ious):.4f}")
        print(f"  最大: {np.max(valid_ious):.4f}")
    
    # 保存结果
    result_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存: {result_path}")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
