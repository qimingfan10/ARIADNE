#!/usr/bin/env python3
"""
混合评估：V8 + V12 自适应选择

策略：
1. 默认使用V8预测
2. 如果V8预测Dice >= 0.75（easy），使用V8结果
3. 如果V8预测Dice < 0.75（hard），换成V12预测
"""

import os
import sys
import json
import torch
import numpy as np
import cv2
import random
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/home/ubuntu/Sa2VA')


def compute_dice_np(pred, gt):
    """计算Dice分数（numpy版本）"""
    inter = (pred & gt).sum()
    return (2 * inter) / (pred.sum() + gt.sum() + 1e-8)


def predict_mask(model, tokenizer, image):
    """使用模型预测mask"""
    w, h = image.size
    
    with torch.no_grad():
        out = model.predict_forward(
            image=image,
            text='<image>Please segment the blood vessel.',
            tokenizer=tokenizer,
            processor=None,
        )
    
    pred = out['prediction_masks'][0][0]
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    pred = pred.astype(np.float32)
    if pred.max() <= 1.0:
        pred = pred * 255
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    pred_binary = (pred > 127).astype(np.uint8)
    
    return pred_binary


def load_gt_mask(ann, data_root, image_size):
    """加载GT mask"""
    w, h = image_size
    gt = np.zeros((h, w), dtype=np.uint8)
    for m in ann['mask']:
        if len(m) >= 6:
            pts = np.array(m).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt, [pts], 255)
    return (gt > 127).astype(np.uint8)


class HybridEvaluator:
    def __init__(self):
        self.v8_path = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v8/step_100"
        self.v13_path = "/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_hard_v13/final"
        self.data_root = "/home/ubuntu/Sa2VA/data/merged_vessel_data"
        self.hard_threshold = 0.75
        
        print("=" * 60)
        print("🔀 Hybrid Evaluation: V8 + V13")
        print(f"   Hard threshold: {self.hard_threshold}")
        print("=" * 60)
        
        self._load_models()
    
    def _load_models(self):
        print("\n📥 Loading V8 model...")
        self.tokenizer_v8 = AutoTokenizer.from_pretrained(
            self.v8_path, trust_remote_code=True
        )
        self.model_v8 = AutoModelForCausalLM.from_pretrained(
            self.v8_path,
            torch_dtype="auto",
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model_v8.eval()
        
        print("📥 Loading V13 model...")
        self.tokenizer_v13 = AutoTokenizer.from_pretrained(
            self.v13_path, trust_remote_code=True
        )
        self.model_v13 = AutoModelForCausalLM.from_pretrained(
            self.v13_path,
            torch_dtype="auto",
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model_v13.eval()
        
        print("✅ Models loaded!")
    
    def evaluate(self, n_samples=10, seed=42):
        """运行混合评估"""
        print(f"\n📊 Evaluating on {n_samples} samples (seed={seed})...")
        
        with open(f'{self.data_root}/annotations.json') as f:
            anns = json.load(f)
        valid = [a for a in anns if 'mask' in a and len(a['mask']) > 0]
        
        random.seed(seed)
        samples = random.sample(valid, min(n_samples, len(valid)))
        
        results = {
            'v8_only': [],
            'v13_only': [],
            'hybrid': [],
            'easy_count': 0,
            'hard_count': 0,
        }
        
        for ann in tqdm(samples, desc="Evaluating"):
            try:
                img_path = f'{self.data_root}/images/{ann["image"]}'
                img = Image.open(img_path).convert('RGB')
                gt = load_gt_mask(ann, self.data_root, img.size)
                
                # V8预测
                pred_v8 = predict_mask(self.model_v8, self.tokenizer_v8, img)
                dice_v8 = compute_dice_np(pred_v8, gt)
                results['v8_only'].append(dice_v8)
                
                # V13预测
                pred_v13 = predict_mask(self.model_v13, self.tokenizer_v13, img)
                dice_v13 = compute_dice_np(pred_v13, gt)
                results['v13_only'].append(dice_v13)
                
                # 混合策略
                if dice_v8 >= self.hard_threshold:
                    # Easy样本，使用V8
                    results['hybrid'].append(dice_v8)
                    results['easy_count'] += 1
                else:
                    # Hard样本，使用V13
                    results['hybrid'].append(dice_v13)
                    results['hard_count'] += 1
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # 打印结果
        print("\n" + "=" * 60)
        print("📊 Results:")
        print("=" * 60)
        
        print(f"\n{'Method':<15} {'Mean Dice':<12} {'Std':<10}")
        print("-" * 40)
        print(f"{'V8 Only':<15} {np.mean(results['v8_only']):.4f}       {np.std(results['v8_only']):.4f}")
        print(f"{'V13 Only':<15} {np.mean(results['v13_only']):.4f}       {np.std(results['v13_only']):.4f}")
        print(f"{'Hybrid':<15} {np.mean(results['hybrid']):.4f}       {np.std(results['hybrid']):.4f}")
        
        print(f"\n📌 Hybrid breakdown:")
        print(f"   Easy (V8): {results['easy_count']} samples")
        print(f"   Hard (V13): {results['hard_count']} samples")
        
        # 详细对比
        print(f"\n📋 Per-sample comparison (V8 vs V13 vs Hybrid):")
        print("-" * 50)
        for i, (v8, v13, hyb) in enumerate(zip(results['v8_only'], results['v13_only'], results['hybrid'])):
            is_hard = v8 < self.hard_threshold
            marker = "🔴 Hard" if is_hard else "🟢 Easy"
            winner = "V13" if is_hard else "V8"
            print(f"  {i+1:2d}. V8={v8:.4f}, V13={v13:.4f} -> {marker} -> Use {winner}: {hyb:.4f}")
        
        return results


def main():
    evaluator = HybridEvaluator()
    
    # 用多个seed评估
    all_results = []
    for seed in [42, 123, 456]:
        print(f"\n{'='*60}")
        print(f"🎲 Seed: {seed}")
        print("=" * 60)
        results = evaluator.evaluate(n_samples=10, seed=seed)
        all_results.append({
            'seed': seed,
            'v8': np.mean(results['v8_only']),
            'v13': np.mean(results['v13_only']),
            'hybrid': np.mean(results['hybrid']),
        })
    
    # 汇总
    print("\n" + "=" * 60)
    print("📊 Summary across all seeds:")
    print("=" * 60)
    print(f"\n{'Seed':<8} {'V8':<10} {'V13':<10} {'Hybrid':<10}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['seed']:<8} {r['v8']:.4f}     {r['v13']:.4f}     {r['hybrid']:.4f}")
    
    avg_v8 = np.mean([r['v8'] for r in all_results])
    avg_v13 = np.mean([r['v13'] for r in all_results])
    avg_hybrid = np.mean([r['hybrid'] for r in all_results])
    
    print("-" * 40)
    print(f"{'Average':<8} {avg_v8:.4f}     {avg_v13:.4f}     {avg_hybrid:.4f}")
    
    print(f"\n🎯 Hybrid improvement over V8: {avg_hybrid - avg_v8:+.4f}")
    print(f"🎯 Hybrid improvement over V13: {avg_hybrid - avg_v13:+.4f}")


if __name__ == "__main__":
    main()
