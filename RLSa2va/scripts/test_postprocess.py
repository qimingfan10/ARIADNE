#!/usr/bin/env python3
"""
测试后处理对Dice的影响（不训练）
"""

import os
import sys
import json
import random
import numpy as np
import torch
from PIL import Image
import cv2
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/home/ubuntu/Sa2VA')

# 配置
HF_MODEL_PATH = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data"

print("=" * 60)
print("🔬 后处理优化测试")
print("=" * 60)

# 加载模型
print("\n📥 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.eval()
print("✅ Model loaded")

# 加载数据
with open(f"{DATA_ROOT}/annotations.json") as f:
    anns = json.load(f)
valid = [a for a in anns if 'mask' in a and len(a['mask']) > 0]
random.seed(42)
samples = random.sample(valid, 10)  # 与evaluate_10_images.py一致

def compute_dice(pred, gt):
    inter = (pred & gt).sum()
    return (2 * inter) / (pred.sum() + gt.sum() + 1e-8)

def evaluate(threshold=0.5, morph_open=0, morph_close=0):
    """评估不同后处理参数"""
    dices = []
    
    for s in samples:
        img = Image.open(f"{DATA_ROOT}/images/{s['image']}").convert('RGB')
        w, h = img.size
        
        # GT mask
        gt = np.zeros((h, w), dtype=np.uint8)
        for m in s['mask']:
            if len(m) >= 6:
                pts = np.array(m).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt, [pts], 255)
        
        # 预测
        with torch.no_grad():
            out = model.predict_forward(
                image=img,
                text='<image>Please segment the blood vessel.',
                tokenizer=tokenizer,
                processor=None,
            )
        
        pred = out['prediction_masks'][0][0]  # [seg_idx][frame_idx]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        pred = pred.astype(np.float32)
        if pred.max() <= 1.0:
            pred = pred * 255
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 后处理
        pred_binary = (pred > threshold * 255).astype(np.uint8)
        
        if morph_close > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
            pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel)
        
        if morph_open > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
            pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_OPEN, kernel)
        
        gt_binary = (gt > 127).astype(np.uint8)
        dice = compute_dice(pred_binary, gt_binary)
        dices.append(dice)
    
    return np.mean(dices)

print("\n📊 Testing different parameters...")
print("-" * 60)

# 测试不同阈值
print("\n1. 阈值测试 (无形态学):")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    dice = evaluate(threshold=thresh)
    marker = " ★" if dice > 0.82 else ""
    print(f"   threshold={thresh}: Dice={dice:.4f}{marker}")

# 测试形态学操作
print("\n2. 形态学操作测试 (threshold=0.5):")
for close_k in [0, 3, 5, 7]:
    for open_k in [0, 3, 5]:
        dice = evaluate(threshold=0.5, morph_open=open_k, morph_close=close_k)
        marker = " ★" if dice > 0.82 else ""
        print(f"   close={close_k}, open={open_k}: Dice={dice:.4f}{marker}")

# 组合测试
print("\n3. 最佳组合搜索:")
best_dice = 0
best_params = {}
for thresh in [0.4, 0.45, 0.5, 0.55, 0.6]:
    for close_k in [0, 3, 5]:
        for open_k in [0, 3]:
            dice = evaluate(threshold=thresh, morph_open=open_k, morph_close=close_k)
            if dice > best_dice:
                best_dice = dice
                best_params = {'threshold': thresh, 'close': close_k, 'open': open_k}

print(f"\n🎯 最佳结果: Dice={best_dice:.4f}")
print(f"   参数: threshold={best_params['threshold']}, close={best_params['close']}, open={best_params['open']}")
print(f"   Baseline: 0.8191")
print(f"   变化: {best_dice - 0.8191:+.4f}")
