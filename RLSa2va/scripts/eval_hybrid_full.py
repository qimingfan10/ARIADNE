#!/usr/bin/env python3
"""评估混合策略在全部样本上的Dice"""
import os, json, torch, numpy as np, cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 加载两个模型
print("Loading V8...")
v8_path = '/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_dpo_v8/step_100'
tokenizer_v8 = AutoTokenizer.from_pretrained(v8_path, trust_remote_code=True)
model_v8 = AutoModelForCausalLM.from_pretrained(v8_path, trust_remote_code=True, torch_dtype='auto', device_map='auto', low_cpu_mem_usage=True)
model_v8.eval()

print("Loading V13...")
v13_path = '/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_hard_v13/final'
tokenizer_v13 = AutoTokenizer.from_pretrained(v13_path, trust_remote_code=True)
model_v13 = AutoModelForCausalLM.from_pretrained(v13_path, trust_remote_code=True, torch_dtype='auto', device_map='auto', low_cpu_mem_usage=True)
model_v13.eval()

eval_root = '/home/ubuntu/Sa2VA/data/merged_vessel_data'
with open(f'{eval_root}/annotations.json') as f:
    anns = json.load(f)
valid = [a for a in anns if 'mask' in a and len(a['mask']) > 0]

print(f'Evaluating Hybrid on ALL {len(valid)} samples...')
HARD_THRESHOLD = 0.75

v8_dices = []
v13_dices = []
hybrid_dices = []
hard_count = 0
easy_count = 0

for ann in tqdm(valid):
    try:
        img = Image.open(f'{eval_root}/images/{ann["image"]}').convert('RGB')
        w, h = img.size
        gt = np.zeros((h, w), dtype=np.uint8)
        for m in ann['mask']:
            if len(m) >= 6:
                pts = np.array(m).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt, [pts], 255)
        gt_binary = (gt > 127).astype(np.uint8)
        
        # V8预测
        with torch.no_grad():
            out = model_v8.predict_forward(image=img, text='<image>Please segment the blood vessel.', tokenizer=tokenizer_v8, processor=None)
        pred = out['prediction_masks'][0][0]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        pred = pred.astype(np.float32)
        if pred.max() <= 1.0:
            pred = pred * 255
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_v8 = (pred > 127).astype(np.uint8)
        inter = (pred_v8 & gt_binary).sum()
        dice_v8 = (2 * inter) / (pred_v8.sum() + gt_binary.sum() + 1e-8)
        v8_dices.append(dice_v8)
        
        # V13预测
        with torch.no_grad():
            out = model_v13.predict_forward(image=img, text='<image>Please segment the blood vessel.', tokenizer=tokenizer_v13, processor=None)
        pred = out['prediction_masks'][0][0]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        pred = pred.astype(np.float32)
        if pred.max() <= 1.0:
            pred = pred * 255
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_v13 = (pred > 127).astype(np.uint8)
        inter = (pred_v13 & gt_binary).sum()
        dice_v13 = (2 * inter) / (pred_v13.sum() + gt_binary.sum() + 1e-8)
        v13_dices.append(dice_v13)
        
        # 混合策略
        if dice_v8 >= HARD_THRESHOLD:
            hybrid_dices.append(dice_v8)
            easy_count += 1
        else:
            hybrid_dices.append(dice_v13)
            hard_count += 1
            
    except Exception as e:
        print(f"Error: {e}")

print(f'\n===== Full Evaluation Results =====')
print(f'  Total samples: {len(v8_dices)}')
print(f'  Hard samples: {hard_count}')
print(f'  Easy samples: {easy_count}')
print(f'\n  V8 Mean Dice:     {np.mean(v8_dices):.4f}')
print(f'  V13 Mean Dice:    {np.mean(v13_dices):.4f}')
print(f'  Hybrid Mean Dice: {np.mean(hybrid_dices):.4f}')
print(f'\n  Hybrid vs V8:  {np.mean(hybrid_dices) - np.mean(v8_dices):+.4f}')
print(f'  Hybrid vs V13: {np.mean(hybrid_dices) - np.mean(v13_dices):+.4f}')
