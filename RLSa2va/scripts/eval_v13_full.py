#!/usr/bin/env python3
"""评估V13在全部样本上的Dice"""
import os, json, torch, numpy as np, cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_path = '/home/ubuntu/Sa2VA/work_dirs/sa2va_26b_hard_v13/final'
print("Loading V13...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype='auto', device_map='auto', low_cpu_mem_usage=True)
model.eval()

eval_root = '/home/ubuntu/Sa2VA/data/merged_vessel_data'
with open(f'{eval_root}/annotations.json') as f:
    anns = json.load(f)
valid = [a for a in anns if 'mask' in a and len(a['mask']) > 0]

print(f'Evaluating V13 on ALL {len(valid)} samples...')
dices = []
for ann in tqdm(valid):
    try:
        img = Image.open(f'{eval_root}/images/{ann["image"]}').convert('RGB')
        w, h = img.size
        gt = np.zeros((h, w), dtype=np.uint8)
        for m in ann['mask']:
            if len(m) >= 6:
                pts = np.array(m).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt, [pts], 255)
        with torch.no_grad():
            out = model.predict_forward(image=img, text='<image>Please segment the blood vessel.', tokenizer=tokenizer, processor=None)
        pred = out['prediction_masks'][0][0]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        pred = pred.astype(np.float32)
        if pred.max() <= 1.0:
            pred = pred * 255
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_binary = (pred > 127).astype(np.uint8)
        gt_binary = (gt > 127).astype(np.uint8)
        inter = (pred_binary & gt_binary).sum()
        dice = (2 * inter) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
        dices.append(dice)
    except Exception as e:
        print(f"Error: {e}")

print(f'\n===== V13 Full Evaluation =====')
print(f'  Samples: {len(dices)}')
print(f'  Mean Dice: {np.mean(dices):.4f}')
print(f'  Std: {np.std(dices):.4f}')
print(f'  V8 baseline: 0.81')
print(f'  Improvement: {np.mean(dices) - 0.81:+.4f}')
