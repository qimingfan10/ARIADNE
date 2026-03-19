#!/usr/bin/env python3
"""
评估DPO训练后的模型效果 - 多GPU版本
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/Sa2VA')

def calculate_iou(pred_mask, gt_mask):
    """计算IoU"""
    pred = pred_mask.flatten() > 0.5
    gt = gt_mask.flatten() > 0.5
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)

def calculate_dice(pred_mask, gt_mask):
    """计算Dice"""
    pred = pred_mask.flatten() > 0.5
    gt = gt_mask.flatten() > 0.5
    intersection = np.logical_and(pred, gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

def load_model(config_path, checkpoint_path):
    """使用DeepSpeed加载模型"""
    from mmengine.config import Config
    from mmengine.registry import MODELS
    
    print("📁 加载配置...")
    cfg = Config.fromfile(config_path)
    
    print("🏗️ 构建模型...")
    # 设置bf16以减少内存
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        model = MODELS.build(cfg.model)
    
    print(f"📥 加载DPO checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 加载LoRA权重
    model_state = model.state_dict()
    loaded, lora_cnt = 0, 0
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v
            loaded += 1
            if 'lora' in k.lower():
                lora_cnt += 1
    
    model.load_state_dict(model_state, strict=False)
    print(f"   加载: {loaded} 参数, LoRA: {lora_cnt}")
    
    model.eval()
    model.to(torch.bfloat16)
    
    # 分配到多GPU
    if torch.cuda.device_count() > 1:
        print(f"   分配到 {torch.cuda.device_count()} GPUs...")
        model = torch.nn.DataParallel(model)
    
    model.cuda()
    return model

def inference(model, image_path, tokenizer):
    """推理单张图片"""
    image = Image.open(image_path).convert('RGB')
    prompt = "<image>\nPlease segment the blood vessel in this image."
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # 获取实际模型（DataParallel包装后）
        actual_model = model.module if hasattr(model, 'module') else model
        
        # 使用Sa2VAModel的predict方法
        if hasattr(actual_model, 'predict'):
            result = actual_model.predict(image, prompt)
        elif hasattr(actual_model, 'generate'):
            # 尝试使用generate方法
            result = actual_model.generate(image, prompt, tokenizer)
        else:
            # 直接使用chat接口
            mllm = actual_model.mllm.model
            response, _ = mllm.chat(
                tokenizer=tokenizer,
                pixel_values=None,  # 会在内部处理
                question=prompt,
                generation_config=dict(max_new_tokens=256),
                history=[],
                return_history=True,
                IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
                IMG_START_TOKEN='<img>',
                IMG_END_TOKEN='</img>',
            )
            return {'text': response, 'masks': []}
    
    return {
        'text': result.get('prediction', '') if isinstance(result, dict) else str(result),
        'masks': result.get('prediction_masks', []) if isinstance(result, dict) else [],
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/home/ubuntu/Sa2VA/work_dirs/dpo_vessel_training/iter_1224.pth')
    parser.add_argument('--config', default='/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_dpo_finetune_v3.py')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 DPO模型推理评估")
    print("=" * 60)
    
    # 加载模型
    model = load_model(args.config, args.checkpoint)
    
    # 获取tokenizer
    actual_model = model.module if hasattr(model, 'module') else model
    tokenizer = actual_model.mllm.tokenizer
    
    # 加载测试数据
    ann_path = '/home/ubuntu/Sa2VA/data/dpo_vessel/dpo_annotations.json'
    with open(ann_path) as f:
        annotations = json.load(f)
    
    # 随机选择样本
    import random
    random.seed(42)
    samples = random.sample(annotations, min(args.num_samples, len(annotations)))
    
    print(f"\n📸 测试 {len(samples)} 张图片...")
    
    results = []
    for i, ann in enumerate(samples):
        img_path = os.path.join('/home/ubuntu/Sa2VA/data/dpo_vessel', ann['image'])
        if not os.path.exists(img_path):
            continue
        
        print(f"\n[{i+1}/{len(samples)}] {Path(img_path).name}")
        
        try:
            result = inference(model, img_path, tokenizer)
            has_mask = len(result['masks']) > 0
            print(f"   输出: {result['text'][:60]}...")
            print(f"   生成mask: {'✓' if has_mask else '✗'}")
            
            results.append({
                'image': ann['image'],
                'has_mask': has_mask,
                'text': result['text'],
                'chosen_iou': ann['chosen_iou'],
                'rejected_iou': ann['rejected_iou'],
            })
        except Exception as e:
            print(f"   错误: {e}")
    
    # 统计
    success = sum(1 for r in results if r['has_mask'])
    print(f"\n" + "=" * 60)
    print(f"📊 评估结果")
    print(f"   成功生成mask: {success}/{len(results)}")
    print("=" * 60)

if __name__ == '__main__':
    main()
