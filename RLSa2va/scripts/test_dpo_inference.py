#!/usr/bin/env python3
"""
直接测试DPO模型推理效果
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/ubuntu/Sa2VA')

def load_model_with_checkpoint(config_path, checkpoint_path):
    """加载模型并应用checkpoint"""
    from mmengine.config import Config
    from mmengine.registry import MODELS
    
    print("📁 加载配置...")
    cfg = Config.fromfile(config_path)
    
    print("🏗️ 构建模型（包含基础权重）...")
    model = MODELS.build(cfg.model)
    
    # 模型构建时已经加载了基础权重，现在加载DPO训练的LoRA权重
    print(f"📥 加载DPO LoRA权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 只加载LoRA相关的权重
    model_state = model.state_dict()
    loaded = 0
    lora_loaded = 0
    for key, value in state_dict.items():
        if key in model_state and value.shape == model_state[key].shape:
            model_state[key] = value
            loaded += 1
            if 'lora' in key.lower():
                lora_loaded += 1
    
    model.load_state_dict(model_state, strict=False)
    print(f"   加载了 {loaded} 个参数 (其中LoRA: {lora_loaded})")
    
    model.eval()
    
    # 使用多GPU
    import torch.distributed as dist
    if torch.cuda.device_count() > 1:
        print(f"   使用 {torch.cuda.device_count()} 个GPU")
        # 使用device_map自动分配
        from accelerate import dispatch_model, infer_auto_device_map
        device_map = infer_auto_device_map(model, max_memory={i: "22GiB" for i in range(torch.cuda.device_count())})
        model = dispatch_model(model, device_map=device_map)
    else:
        model.to(torch.bfloat16)
        model.cuda()
    
    return model

def inference_single(model, image_path, prompt="Please segment the blood vessel in this image."):
    """对单张图片进行推理"""
    from transformers import AutoTokenizer
    
    image = Image.open(image_path).convert('RGB')
    
    # 获取tokenizer
    tokenizer = model.mllm.tokenizer
    
    with torch.no_grad():
        try:
            # 使用模型的predict_forward方法
            result = model.mllm.model.predict_forward(
                image=image,
                text=f"<image>\n{prompt}",
                tokenizer=tokenizer,
            )
            
            pred_text = result.get('prediction', '')
            masks = result.get('prediction_masks', [])
            
            return {
                'text': pred_text,
                'masks': masks,
                'success': len(masks) > 0
            }
        except Exception as e:
            print(f"   推理错误: {e}")
            return {'text': '', 'masks': [], 'success': False, 'error': str(e)}

def visualize_result(image_path, result, output_path):
    """可视化结果"""
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img_array)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    if result['masks']:
        mask = result['masks'][0]
        if len(mask.shape) == 3:
            mask = mask[0]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Overlay mask
        overlay = img_array.copy().astype(float)
        mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (img_array.shape[1], img_array.shape[0]), Image.NEAREST)) / 255.0
        
        overlay[:, :, 1] = np.clip(overlay[:, :, 1] + mask_resized * 100, 0, 255)
        
        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title(f'Prediction\n{result["text"][:50]}...')
    else:
        axes[1].imshow(img_array)
        axes[1].set_title(f'No mask\n{result["text"][:50]}...')
    
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   保存: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='/home/ubuntu/Sa2VA/work_dirs/dpo_vessel_training/iter_1224.pth')
    parser.add_argument('--config', type=str,
                       default='/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_dpo_finetune_v3.py')
    parser.add_argument('--test_dir', type=str,
                       default='/home/ubuntu/Sa2VA/data/dpo_vessel/images')
    parser.add_argument('--output_dir', type=str,
                       default='/home/ubuntu/Sa2VA/work_dirs/dpo_vessel_training/test_results')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 DPO模型推理测试")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model_with_checkpoint(args.config, args.checkpoint)
    
    # 获取测试图片
    from pathlib import Path
    test_images = list(Path(args.test_dir).glob("*.jpg"))[:args.num_samples]
    
    print(f"\n📸 测试 {len(test_images)} 张图片...")
    
    results = []
    for i, img_path in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] {img_path.name}")
        
        result = inference_single(model, str(img_path))
        results.append(result)
        
        print(f"   输出: {result['text'][:80]}...")
        print(f"   有mask: {result['success']}")
        
        # 可视化
        output_path = os.path.join(args.output_dir, f"result_{i+1}.png")
        visualize_result(str(img_path), result, output_path)
    
    # 统计
    success_count = sum(1 for r in results if r['success'])
    print(f"\n" + "=" * 60)
    print(f"📊 测试结果: {success_count}/{len(results)} 成功生成mask")
    print(f"   结果保存在: {args.output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
