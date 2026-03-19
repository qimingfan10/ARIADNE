#!/usr/bin/env python3
"""
合并DPO训练的LoRA权重到基础模型
"""

import os
import sys
import torch
import argparse
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/Sa2VA')

def merge_lora_weights(
    base_model_path: str,
    lora_checkpoint_path: str,
    output_path: str,
    config_path: str = None
):
    """合并LoRA权重到基础模型"""
    from mmengine.config import Config
    from mmengine.registry import MODELS
    
    print("=" * 60)
    print("🔧 合并DPO LoRA权重")
    print("=" * 60)
    
    # 加载配置
    if config_path is None:
        config_path = '/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_dpo_finetune_v3.py'
    
    print(f"\n📁 加载配置: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # 构建模型
    print("\n🏗️ 构建模型...")
    model = MODELS.build(cfg.model)
    
    # 加载LoRA checkpoint
    print(f"\n📥 加载LoRA权重: {lora_checkpoint_path}")
    checkpoint = torch.load(lora_checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"   Checkpoint包含 {len(state_dict)} 个参数")
    
    # 过滤并加载权重
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for key, value in state_dict.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                model_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key}: {value.shape} vs {model_state[key].shape}")
        else:
            skipped_keys.append(f"{key}: not in model")
    
    print(f"   成功加载: {len(loaded_keys)} 个参数")
    print(f"   跳过: {len(skipped_keys)} 个参数")
    
    if skipped_keys[:5]:
        print("   跳过的参数示例:")
        for k in skipped_keys[:5]:
            print(f"     - {k[:80]}")
    
    model.load_state_dict(model_state)
    
    # 合并LoRA权重
    print("\n🔀 合并LoRA权重...")
    
    try:
        # 检查是否有LoRA
        if hasattr(model.mllm, 'model') and hasattr(model.mllm.model, 'language_model'):
            llm = model.mllm.model.language_model
            if hasattr(llm, 'merge_and_unload'):
                print("   使用PEFT merge_and_unload...")
                model.mllm.model.language_model = llm.merge_and_unload()
                print("   ✅ LoRA合并成功!")
            elif hasattr(llm, 'base_model'):
                print("   使用手动LoRA合并...")
                # 手动合并
                for name, module in llm.named_modules():
                    if hasattr(module, 'merge'):
                        module.merge()
                print("   ✅ LoRA合并成功!")
            else:
                print("   ⚠️ 未检测到LoRA层，跳过合并")
    except Exception as e:
        print(f"   ⚠️ LoRA合并失败: {e}")
        print("   继续保存未合并的权重...")
    
    # 保存合并后的模型
    os.makedirs(output_path, exist_ok=True)
    
    # 保存state_dict
    output_file = os.path.join(output_path, 'pytorch_model.bin')
    print(f"\n💾 保存模型: {output_file}")
    torch.save(model.state_dict(), output_file)
    
    # 复制tokenizer
    tokenizer_src = '/home/ubuntu/Sa2VA/tokenizer_with_special_tokens'
    if os.path.exists(tokenizer_src):
        import shutil
        for f in os.listdir(tokenizer_src):
            src = os.path.join(tokenizer_src, f)
            dst = os.path.join(output_path, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"   ✅ Tokenizer已复制")
    
    print("\n" + "=" * 60)
    print("✅ 合并完成!")
    print(f"   输出路径: {output_path}")
    print("=" * 60)
    
    return model

def quick_test(model, test_image_path):
    """快速测试合并后的模型"""
    print("\n🧪 快速测试...")
    
    from PIL import Image
    
    image = Image.open(test_image_path).convert('RGB')
    
    # 简单检查模型结构
    print(f"   模型类型: {type(model)}")
    print(f"   MLLM类型: {type(model.mllm)}")
    
    # 检查参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/ubuntu/Sa2VA/work_dirs/dpo_vessel_training/iter_1224.pth')
    parser.add_argument('--output', type=str,
                       default='/home/ubuntu/Sa2VA/work_dirs/dpo_vessel_training/merged_model')
    parser.add_argument('--config', type=str,
                       default='/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_dpo_finetune_v3.py')
    parser.add_argument('--test', action='store_true', help='运行快速测试')
    args = parser.parse_args()
    
    model = merge_lora_weights(
        base_model_path=None,  # 从config获取
        lora_checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config
    )
    
    if args.test:
        test_image = '/home/ubuntu/Sa2VA/data/dpo_vessel/images/An Cong Xue(0000932433)_1-3_1_051C3E6A_frame_000011.jpg'
        if os.path.exists(test_image):
            quick_test(model, test_image)

if __name__ == '__main__':
    main()
