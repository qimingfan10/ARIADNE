#!/usr/bin/env python3
"""
将DPO训练的LoRA权重应用到HuggingFace模型
"""

import os
import sys
import torch
import shutil
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/Sa2VA')

def apply_dpo_lora(base_hf_path, dpo_checkpoint_path, output_path):
    """将DPO LoRA权重应用到HF模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import save_file, load_file
    
    print("=" * 60)
    print("🔧 应用DPO LoRA权重")
    print("=" * 60)
    
    # 加载DPO checkpoint
    print(f"\n📥 加载DPO checkpoint: {dpo_checkpoint_path}")
    dpo_ckpt = torch.load(dpo_checkpoint_path, map_location='cpu', weights_only=False)
    dpo_state = dpo_ckpt.get('state_dict', dpo_ckpt)
    
    # 统计LoRA参数
    lora_params = {k: v for k, v in dpo_state.items() if 'lora' in k.lower()}
    print(f"   DPO LoRA参数: {len(lora_params)}")
    
    # 加载基础HF模型
    print(f"\n📥 加载基础模型: {base_hf_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_hf_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 复制基础模型文件
    print(f"\n📋 复制基础模型文件...")
    for f in os.listdir(base_hf_path):
        src = os.path.join(base_hf_path, f)
        dst = os.path.join(output_path, f)
        if os.path.isfile(src) and not f.endswith('.safetensors'):
            shutil.copy2(src, dst)
    
    # 获取模型state_dict
    model_state = model.state_dict()
    
    # 映射DPO参数名到HF参数名
    # DPO格式: mllm.model.language_model.base_model.model.xxx
    # HF格式: language_model.model.xxx
    
    print(f"\n🔀 应用LoRA权重 (手动合并)...")
    
    # LoRA配置
    lora_alpha = 128  # 从配置中获取
    lora_r = 64
    scaling = lora_alpha / lora_r
    
    # 收集LoRA A和B矩阵
    lora_pairs = {}  # {base_key: {'A': tensor, 'B': tensor}}
    
    for dpo_key, dpo_value in lora_params.items():
        # 解析key: mllm.model.language_model.base_model.model.model.layers.X.xxx.lora_A.default.weight
        if 'lora_A' in dpo_key:
            base_key = dpo_key.replace('.lora_A.default.weight', '')
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]['A'] = dpo_value
        elif 'lora_B' in dpo_key:
            base_key = dpo_key.replace('.lora_B.default.weight', '')
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]['B'] = dpo_value
    
    print(f"   找到 {len(lora_pairs)} 个LoRA层对")
    
    # 合并LoRA到基础权重
    applied = 0
    for base_key, lora_dict in lora_pairs.items():
        if 'A' not in lora_dict or 'B' not in lora_dict:
            continue
        
        # 转换key到HF格式
        # mllm.model.language_model.base_model.model.model.layers.X.xxx
        # -> language_model.model.layers.X.xxx
        hf_key = base_key
        
        # 移除PEFT前缀
        if '.base_model.model.' in hf_key:
            hf_key = hf_key.replace('.base_model.model.', '.')
        
        # 移除mllm前缀
        if hf_key.startswith('mllm.model.'):
            hf_key = hf_key[len('mllm.model.'):]
        
        # 添加.weight后缀
        hf_key = hf_key + '.weight'
        
        if hf_key in model_state:
            # W' = W + B @ A * scaling
            lora_A = lora_dict['A'].float()
            lora_B = lora_dict['B'].float()
            delta = (lora_B @ lora_A) * scaling
            
            original = model_state[hf_key].float()
            if delta.shape == original.shape:
                model_state[hf_key] = (original + delta).to(torch.bfloat16)
                applied += 1
            else:
                print(f"   形状不匹配: {hf_key} - delta {delta.shape} vs orig {original.shape}")
        else:
            # 尝试去掉一层model
            alt_key = hf_key.replace('language_model.model.model.', 'language_model.model.')
            if alt_key in model_state:
                lora_A = lora_dict['A'].float()
                lora_B = lora_dict['B'].float()
                delta = (lora_B @ lora_A) * scaling
                original = model_state[alt_key].float()
                if delta.shape == original.shape:
                    model_state[alt_key] = (original + delta).to(torch.bfloat16)
                    applied += 1
    
    print(f"   成功合并: {applied}/{len(lora_pairs)} LoRA层")
    
    # 加载更新后的权重
    model.load_state_dict(model_state, strict=False)
    
    # 保存模型
    print(f"\n💾 保存模型到: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    
    # 复制tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_hf_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✅ 完成! 模型保存在: {output_path}")
    
    return output_path

def test_model(model_path, test_image):
    """测试模型"""
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n🧪 测试模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda().eval()
    
    image = Image.open(test_image).convert('RGB')
    prompt = '<image>\nPlease segment the blood vessel in this image.'
    
    with torch.no_grad():
        result = model.predict_forward(
            image=image,
            text=prompt,
            tokenizer=tokenizer,
        )
    
    print(f"   输出: {result['prediction']}")
    print(f"   有mask: {len(result.get('prediction_masks', [])) > 0}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192_hf')
    parser.add_argument('--dpo_checkpoint', default='/home/ubuntu/Sa2VA/work_dirs/dpo_vessel_training/iter_1224.pth')
    parser.add_argument('--output', default='/home/ubuntu/Sa2VA/work_dirs/dpo_vessel_training/dpo_model_hf')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    output_path = apply_dpo_lora(args.base_model, args.dpo_checkpoint, args.output)
    
    if args.test:
        test_image = '/home/ubuntu/Sa2VA/data/dpo_vessel/images/An Cong Xue(0000932433)_1-3_1_051C3E6A_frame_000011.jpg'
        test_model(output_path, test_image)

if __name__ == '__main__':
    main()
