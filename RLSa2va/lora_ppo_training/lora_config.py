"""
LoRA配置
使用PEFT库进行低秩适配
"""

from peft import LoraConfig, TaskType, get_peft_model
import torch


def get_lora_config(
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=None,
    bias="none"
):
    """
    创建LoRA配置
    
    Args:
        lora_rank: LoRA秩（越大越多参数，推荐16-64）
        lora_alpha: LoRA缩放因子（通常是rank的2倍）
        lora_dropout: Dropout率
        target_modules: 要应用LoRA的模块名称
        bias: 是否训练bias
    
    Returns:
        LoraConfig对象
    """
    if target_modules is None:
        # 默认：Attention层的Q/K/V/O投影
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            # 可选：FFN层
            # "gate_proj",
            # "up_proj",
            # "down_proj"
        ]
    
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    
    return config


def apply_lora_to_model(model, lora_config):
    """
    将LoRA应用到模型
    
    Args:
        model: 基础模型
        lora_config: LoRA配置
    
    Returns:
        带LoRA的模型
    """
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model


def save_lora_weights(model, output_path):
    """保存LoRA权重"""
    model.save_pretrained(output_path)
    print(f"✅ LoRA权重已保存至: {output_path}")


def load_lora_weights(model, lora_path):
    """加载LoRA权重"""
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, lora_path)
    print(f"✅ LoRA权重已加载: {lora_path}")
    return model


def merge_lora_weights(model, output_path=None):
    """
    合并LoRA权重到基础模型
    用于部署时减少推理开销
    """
    model = model.merge_and_unload()
    
    if output_path is not None:
        model.save_pretrained(output_path)
        print(f"✅ 合并后的模型已保存至: {output_path}")
    
    return model


def get_trainable_parameters_info(model):
    """获取可训练参数信息"""
    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    info = {
        'trainable_params': trainable_params,
        'all_params': all_params,
        'trainable_percentage': 100 * trainable_params / all_params
    }
    
    return info


# 预设配置
LORA_CONFIGS = {
    'small': {
        'lora_rank': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'target_modules': ["q_proj", "v_proj"]  # 仅Q和V
    },
    'medium': {
        'lora_rank': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.05,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
    },
    'large': {
        'lora_rank': 64,
        'lora_alpha': 128,
        'lora_dropout': 0.1,
        'target_modules': [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    }
}


def get_preset_lora_config(preset='medium'):
    """获取预设的LoRA配置"""
    if preset not in LORA_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(LORA_CONFIGS.keys())}")
    
    return get_lora_config(**LORA_CONFIGS[preset])
