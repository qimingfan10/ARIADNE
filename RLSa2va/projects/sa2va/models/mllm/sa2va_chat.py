"""
Sa2VAChatModel MLLM Wrapper for training
支持加载HuggingFace格式的Sa2VA模型进行DPO训练
"""

import torch
import torch.nn as nn
from typing import List, Optional
from mmengine.logging import print_log
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


@MODELS.register_module()
class Sa2VAChatMLLM(BaseModel):
    """
    Sa2VAChatModel MLLM Wrapper
    用于加载和训练HuggingFace格式的Sa2VA模型
    """
    
    def __init__(
        self,
        model_path: str,
        freeze_llm: bool = False,
        freeze_visual_encoder: bool = True,
        llm_lora: Optional[dict] = None,
        visual_encoder_lora: Optional[dict] = None,
        quantization_llm: bool = False,
        pretrained_pth: Optional[str] = None,
    ):
        super().__init__()
        
        print_log(f'Sa2VAChatMLLM: Loading model from {model_path}', logger='current')
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        print_log(f'Sa2VAChatMLLM: Model loaded successfully', logger='current')
        
        # 冻结设置
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        
        # LoRA配置
        self.llm_lora_config = llm_lora
        self.visual_encoder_lora_config = visual_encoder_lora
        
        # 冻结视觉编码器
        if freeze_visual_encoder and hasattr(self.model, 'vision_model'):
            self.model.vision_model.requires_grad_(False)
            print_log('Sa2VAChatMLLM: Frozen vision_model', logger='current')
        
        # 冻结LLM
        if freeze_llm and hasattr(self.model, 'language_model'):
            self.model.language_model.requires_grad_(False)
            print_log('Sa2VAChatMLLM: Frozen language_model', logger='current')
    
    def add_special_tokens(self, tokenizer, special_tokens: List[str]) -> None:
        """添加特殊token"""
        print_log(f'Sa2VAChatMLLM: Adding special tokens {special_tokens}', logger='current')
        
        current_vocab_size = len(tokenizer)
        num_new = tokenizer.add_tokens(special_tokens, special_tokens=True)
        
        if num_new > 0:
            # Resize embeddings
            if hasattr(self.model, 'language_model'):
                self.model.language_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            elif hasattr(self.model, 'resize_token_embeddings'):
                self.model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            print_log(f'Sa2VAChatMLLM: Added {num_new} tokens, resized to {len(tokenizer)}', logger='current')
        
        self.tokenizer = tokenizer
    
    def manual_prepare_llm_for_lora(self):
        """应用LoRA到LLM"""
        if not self.use_llm_lora or self.llm_lora_config is None:
            print_log('Sa2VAChatMLLM: No LoRA config, skipping', logger='current')
            return
        
        print_log('Sa2VAChatMLLM: Applying LoRA to language_model', logger='current')
        
        # 获取LoRA配置
        lora_config_dict = self.llm_lora_config.copy()
        lora_type = lora_config_dict.pop('type', LoraConfig)
        
        # 设置target_modules
        if 'target_modules' not in lora_config_dict:
            lora_config_dict['target_modules'] = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ]
        
        lora_config = lora_type(**lora_config_dict)
        
        # 应用LoRA
        if hasattr(self.model, 'language_model'):
            self.model.language_model = get_peft_model(self.model.language_model, lora_config)
            print_log('Sa2VAChatMLLM: LoRA applied to language_model', logger='current')
        else:
            print_log('Sa2VAChatMLLM: No language_model found, applying to whole model', logger='current')
            self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print_log(f'Sa2VAChatMLLM: Trainable {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)', logger='current')
    
    def get_embedding_size(self):
        """获取embedding大小"""
        if hasattr(self.model, 'language_model'):
            return self.model.language_model.config.hidden_size
        return self.model.config.hidden_size
    
    def forward(self, *args, **kwargs):
        """Forward pass"""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate"""
        return self.model.generate(*args, **kwargs)
    
    def predict_forward(self, *args, **kwargs):
        """Predict forward for inference"""
        return self.model.predict_forward(*args, **kwargs)
