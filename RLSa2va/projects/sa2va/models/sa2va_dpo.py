"""
Sa2VA DPO Model
用于DPO训练的模型，在Sa2VAModel基础上添加DPO损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from mmengine.model import BaseModel
from xtuner.registry import BUILDER

from .sa2va import Sa2VAModel


class Sa2VADPOModel(Sa2VAModel):
    """Sa2VA模型 + DPO训练支持"""
    
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 template=None,
                 arch_type: Literal['intern_vl', 'qwen', 'llava'] = 'intern_vl',
                 training_bs: int = 0,
                 # DPO参数
                 dpo_beta: float = 0.1,
                 use_dpo_loss: bool = True,
                 ):
        
        super().__init__(
            mllm=mllm,
            tokenizer=tokenizer,
            grounding_encoder=grounding_encoder,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            torch_dtype=torch_dtype,
            pretrained_pth=pretrained_pth,
            frozen_sam2_decoder=frozen_sam2_decoder,
            special_tokens=special_tokens,
            loss_sample_points=loss_sample_points,
            num_points=num_points,
            template=template,
            arch_type=arch_type,
            training_bs=training_bs,
        )
        
        self.dpo_beta = dpo_beta
        self.use_dpo_loss = use_dpo_loss
        print(f"Sa2VADPOModel: DPO beta={dpo_beta}, use_dpo={use_dpo_loss}")
    
    def compute_dice_score(self, pred_mask, target_mask):
        """计算Dice分数"""
        pred_flat = pred_mask.flatten()
        target_flat = target_mask.flatten()
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)
        return dice
    
    def compute_dpo_loss(self, pred_masks, chosen_masks, rejected_masks):
        """
        计算DPO损失
        
        Args:
            pred_masks: 预测的mask logits [B, H, W]
            chosen_masks: chosen masks [B, H, W]
            rejected_masks: rejected masks [B, H, W]
        
        Returns:
            dpo_loss: DPO损失
            metrics: 额外指标字典
        """
        batch_size = pred_masks.shape[0]
        total_loss = 0
        total_prefer = 0
        total_dice = 0
        
        pred_probs = torch.sigmoid(pred_masks)
        
        for i in range(batch_size):
            pred_prob = pred_probs[i]
            chosen = chosen_masks[i].to(pred_prob.device).to(pred_prob.dtype)
            rejected = rejected_masks[i].to(pred_prob.device).to(pred_prob.dtype)
            
            # 计算与chosen/rejected的Dice相似度
            dice_chosen = self.compute_dice_score(pred_prob, chosen)
            dice_rejected = self.compute_dice_score(pred_prob, rejected)
            
            # DPO: log概率差
            log_prob_chosen = torch.log(dice_chosen + 1e-8)
            log_prob_rejected = torch.log(dice_rejected + 1e-8)
            
            # DPO损失
            logits = self.dpo_beta * (log_prob_chosen - log_prob_rejected)
            loss = -F.logsigmoid(logits)
            
            total_loss += loss
            total_prefer += (dice_chosen > dice_rejected).float()
            total_dice += dice_chosen
        
        avg_loss = total_loss / batch_size
        avg_prefer = total_prefer / batch_size
        avg_dice = total_dice / batch_size
        
        return avg_loss, {
            'dpo_prefer': avg_prefer.item(),
            'dpo_dice': avg_dice.item(),
        }
    
    def forward(self, data, data_samples=None, mode='loss'):
        """
        Forward方法，支持DPO训练
        """
        # 检查是否有DPO数据
        is_dpo = data.pop('is_dpo', None)
        chosen_masks = data.pop('chosen_mask', None)
        rejected_masks = data.pop('rejected_mask', None)
        
        # 调用父类forward获取标准输出
        loss_dict = super().forward(data, data_samples, mode)
        
        # 如果有DPO数据，添加DPO损失
        if self.use_dpo_loss and is_dpo is not None and chosen_masks is not None:
            # 需要获取预测的masks
            # 这里我们使用父类forward中计算的pred_masks
            # 但由于父类forward不返回pred_masks，我们需要重新计算
            
            # 获取输入数据
            g_pixel_values = data.get('g_pixel_values', None)
            input_ids = data.get('input_ids', None)
            
            if g_pixel_values is not None and input_ids is not None:
                # 获取hidden states
                output = self.mllm(data, data_samples, mode)
                hidden_states = output.hidden_states[-1]
                hidden_states = self.text_hidden_fcs(hidden_states)
                
                # 获取[SEG] token的embedding
                seg_token_mask = input_ids == self.seg_token_idx
                if seg_token_mask.sum() > 0:
                    pred_embeddings = hidden_states[seg_token_mask]
                    
                    # 预测masks
                    g_pixel_values_stacked = torch.stack([
                        self.grounding_encoder.preprocess_image(pixel) 
                        for pixel in g_pixel_values
                    ])
                    
                    language_embeddings = pred_embeddings[:, None]
                    sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values_stacked)
                    pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings)
                    
                    # Resize chosen/rejected masks
                    pred_size = pred_masks.shape[-2:]
                    chosen_resized = F.interpolate(
                        chosen_masks.unsqueeze(1).float(), 
                        size=pred_size, mode='nearest'
                    ).squeeze(1)
                    rejected_resized = F.interpolate(
                        rejected_masks.unsqueeze(1).float(),
                        size=pred_size, mode='nearest'
                    ).squeeze(1)
                    
                    # 计算DPO损失
                    dpo_loss, dpo_metrics = self.compute_dpo_loss(
                        pred_masks.squeeze(1), chosen_resized, rejected_resized
                    )
                    
                    loss_dict['loss_dpo'] = dpo_loss
                    loss_dict['dpo_prefer'] = torch.tensor(dpo_metrics['dpo_prefer'])
                    loss_dict['dpo_dice'] = torch.tensor(dpo_metrics['dpo_dice'])
        
        return loss_dict
