"""
Sa2VA DPO (Direct Preference Optimization) 模型

DPO Loss公式：
L_DPO = -E[log σ(β * (log π(chosen) - log π_ref(chosen) - log π(rejected) + log π_ref(rejected)))]

简化版（LoRA模式，无需reference model）：
L_DPO = -E[log σ(β * (log π(chosen) - log π(rejected)))]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from mmengine.model import BaseModel
from xtuner.registry import BUILDER


class Sa2VADPOModel(BaseModel):
    """Sa2VA DPO训练模型"""
    
    def __init__(
        self,
        mllm: dict,
        grounding_encoder: dict,
        tokenizer: dict,
        special_tokens: List[str],
        pretrained_pth: Optional[str] = None,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        use_reference_model: bool = False,
        training_bs: int = 1,
        **kwargs
    ):
        super().__init__()
        
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.use_reference_model = use_reference_model
        
        # 构建基础Sa2VA模型
        from projects.sa2va.models import Sa2VAModel
        self.model = Sa2VAModel(
            mllm=mllm,
            grounding_encoder=grounding_encoder,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            pretrained_pth=pretrained_pth,
            training_bs=training_bs,
            **kwargs
        )
        
        # 如果需要reference model（非LoRA模式）
        if use_reference_model:
            import copy
            self.ref_model = copy.deepcopy(self.model)
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        else:
            self.ref_model = None
        
        print(f"\n{'='*60}")
        print("🎯 Sa2VA DPO Model 初始化")
        print(f"{'='*60}")
        print(f"  - beta: {beta}")
        print(f"  - label_smoothing: {label_smoothing}")
        print(f"  - use_reference_model: {use_reference_model}")
        print(f"{'='*60}\n")
    
    def compute_log_probs(
        self,
        model: nn.Module,
        images: List,
        masks: torch.Tensor,
        prompts: List[str]
    ) -> torch.Tensor:
        """
        计算给定mask的log概率
        
        对于分割任务，我们计算的是每个像素的log概率
        然后对整个mask进行平均
        """
        batch_size = len(images)
        log_probs = []
        
        for i in range(batch_size):
            image = images[i]
            mask = masks[i]  # [H, W] or [1, H, W]
            prompt = prompts[i]
            
            # 获取模型的分割输出（logits）
            with torch.set_grad_enabled(model.training):
                outputs = model.forward_segmentation(
                    image=image,
                    prompt=prompt,
                    return_logits=True
                )
                
                if outputs is None or 'logits' not in outputs:
                    # 如果没有logits，返回dummy值
                    log_probs.append(torch.tensor(0.0, device=mask.device))
                    continue
                
                logits = outputs['logits']  # [1, 1, H, W]
                
                # 确保mask和logits尺寸一致
                if logits.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=logits.shape[-2:],
                        mode='nearest'
                    ).squeeze()
                
                # 计算每个像素的log概率
                # 使用Binary Cross Entropy的负值作为log probability
                # log p(y|x) = y * log(σ(logits)) + (1-y) * log(1-σ(logits))
                probs = torch.sigmoid(logits)
                mask_flat = mask.flatten().float()
                probs_flat = probs.flatten()
                
                # 避免log(0)
                eps = 1e-7
                probs_flat = probs_flat.clamp(eps, 1 - eps)
                
                # 计算log概率
                log_p = mask_flat * torch.log(probs_flat) + (1 - mask_flat) * torch.log(1 - probs_flat)
                
                # 对整个mask平均
                log_prob = log_p.mean()
                log_probs.append(log_prob)
        
        return torch.stack(log_probs)
    
    def compute_segmentation_log_probs(
        self,
        images: List,
        masks: torch.Tensor,
        prompts: List[str],
        use_reference: bool = False
    ) -> torch.Tensor:
        """计算分割mask的log概率"""
        model = self.ref_model if (use_reference and self.ref_model is not None) else self.model
        
        if use_reference and self.ref_model is not None:
            with torch.no_grad():
                return self.compute_log_probs(model, images, masks, prompts)
        else:
            return self.compute_log_probs(model, images, masks, prompts)
    
    def dpo_loss(
        self,
        chosen_log_probs: torch.Tensor,
        rejected_log_probs: torch.Tensor,
        ref_chosen_log_probs: Optional[torch.Tensor] = None,
        ref_rejected_log_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算DPO Loss
        
        Args:
            chosen_log_probs: log π(chosen)
            rejected_log_probs: log π(rejected)
            ref_chosen_log_probs: log π_ref(chosen) (可选)
            ref_rejected_log_probs: log π_ref(rejected) (可选)
        
        Returns:
            loss: DPO loss
            metrics: 训练指标
        """
        if ref_chosen_log_probs is not None and ref_rejected_log_probs is not None:
            # 完整DPO公式
            chosen_rewards = self.beta * (chosen_log_probs - ref_chosen_log_probs)
            rejected_rewards = self.beta * (rejected_log_probs - ref_rejected_log_probs)
        else:
            # 简化版（无reference model）
            chosen_rewards = self.beta * chosen_log_probs
            rejected_rewards = self.beta * rejected_log_probs
        
        # DPO loss: -log σ(chosen_reward - rejected_reward)
        logits = chosen_rewards - rejected_rewards
        
        if self.label_smoothing > 0:
            # Label smoothing
            loss = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            )
        else:
            loss = -F.logsigmoid(logits)
        
        loss = loss.mean()
        
        # 计算指标
        with torch.no_grad():
            chosen_probs = torch.sigmoid(chosen_rewards)
            rejected_probs = torch.sigmoid(rejected_rewards)
            accuracy = (logits > 0).float().mean()
            margin = (chosen_rewards - rejected_rewards).mean()
        
        metrics = {
            'dpo_loss': loss.item(),
            'chosen_rewards': chosen_rewards.mean().item(),
            'rejected_rewards': rejected_rewards.mean().item(),
            'accuracy': accuracy.item(),
            'margin': margin.item(),
        }
        
        return loss, metrics
    
    def forward(self, data: Dict, mode: str = 'loss') -> Dict:
        """
        前向传播
        
        Args:
            data: 包含以下键的字典
                - images: 图像列表
                - chosen_masks: 胜者masks [B, H, W]
                - rejected_masks: 败者masks [B, H, W]
                - prompts: prompt列表
            mode: 'loss' 或 'predict'
        """
        if mode == 'predict':
            return self.predict(data)
        
        images = data['images']
        chosen_masks = data['chosen_masks']
        rejected_masks = data['rejected_masks']
        prompts = data['prompts']
        
        # 计算chosen的log概率
        chosen_log_probs = self.compute_segmentation_log_probs(
            images, chosen_masks, prompts, use_reference=False
        )
        
        # 计算rejected的log概率
        rejected_log_probs = self.compute_segmentation_log_probs(
            images, rejected_masks, prompts, use_reference=False
        )
        
        # 如果使用reference model
        ref_chosen_log_probs = None
        ref_rejected_log_probs = None
        
        if self.use_reference_model and self.ref_model is not None:
            ref_chosen_log_probs = self.compute_segmentation_log_probs(
                images, chosen_masks, prompts, use_reference=True
            )
            ref_rejected_log_probs = self.compute_segmentation_log_probs(
                images, rejected_masks, prompts, use_reference=True
            )
        
        # 计算DPO loss
        loss, metrics = self.dpo_loss(
            chosen_log_probs,
            rejected_log_probs,
            ref_chosen_log_probs,
            ref_rejected_log_probs
        )
        
        # 返回loss字典
        return {
            'loss': loss,
            **{f'train/{k}': v for k, v in metrics.items()}
        }
    
    def predict(self, data: Dict) -> Dict:
        """推理模式"""
        return self.model.predict(data)


# 简化版DPO Loss函数（可独立使用）
def dpo_loss_simple(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    简化版DPO Loss计算
    
    Returns:
        losses: per-sample losses
        chosen_rewards: implicit rewards for chosen
        rejected_rewards: implicit rewards for rejected
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = pi_logratios - ref_logratios
    
    losses = -F.logsigmoid(beta * logits)
    
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    
    return losses, chosen_rewards, rejected_rewards
