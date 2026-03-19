"""
Sa2VA DPO (Direct Preference Optimization) 训练配置

DPO优势：
1. 不需要Critic网络 → 显存减半
2. 直接从偏好对学习 → 训练更稳定
3. 复用MMEngine框架 → BFloat16兼容
"""

from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from peft import LoraConfig

from projects.sa2va.models import Sa2VAModel, SAM2TrainRunner, DirectResize, InternVLMLLM
from projects.sa2va.models.sa2va_dpo_model import Sa2VADPOModel
from projects.sa2va.datasets.dpo_vessel_dataset import DPOVesselDataset, dpo_collect_fn

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model - 使用已微调的Sa2VA模型
path = "/home/ubuntu/huggingface_cache/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9"

# 使用已训练的血管分割模型作为起点
pretrained_pth = "/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth"

# DPO 超参数
beta = 0.1  # DPO温度参数，控制偏好强度
label_smoothing = 0.0  # 标签平滑

# Data
template = "internlm2_chat"
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 4096

# Scheduler & Optimizer
batch_size = 1
accumulative_counts = 4  # DPO需要更小的batch（每个样本包含chosen+rejected）
dataloader_num_workers = 4
max_epochs = 2  # DPO通常需要更少的epoch
optim_type = AdamW
lr = 5e-6  # DPO使用更小的学习率
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.1

# Save
save_steps = 200
save_total_limit = 3

# 特殊tokens
special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)

#######################################################################
#            PART 2  Model (DPO版本)                                  #
#######################################################################
model = dict(
    type=Sa2VADPOModel,  # DPO专用模型wrapper（直接使用类）
    
    # 基础Sa2VA模型配置
    training_bs=batch_size,
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    
    # DPO参数
    beta=beta,
    label_smoothing=label_smoothing,
    
    # 是否使用reference model（可选）
    # 如果使用LoRA，可以不加载reference model
    use_reference_model=False,  # LoRA模式下不需要
    
    mllm=dict(
        type=InternVLMLLM,
        model_path=path,
        freeze_llm=False,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM',
            modules_to_save=["embed_tokens", "lm_head"]
        ),
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
    ),
)

#######################################################################
#                      PART 3  DPO Dataset                            #
#######################################################################

DATA_ROOT = '/home/ubuntu/Sa2VA/data/dpo_vessel'

train_dataset = dict(
    type=DPOVesselDataset,
    data_root=DATA_ROOT,
    ann_file='dpo_annotations.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    extra_image_processor=extra_image_processor,
    max_length=max_length,
    min_iou_gap=0.03,  # 最小IoU差距（已在数据集生成时过滤）
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=dpo_collect_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
custom_hooks = []

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)
log_processor = dict(by_epoch=False)

# DeepSpeed配置 - DPO只需要单模型，显存需求更低
strategy = dict(
    type='DeepSpeedStrategy',
    gradient_accumulation_steps=accumulative_counts,
    gradient_clipping=max_norm,
    zero_optimization=dict(
        stage=2,  # DPO可以用ZeRO-2，因为模型更小
        offload_optimizer=dict(
            device='cpu',
            pin_memory=True
        ),
        overlap_comm=True,
        contiguous_gradients=True,
        reduce_bucket_size='auto',
    ),
    bf16=dict(enabled=True),
    zero_allow_untested_optimizer=True,
    wall_clock_breakdown=False
)
