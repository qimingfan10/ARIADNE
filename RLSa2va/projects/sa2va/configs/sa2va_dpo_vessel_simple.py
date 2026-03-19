"""
Sa2VA DPO 训练配置 - 简化版

直接复用Sa2VAModel，使用DPO数据集格式
"""

from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.sa2va.models import Sa2VAModel, SAM2TrainRunner, DirectResize, InternVLMLLM
from projects.sa2va.datasets import sa2va_collect_fn, Sa2VAFinetuneDataset
from projects.sa2va.datasets.data_utils import ConcatDatasetSa2VA

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
path = "/home/ubuntu/huggingface_cache/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9"

# 从已训练的血管分割模型继续
pretrained_pth = "/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth"

# Data
template = "internlm2_chat"
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 4096

# DPO使用更小的学习率和更少的epoch
batch_size = 1
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 2
optim_type = AdamW
lr = 5e-6  # DPO使用更小的学习率
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.1

save_steps = 300
save_total_limit = 3

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
#            PART 2  Model (标准Sa2VAModel)                           #
#######################################################################
model = dict(
    type=Sa2VAModel,
    training_bs=batch_size,
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVLMLLM,
        model_path=path,
        freeze_llm=False,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=16,  # 降低LoRA rank以减少显存
            lora_alpha=32,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM',
            # 不保存embed_tokens和lm_head，减少显存
        ),
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=3.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=1.0)
)

#######################################################################
#                      PART 3  DPO Dataset (使用标准格式)              #
#######################################################################

# 使用DPO生成的chosen masks作为GT进行微调
# 这是一种简化的DPO：用高质量mask替代原始GT
DATA_ROOT = '/home/ubuntu/Sa2VA/data/'
DPO_ROOT = DATA_ROOT + 'dpo_vessel/'

sa2va_dpo_dataset_configs=dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    max_length=max_length,
)

# 使用DPO的chosen masks作为训练目标
sa2va_dpo_finetune_configs = [
    dict(
        type=Sa2VAFinetuneDataset,
        name='DPOVesselFinetuneDataset',
        data_root=DPO_ROOT,
        data_prefix=dict(img_path=''),  # images是软链接
        ann_file='dpo_chosen_annotations.json',  # 只使用chosen
        arch_type='intern_vl',
        serialize_data=False,
        repeats=1,
        **sa2va_dpo_dataset_configs,
    )
]

train_dataset = dict(
    type=ConcatDatasetSa2VA, datasets=[
        *sa2va_dpo_finetune_configs,
    ]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=sa2va_collect_fn)
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

# DeepSpeed - 使用ZeRO-3减少显存
strategy = dict(
    type='DeepSpeedStrategy',
    gradient_accumulation_steps=accumulative_counts,
    gradient_clipping=max_norm,
    zero_optimization=dict(
        stage=3,  # ZeRO-3: 分片模型参数
        offload_optimizer=dict(device='cpu', pin_memory=True),
        offload_param=dict(device='cpu', pin_memory=True),  # 分片参数到CPU
        overlap_comm=True,
        contiguous_gradients=True,
        sub_group_size=1e9,
        reduce_bucket_size='auto',
        stage3_prefetch_bucket_size='auto',
        stage3_param_persistence_threshold='auto',
        stage3_max_live_parameters=1e9,
        stage3_max_reuse_distance=1e9,
        stage3_gather_16bit_weights_on_model_save=True
    ),
    bf16=dict(enabled=True),
    zero_allow_untested_optimizer=True,
    wall_clock_breakdown=False
)
