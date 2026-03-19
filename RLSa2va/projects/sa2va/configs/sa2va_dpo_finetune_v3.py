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
from projects.sa2va.datasets import (
    sa2va_collect_fn, Sa2VAFinetuneDataset
)

from projects.sa2va.datasets.data_utils import ConcatDatasetSa2VA

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model - 使用InternVL3-8B作为基础模型
path = "/home/ubuntu/huggingface_cache/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9"
pretrained_pth = None  # 不使用pretrained_pth，从头开始

# Data
template = "internlm2_chat"
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 4096

# Scheduler & Optimizer - 针对更大数据集优化
batch_size = 1  # 每个GPU的批次大小
accumulative_counts = 8  # 梯度累积步数
dataloader_num_workers = 4
max_epochs = 1  # DPO只需要1个epoch
optim_type = AdamW
lr = 5e-6  # DPO使用更小的学习率
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.1

# Save
save_steps = 500
save_total_limit = 5

# 血管分割专用的特殊token
special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path='/home/ubuntu/Sa2VA/tokenizer_with_special_tokens',  # 预构建tokenizer
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
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
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM',
            # 移除modules_to_save以避免OOM
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
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

DATA_ROOT = '/home/ubuntu/Sa2VA/data/'

# 血管分割数据集配置
sa2va_default_dataset_configs=dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    max_length=max_length,
)

######################### Merged血管分割数据集 ##################################

DPO_VESSEL_ROOT = DATA_ROOT + 'dpo_vessel/'

sa2va_merged_vessel_finetune_configs = [
    dict(
        type=Sa2VAFinetuneDataset,
        name='DPOVesselFinetuneDataset',
        data_root=DPO_VESSEL_ROOT,
        data_prefix=dict(img_path='images/'),
        ann_file='dpo_chosen_annotations.json',
        arch_type='intern_vl',
        serialize_data=False,
        repeats=1,  # DPO数据
        **sa2va_default_dataset_configs,
    )
]

train_dataset = dict(
    type=ConcatDatasetSa2VA, datasets=[
        *sa2va_merged_vessel_finetune_configs,
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
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
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

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint - 从头开始训练新数据集
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=42, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

# DeepSpeed strategy for multi-GPU training with model sharding
strategy = dict(
    type='DeepSpeedStrategy',
    gradient_accumulation_steps=accumulative_counts,
    gradient_clipping=max_norm,
    zero_optimization=dict(
        stage=3,  # ZeRO-3: 分片模型参数、梯度和优化器状态
        offload_optimizer=dict(
            device='cpu',
            pin_memory=True
        ),
        offload_param=dict(
            device='cpu',
            pin_memory=True
        ),
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
