DATA_ROOT = '/home/ubuntu/Sa2VA/data/'
# DPO微调数据集
DPO_VESSEL_ROOT = '/home/ubuntu/Sa2VA/data/dpo_vessel/'
accumulative_counts = 8
batch_size = 1
betas = (
    0.9,
    0.999,
)
custom_hooks = []
dataloader_num_workers = 4
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        max_keep_ckpts=5,
        save_optimizer=False,
        type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        interval=10,
        log_metric_by_epoch=False,
        type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
extra_image_processor = dict(
    target_length=1024, type='projects.sa2va.models.DirectResize')
launcher = 'none'
load_from = None  # 使用pretrained_pth加载
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr = 5e-06  # DPO使用更小的学习率
max_epochs = 1  # DPO只需要1个epoch
max_length = 4096
max_norm = 1
model = dict(
    frozen_sam2_decoder=False,
    grounding_encoder=dict(type='projects.sa2va.models.SAM2TrainRunner'),
    loss_dice=dict(
        activate=True,
        eps=1.0,
        loss_weight=1.0,
        naive_dice=True,
        reduction='mean',
        type='third_parts.mmdet.models.losses.DiceLoss',
        use_sigmoid=True),
    loss_mask=dict(
        loss_weight=3.0,
        reduction='mean',
        type='third_parts.mmdet.models.losses.CrossEntropyLoss',
        use_sigmoid=True),
    loss_sample_points=True,
    mllm=dict(
        freeze_llm=False,
        freeze_visual_encoder=True,
        llm_lora=dict(
            bias='none',
            lora_alpha=128,
            lora_dropout=0.1,
            modules_to_save=[
                'embed_tokens',
                'lm_head',
            ],
            r=64,
            task_type='CAUSAL_LM',
            type='peft.LoraConfig'),
        model_path=
        '/home/ubuntu/huggingface_cache/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9',
        type='projects.sa2va.models.InternVLMLLM'),
    pretrained_pth='/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth',
    special_tokens=[
        '[SEG]',
        '<p>',
        '</p>',
        '<vp>',
        '</vp>',
    ],
    tokenizer=dict(
        padding_side='right',
        pretrained_model_name_or_path='/home/ubuntu/Sa2VA/tokenizer_with_special_tokens',  # 预构建的tokenizer，已包含special tokens
        trust_remote_code=True,
        type='transformers.AutoTokenizer.from_pretrained'),
    training_bs=1,
    type='projects.sa2va.models.Sa2VAModel')
optim_type = 'torch.optim.AdamW'
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=2e-05,
        type='torch.optim.AdamW',
        weight_decay=0.05),
    type='DeepSpeedOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=0.30000000000000004,
        start_factor=1e-05,
        type='mmengine.optim.LinearLR'),
    dict(
        begin=0.30000000000000004,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        eta_min=0.0,
        type='mmengine.optim.CosineAnnealingLR'),
]
path = '/home/ubuntu/huggingface_cache/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9'
pretrained_pth = '/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth'
prompt_template = 'xtuner.utils.PROMPT_TEMPLATE.internlm2_chat'
randomness = dict(deterministic=False, seed=42)
resume = False
runner_type = 'FlexibleRunner'
sa2va_default_dataset_configs = dict(
    extra_image_processor=dict(
        target_length=1024, type='projects.sa2va.models.DirectResize'),
    max_length=4096,
    prompt_template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
    special_tokens=[
        '[SEG]',
        '<p>',
        '</p>',
        '<vp>',
        '</vp>',
    ],
    tokenizer=dict(
        padding_side='right',
        pretrained_model_name_or_path=
        '/home/ubuntu/Sa2VA/tokenizer_with_special_tokens',
        trust_remote_code=True,
        type='transformers.AutoTokenizer.from_pretrained'))
sa2va_merged_vessel_finetune_configs = [
    dict(
        ann_file='dpo_chosen_annotations.json',
        arch_type='intern_vl',
        data_prefix=dict(img_path=''),
        data_root=DPO_VESSEL_ROOT,
        extra_image_processor=dict(
            target_length=1024, type='projects.sa2va.models.DirectResize'),
        max_length=4096,
        name='DPOVesselFinetuneDataset',
        prompt_template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
        repeats=1,
        serialize_data=False,
        special_tokens=[
            '[SEG]',
            '<p>',
            '</p>',
            '<vp>',
            '</vp>',
        ],
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path=
            '/home/ubuntu/Sa2VA/tokenizer_with_special_tokens',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='projects.sa2va.datasets.Sa2VAFinetuneDataset'),
]
save_steps = 500
save_total_limit = 5
special_tokens = [
    '[SEG]',
    '<p>',
    '</p>',
    '<vp>',
    '</vp>',
]
strategy = dict(
    config=dict(
        bf16=dict(enabled=True),
        fp16=dict(enabled=False),
        gradient_accumulation_steps=8,
        gradient_clipping=1.0,
        train_micro_batch_size_per_gpu=1,
        wall_clock_breakdown=False,
        zero_allow_untested_optimizer=True,
        zero_optimization=dict(
            contiguous_gradients=True,
            overlap_comm=True,
            reduce_bucket_size='auto',
            stage=3,
            stage3_gather_16bit_weights_on_model_save=True,
            stage3_max_live_parameters=1000000000.0,
            stage3_max_reuse_distance=1000000000.0,
            stage3_param_persistence_threshold='auto',
            stage3_prefetch_bucket_size='auto',
            sub_group_size=1000000000.0)),
    exclude_frozen_parameters=True,
    gradient_accumulation_steps=8,
    gradient_clipping=1,
    sequence_parallel_size=1,
    train_micro_batch_size_per_gpu=1,
    type='xtuner.engine.DeepSpeedStrategy')
template = 'internlm2_chat'
tokenizer = dict(
    padding_side='right',
    pretrained_model_name_or_path=
    '/home/ubuntu/Sa2VA/tokenizer_with_special_tokens',
    trust_remote_code=True,
    type='transformers.AutoTokenizer.from_pretrained')
train_cfg = dict(max_epochs=1, type='xtuner.engine.runner.TrainLoop')
train_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='projects.sa2va.datasets.sa2va_collect_fn'),
    dataset=dict(
        datasets=[
            dict(
                ann_file='dpo_chosen_annotations.json',
                arch_type='intern_vl',
                data_prefix=dict(img_path=''),
                data_root=DPO_VESSEL_ROOT,
                extra_image_processor=dict(
                    target_length=1024,
                    type='projects.sa2va.models.DirectResize'),
                max_length=4096,
                name='DPOVesselFinetuneDataset',
                prompt_template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
                repeats=1,
                serialize_data=False,
                special_tokens=[
                    '[SEG]',
                    '<p>',
                    '</p>',
                    '<vp>',
                    '</vp>',
                ],
                tokenizer=dict(
                    padding_side='right',
                    pretrained_model_name_or_path=
                    '/home/ubuntu/Sa2VA/tokenizer_with_special_tokens',
                    trust_remote_code=True,
                    type='transformers.AutoTokenizer.from_pretrained'),
                type='projects.sa2va.datasets.Sa2VAFinetuneDataset'),
        ],
        type='projects.sa2va.datasets.data_utils.ConcatDatasetSa2VA'),
    num_workers=4,
    sampler=dict(
        length_property='modality_length',
        per_device_batch_size=8,
        type='xtuner.dataset.samplers.LengthGroupedSampler'))
train_dataset = dict(
    datasets=[
        dict(
            ann_file='annotations.json',
            arch_type='intern_vl',
            data_prefix=dict(img_path='images/'),
            data_root='/home/ubuntu/Sa2VA/data/merged_vessel_data/',
            extra_image_processor=dict(
                target_length=1024, type='projects.sa2va.models.DirectResize'),
            max_length=4096,
            name='DPOVesselFinetuneDataset',
            prompt_template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
            repeats=1,
            serialize_data=False,
            special_tokens=[
                '[SEG]',
                '<p>',
                '</p>',
                '<vp>',
                '</vp>',
            ],
            tokenizer=dict(
                padding_side='right',
                pretrained_model_name_or_path=
                '/home/ubuntu/Sa2VA/tokenizer_with_special_tokens',
                trust_remote_code=True,
                type='transformers.AutoTokenizer.from_pretrained'),
            type='projects.sa2va.datasets.Sa2VAFinetuneDataset'),
    ],
    type='projects.sa2va.datasets.data_utils.ConcatDatasetSa2VA')
visualizer = None
warmup_ratio = 0.1
weight_decay = 0.05
work_dir = 'work_dirs/dpo_vessel_training'
