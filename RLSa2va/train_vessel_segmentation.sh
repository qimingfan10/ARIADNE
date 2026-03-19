#!/bin/bash

# Sa2VA血管分割微调训练脚本
# 使用单GPU训练以避免复杂的分布式设置

# 激活micromamba环境
echo "激活micromamba环境: topo-sarl"
export MAMBA_ROOT_PREFIX=/home/ubuntu/micromamba
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate topo-sarl

# 检查环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "topo-sarl" ]; then
    echo "错误: 无法激活topo-sarl环境"
    exit 1
fi

echo "环境激活成功: $CONDA_DEFAULT_ENV"
echo "开始Sa2VA血管分割微调训练..."
echo "配置文件: projects/sa2va/configs/sa2va_vessel_finetune.py"
echo "数据路径: /home/ubuntu/Sa2VA/data/vessel_data/"
echo "模型路径: /home/ubuntu/Sa2VA-26B.pth"

# # 设置CUDA路径
# export CUDA_HOME=/usr/local/cuda-11.7
# export PATH=$CUDA_HOME/bin:$PATH

# # 验证CUDA安装
# if [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
#     echo "错误: 找不到nvcc，请检查CUDA安装路径"
#     exit 1
# fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/ubuntu/Sa2VA:$PYTHONPATH

# 设置Hugging Face缓存目录（使用有足够空间的目录）
export HF_HOME=/home/ubuntu/huggingface_cache
export TRANSFORMERS_CACHE=/home/ubuntu/huggingface_cache
export HF_DATASETS_CACHE=/home/ubuntu/huggingface_cache

# 禁用在线模式，强制使用本地缓存
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 设置PyTorch CUDA内存管理，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 创建输出目录
mkdir -p /home/ubuntu/Sa2VA/work_dirs/vessel_segmentation

# 检查数据是否存在
if [ ! -f "/home/ubuntu/Sa2VA/data/vessel_data/annotations.json" ]; then
    echo "错误: annotations.json文件不存在"
    exit 1
fi

if [ ! -d "/home/ubuntu/Sa2VA/data/vessel_data/images" ]; then
    echo "错误: images目录不存在"
    exit 1
fi

if [ ! -f "/home/ubuntu/Sa2VA-26B.pth" ]; then
    echo "错误: Sa2VA-26B.pth模型文件不存在"
    exit 1
fi

echo "数据和模型检查完成，开始训练..."

# 启动训练 - 使用单GPU
cd /home/ubuntu/Sa2VA

# # 设置CUDA路径
# export CUDA_HOME=/usr/local/cuda-11.7
# export PATH=$CUDA_HOME/bin:$PATH

# # 验证CUDA安装
# if [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
#     echo "错误: 找不到nvcc，请检查CUDA安装路径"
#     exit 1
# fi

export CUDA_VISIBLE_DEVICES=0,1

/home/ubuntu/micromamba/envs/topo-sarl/bin/python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29500 \
    /home/ubuntu/micromamba/envs/topo-sarl/lib/python3.10/site-packages/xtuner/tools/train.py \
    projects/sa2va/configs/sa2va_vessel_finetune.py \
    --work-dir work_dirs/vessel_segmentation \
    2>&1 | tee work_dirs/vessel_segmentation/training.log

echo "训练完成！"
echo "检查点保存在: work_dirs/vessel_segmentation/"
echo "训练日志保存在: work_dirs/vessel_segmentation/training.log"
