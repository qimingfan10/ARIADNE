#!/bin/bash

# LoRA + PPO训练脚本
# 目标：通过强化学习优化Dice到0.87+, Recall到0.85+

echo "========================================"
echo "Sa2VA LoRA + PPO微调"
echo "========================================"

# 模式：quick（快速测试） 或 full（完整训练）
MODE=${1:-quick}

# 配置
MODEL_PATH="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
DATA_ROOT="/home/ubuntu/Sa2VA/data/merged_vessel_data"
OUTPUT_DIR="/home/ubuntu/Sa2VA/lora_ppo_training/output"
GPU=1

# LoRA参数
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.05

# 根据模式设置参数
if [ "$MODE" == "quick" ]; then
    echo "模式: 快速测试"
    MAX_TRAIN_SAMPLES=50
    MAX_VAL_SAMPLES=20
    NUM_EPOCHS=1
    LEARNING_RATE=5e-5
    VAL_FREQ=50
elif [ "$MODE" == "full" ]; then
    echo "模式: 完整训练"
    MAX_TRAIN_SAMPLES=1000
    MAX_VAL_SAMPLES=100
    NUM_EPOCHS=3
    LEARNING_RATE=5e-5
    VAL_FREQ=100
else
    echo "错误: 未知模式 '$MODE'"
    echo "用法: bash run_lora_ppo.sh [quick|full]"
    exit 1
fi

# 奖励函数参数
REWARD_TYPE="multi_objective"  # multi_objective | simple_dice | recall_focused
DICE_WEIGHT=0.5
RECALL_WEIGHT=0.2
TOPOLOGY_WEIGHT=0.2
LENGTH_WEIGHT=0.1
RECALL_TARGET=0.85

# Prompt
PROMPT="Please segment the blood vessel."

echo ""
echo "配置信息:"
echo "  模型: $MODEL_PATH"
echo "  数据集: $DATA_ROOT"
echo "  GPU: GPU$GPU"
echo "  LoRA Rank: $LORA_RANK"
echo "  训练样本: $MAX_TRAIN_SAMPLES"
echo "  验证样本: $MAX_VAL_SAMPLES"
echo "  训练轮数: $NUM_EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  奖励类型: $REWARD_TYPE"
echo "  Recall目标: $RECALL_TARGET"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/train_${MODE}_${TIMESTAMP}.log"

echo "开始训练..."
echo "日志文件: $LOG_FILE"
echo ""

# 运行训练
CUDA_VISIBLE_DEVICES=$GPU python3 /home/ubuntu/Sa2VA/lora_ppo_training/train_lora_ppo.py \
    --model_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_val_samples $MAX_VAL_SAMPLES \
    --reward_type $REWARD_TYPE \
    --dice_weight $DICE_WEIGHT \
    --recall_weight $RECALL_WEIGHT \
    --topology_weight $TOPOLOGY_WEIGHT \
    --length_weight $LENGTH_WEIGHT \
    --recall_target $RECALL_TARGET \
    --prompt "$PROMPT" \
    --gpu 0 \
    --num_workers 4 \
    --val_freq $VAL_FREQ \
    --save_freq 1 \
    --log_freq 10 \
    2>&1 | tee $LOG_FILE

echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"
echo "日志文件: $LOG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo ""
