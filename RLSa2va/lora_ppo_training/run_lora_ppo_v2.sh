#!/bin/bash

# LoRA + PPO 优化版训练（Version 2）
# 改进：更多数据 + 更高学习率 + 更大rank + 强化Recall

echo "========================================"
echo "LoRA + PPO 优化版训练 V2"
echo "========================================"
echo "改进点:"
echo "  1. LoRA Rank: 32 → 64"
echo "  2. 学习率: 5e-5 → 1e-4"
echo "  3. 训练样本: 1000 → 1220（全部数据）"
echo "  4. 训练轮数: 3 → 10 epochs"
echo "  5. Recall权重: 0.2 → 0.4（强化）"
echo "========================================"

# 配置
MODEL_PATH="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
DATA_ROOT="/home/ubuntu/Sa2VA/data/merged_vessel_data"
OUTPUT_DIR="/home/ubuntu/Sa2VA/lora_ppo_training/output_v2"
GPU=1

# 优化后的LoRA参数
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

# 优化后的训练参数
MAX_TRAIN_SAMPLES=1220  # 使用全部数据
MAX_VAL_SAMPLES=100
NUM_EPOCHS=10          # 增加训练轮数
LEARNING_RATE=1e-4     # 提高学习率

# 优化后的奖励权重（强化Recall）
REWARD_TYPE="multi_objective"
DICE_WEIGHT=0.4        # 0.5 → 0.4
RECALL_WEIGHT=0.4      # 0.2 → 0.4 (加倍)
TOPOLOGY_WEIGHT=0.15   # 0.2 → 0.15
LENGTH_WEIGHT=0.05     # 0.1 → 0.05
RECALL_TARGET=0.85

# Prompt
PROMPT="Please segment the blood vessel."

echo ""
echo "详细配置:"
echo "  模型: $MODEL_PATH"
echo "  数据集: $DATA_ROOT"
echo "  GPU: GPU$GPU"
echo ""
echo "  LoRA Rank: $LORA_RANK ⬆"
echo "  LoRA Alpha: $LORA_ALPHA ⬆"
echo "  学习率: $LEARNING_RATE ⬆"
echo ""
echo "  训练样本: $MAX_TRAIN_SAMPLES ⬆"
echo "  验证样本: $MAX_VAL_SAMPLES"
echo "  训练轮数: $NUM_EPOCHS ⬆"
echo ""
echo "  奖励类型: $REWARD_TYPE"
echo "  Dice权重: $DICE_WEIGHT"
echo "  Recall权重: $RECALL_WEIGHT ⬆⬆"
echo "  拓扑权重: $TOPOLOGY_WEIGHT"
echo "  长度权重: $LENGTH_WEIGHT"
echo "  Recall目标: $RECALL_TARGET"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/train_v2_${TIMESTAMP}.log"

echo "开始训练..."
echo "日志文件: $LOG_FILE"
echo "预计时间: 10-15小时"
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
    --val_freq 100 \
    --save_freq 2 \
    --log_freq 10 \
    2>&1 | tee $LOG_FILE

echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"
echo "日志文件: $LOG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "查看结果:"
echo "  cat $OUTPUT_DIR/sa2va_lora_ppo_*/training_info.json"
echo ""
