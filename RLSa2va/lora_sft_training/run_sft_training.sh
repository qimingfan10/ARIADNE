#!/bin/bash
#
# LoRA SFT训练启动脚本
#

# 设置GPU
export CUDA_VISIBLE_DEVICES=3

# 训练配置
MODEL_PATH="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
DATA_ROOT="/home/ubuntu/Sa2VA/Segment_DATA_Merged_512"
OUTPUT_DIR="/home/ubuntu/Sa2VA/lora_sft_training/output_sft"

# LoRA配置（根据文档建议）
LORA_RANK=64        # 或128（更大的容量）
LORA_ALPHA=128      # rank的2倍
LORA_DROPOUT=0.05

# 训练超参数
NUM_EPOCHS=15
BATCH_SIZE=1        # 根据显存调整
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01

# Loss权重
WEIGHT_DICE=1.0
WEIGHT_FOCAL=1.0
WEIGHT_BCE=0.5

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 启动训练
echo "开始LoRA SFT训练..."
echo "模型: $MODEL_PATH"
echo "数据: $DATA_ROOT"
echo "输出: $OUTPUT_DIR"
echo "LoRA: rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo ""

python3 train_lora_sft.py \
    --model_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --weight_dice $WEIGHT_DICE \
    --weight_focal $WEIGHT_FOCAL \
    --weight_bce $WEIGHT_BCE \
    --gradient_checkpointing \
    --save_every 3 \
    --gpu 3 \
    2>&1 | tee $OUTPUT_DIR/training.log

echo ""
echo "训练完成！"
echo "日志保存在: $OUTPUT_DIR/training.log"
