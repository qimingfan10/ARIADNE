#!/bin/bash
#######################################################################
# Sa2VA DPO 训练脚本
#
# DPO (Direct Preference Optimization) 优势：
# 1. 不需要Critic网络 → 显存减半
# 2. 直接从偏好对学习 → 训练更稳定
# 3. 复用MMEngine框架 → BFloat16兼容
#
# 使用方法：
#   bash train_dpo_vessel.sh          # 前台运行
#   bash train_dpo_vessel.sh bg       # 后台运行
#######################################################################

set -e

# 设置PYTHONPATH
export PYTHONPATH="/home/ubuntu/Sa2VA:$PYTHONPATH"

# 配置 - 复用原始训练配置（经过验证可以成功运行）
CONFIG="projects/sa2va/configs/sa2va_dpo_finetune_v3.py"
WORK_DIR="work_dirs/dpo_vessel_training"
GPUS=4  # 使用4张GPU
LOG_FILE="$WORK_DIR/dpo_training_$(date +%Y%m%d_%H%M%S).log"

# 是否后台运行
BACKGROUND=${1:-""}  # 第一个参数，默认为空

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "========================================"
echo "  Sa2VA DPO 血管分割训练"
echo "========================================"
echo -e "${NC}"

# 检查数据集
DPO_DATA="/home/ubuntu/Sa2VA/data/dpo_vessel"
ANN_FILE="$DPO_DATA/dpo_chosen_annotations.json"

if [ ! -f "$ANN_FILE" ]; then
    echo -e "${YELLOW}⚠️ DPO数据集不存在，正在生成...${NC}"
    # Step 1: 生成偏好对
    python3 scripts/generate_dpo_dataset_v2.py \
        --data_root /home/ubuntu/Sa2VA/data/merged_vessel_data \
        --output_dir $DPO_DATA \
        --num_samples 7 \
        --min_iou_gap 0.03
    # Step 2: 转换为Sa2VA格式
    python3 scripts/convert_dpo_to_sa2va_format.py \
        --dpo_root $DPO_DATA \
        --dpo_ann $DPO_DATA/dpo_annotations.json \
        --output $ANN_FILE
fi

# 统计数据
NUM_PAIRS=$(python3 -c "import json; print(len(json.load(open('$ANN_FILE'))))")
echo -e "${GREEN}✅ DPO数据集就绪${NC}"
echo "   - 偏好对数量: $NUM_PAIRS"
echo "   - 数据路径: $DPO_DATA"
echo ""

# 创建工作目录
mkdir -p $WORK_DIR

# 构建训练命令 - 使用xtuner方式（与之前成功训练一致）
cd /home/ubuntu/Sa2VA

# 激活conda环境
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate topo-sarl

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 使用xtuner + deepspeed_zero3（与之前成功训练一致）
export NPROC_PER_NODE=$GPUS
TRAIN_CMD="python -m xtuner.tools.train $CONFIG --work-dir $WORK_DIR --deepspeed deepspeed_zero3"

# 开始训练
echo -e "${GREEN}🚀 开始DPO训练...${NC}"
echo "   - 配置文件: $CONFIG"
echo "   - 工作目录: $WORK_DIR"
echo "   - GPU数量: $GPUS"
echo "   - 日志文件: $LOG_FILE"
echo ""

if [ "$BACKGROUND" = "bg" ]; then
    # 后台运行
    echo -e "${YELLOW}📋 后台模式启动...${NC}"
    nohup $TRAIN_CMD > $LOG_FILE 2>&1 &
    PID=$!
    echo $PID > $WORK_DIR/train.pid
    
    echo ""
    echo -e "${GREEN}✅ 训练已在后台启动!${NC}"
    echo "   - PID: $PID"
    echo "   - 日志: $LOG_FILE"
    echo ""
    echo "📊 监控命令:"
    echo "   tail -f $LOG_FILE"
    echo ""
    echo "🛑 停止训练:"
    echo "   kill $PID"
else
    # 前台运行
    $TRAIN_CMD 2>&1 | tee $LOG_FILE
    
    echo ""
    echo -e "${GREEN}✅ DPO训练完成!${NC}"
    echo "   - 模型保存在: $WORK_DIR"
fi
