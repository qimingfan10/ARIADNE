#!/bin/bash
# 实验三：Reward Network RL - 后台运行脚本
# 使用GPU1（最空闲），避开GPU3

# 配置
SCRIPT_DIR="/home/ubuntu/Sa2VA/rl_reward_network"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/experiment3_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/experiment3.pid"

# 创建日志目录
mkdir -p "${LOG_DIR}"
mkdir -p "${SCRIPT_DIR}/outputs"

# 日志函数
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [EXP3] [INFO] $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [EXP3] [ERROR] $1" | tee -a "${LOG_FILE}"
}

log_debug() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [EXP3] [DEBUG] $1" | tee -a "${LOG_FILE}"
}

# 开始训练
log_info "=========================================="
log_info "实验三：Reward Network RL 启动"
log_info "=========================================="
log_info "日志文件: ${LOG_FILE}"
log_info "PID文件: ${PID_FILE}"
log_info "指定GPU: GPU1 (最空闲)"

# 检查环境
log_debug "检查Python环境..."
python3 --version >> "${LOG_FILE}" 2>&1
if [ $? -ne 0 ]; then
    log_error "Python环境检查失败"
    exit 1
fi
log_info "✅ Python环境正常"

# 检查GPU
log_debug "检查GPU状态..."
nvidia-smi >> "${LOG_FILE}" 2>&1
if [ $? -eq 0 ]; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    log_info "✅ 检测到 ${GPU_COUNT} 个GPU"
    log_info "GPU使用情况:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> "${LOG_FILE}" 2>&1
else
    log_error "⚠️  未检测到GPU"
fi

# 检查依赖
log_debug "检查Python依赖..."
python3 -c "import torch; import torchvision" >> "${LOG_FILE}" 2>&1
if [ $? -ne 0 ]; then
    log_error "PyTorch依赖检查失败"
    exit 1
fi
log_info "✅ Python依赖正常"

# 检查模型文件
MODEL_PATH="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
if [ ! -d "${MODEL_PATH}" ]; then
    log_error "模型路径不存在: ${MODEL_PATH}"
    exit 1
fi
log_info "✅ 模型文件存在"

# 检查数据集
DATA_PATH="/home/ubuntu/Sa2VA/data/merged_vessel_data"
if [ ! -f "${DATA_PATH}/annotations.json" ]; then
    log_error "数据集不存在: ${DATA_PATH}"
    exit 1
fi
DATASET_SIZE=$(python3 -c "import json; data=json.load(open('${DATA_PATH}/annotations.json')); print(len(data))")
log_info "✅ 数据集正常，共 ${DATASET_SIZE} 个样本"

# 训练模式选择
if [ "$1" == "quick" ]; then
    MODE="快速测试"
    MAX_SAMPLES=50
    EPOCHS=10
else
    MODE="完整训练"
    MAX_SAMPLES=200
    EPOCHS=20
fi

log_info "训练模式: ${MODE}"
log_info "配置参数:"
log_info "  - MAX_SAMPLES: ${MAX_SAMPLES}"
log_info "  - EPOCHS: ${EPOCHS}"
log_info "  - GPU: GPU1 (避开GPU3)"
log_info "  - 训练步骤: 1.训练Reward Network"

# 构建训练命令
# 步骤1：训练Reward Network
# 使用GPU1，减小batch_size避免OOM
TRAIN_CMD="CUDA_VISIBLE_DEVICES=1 python3 ${SCRIPT_DIR}/train_reward_network.py \
    --model_path ${MODEL_PATH} \
    --data_root ${DATA_PATH} \
    --output_dir ${SCRIPT_DIR}/outputs \
    --max_samples ${MAX_SAMPLES} \
    --epochs ${EPOCHS} \
    --batch_size 4 \
    --lr 1e-3 \
    --gpu 0"

log_info "训练命令:"
log_info "${TRAIN_CMD}"

# 后台运行
log_info "=========================================="
log_info "开始训练（实验三-步骤1）..."
log_info "=========================================="

nohup bash -c "${TRAIN_CMD}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo ${TRAIN_PID} > "${PID_FILE}"

log_info "实验三进程已启动，PID: ${TRAIN_PID}"
log_info ""
log_info "监控训练进度:"
log_info "  查看日志: tail -f ${LOG_FILE}"
log_info "  查看GPU: watch -n 1 nvidia-smi"
log_info "  停止训练: kill ${TRAIN_PID}"
log_info ""
log_info "TensorBoard (端口6008):"
log_info "  tensorboard --logdir ${SCRIPT_DIR}/outputs/*/logs --port 6008"
log_info ""

# 检查进程启动
sleep 3
if ps -p ${TRAIN_PID} > /dev/null 2>&1; then
    log_info "✅ 实验三训练进程运行正常"
    log_info "使用GPU1训练Reward Network"
else
    log_error "❌ 实验三训练进程启动失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "实验三（步骤1/2）后台训练已启动！"
echo "=========================================="
echo "日志文件: ${LOG_FILE}"
echo "进程PID: ${TRAIN_PID}"
echo "使用GPU: GPU1"
echo ""
echo "实时查看训练日志:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "TensorBoard (端口6008):"
echo "  tensorboard --logdir ${SCRIPT_DIR}/outputs/*/logs --port 6008"
echo ""
echo "停止实验三:"
echo "  kill ${TRAIN_PID}"
echo ""
echo "说明: 这是实验三的第1步（训练Reward Network）"
echo "      完成后将进行第2步（使用Reward Network微调Sa2VA）"
echo ""
