#!/bin/bash

################################################################################
# 实验三步骤2：使用Reward Network微调Sa2VA
# 这是实验三的第二步，使用训练好的Reward Network作为奖励函数来微调Sa2VA
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/step2_finetune_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/step2_finetune.pid"

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP2] [INFO] $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP2] [ERROR] $1" | tee -a "${LOG_FILE}"
}

log_debug() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP2] [DEBUG] $1" >> "${LOG_FILE}"
}

# 配置参数
MODEL_PATH="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
DATA_PATH="/home/ubuntu/Sa2VA/data/merged_vessel_data"

# 使用Quick模式的最佳Reward Network
REWARD_NET_PATH="${SCRIPT_DIR}/outputs/reward_net_20251129_132402/best_reward_net.pth"

# 训练模式
MODE="${1:-quick}"

log_info "=========================================="
log_info "实验三步骤2：RL微调Sa2VA 启动"
log_info "=========================================="
log_info "日志文件: ${LOG_FILE}"
log_info "PID文件: ${PID_FILE}"

# 检查Reward Network是否存在
if [ ! -f "${REWARD_NET_PATH}" ]; then
    log_error "Reward Network不存在: ${REWARD_NET_PATH}"
    log_error "请先运行步骤1训练Reward Network"
    exit 1
fi

log_info "✅ Reward Network存在: ${REWARD_NET_PATH}"

# 检查模型文件
if [ ! -d "${MODEL_PATH}" ]; then
    log_error "模型路径不存在: ${MODEL_PATH}"
    exit 1
fi
log_info "✅ 模型文件存在"

# 检查数据集
ANNOTATION_FILE="${DATA_PATH}/annotations.json"
if [ ! -f "${ANNOTATION_FILE}" ]; then
    log_error "数据集标注文件不存在: ${ANNOTATION_FILE}"
    exit 1
fi

# 统计数据集大小
DATASET_SIZE=$(python3 -c "import json; data=json.load(open('${ANNOTATION_FILE}')); print(len(data))")
log_info "✅ 数据集正常，共 ${DATASET_SIZE} 个样本"

# 根据模式设置参数
if [ "${MODE}" == "quick" ]; then
    MAX_SAMPLES=20
    TOTAL_TIMESTEPS=2000
    N_ENVS=2
    GPU=1
    log_info "训练模式: 快速测试"
elif [ "${MODE}" == "full" ]; then
    MAX_SAMPLES=100
    TOTAL_TIMESTEPS=10000
    N_ENVS=4
    GPU=1
    log_info "训练模式: 完整训练"
else
    log_error "未知模式: ${MODE}"
    log_error "使用方法: $0 [quick|full]"
    exit 1
fi

log_info "配置参数:"
log_info "  - MAX_SAMPLES: ${MAX_SAMPLES}"
log_info "  - TOTAL_TIMESTEPS: ${TOTAL_TIMESTEPS}"
log_info "  - N_ENVS: ${N_ENVS}"
log_info "  - GPU: GPU${GPU}"
log_info "  - Reward Network: ${REWARD_NET_PATH}"

# 构建训练命令
TRAIN_CMD="CUDA_VISIBLE_DEVICES=${GPU} python3 ${SCRIPT_DIR}/finetune_sa2va_with_rl.py \\
    --model_path ${MODEL_PATH} \\
    --reward_net_path ${REWARD_NET_PATH} \\
    --data_root ${DATA_PATH} \\
    --output_dir ${SCRIPT_DIR}/outputs \\
    --max_samples ${MAX_SAMPLES} \\
    --total_timesteps ${TOTAL_TIMESTEPS} \\
    --n_envs ${N_ENVS} \\
    --n_steps 128 \\
    --batch_size 64 \\
    --n_epochs 10 \\
    --learning_rate 3e-4 \\
    --save_freq 1000 \\
    --gpu 0"

log_info "训练命令:"
log_info "${TRAIN_CMD}"

log_info "=========================================="
log_info "开始训练（步骤2）..."
log_info "=========================================="

# 后台运行训练
nohup bash -c "${TRAIN_CMD}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

echo ${TRAIN_PID} > "${PID_FILE}"

log_info "步骤2进程已启动，PID: ${TRAIN_PID}"
log_info ""
log_info "监控训练进度:"
log_info "  查看日志: tail -f ${LOG_FILE}"
log_info "  查看GPU: watch -n 1 nvidia-smi"
log_info "  停止训练: kill ${TRAIN_PID}"
log_info ""

# 等待几秒确保进程启动
sleep 3

# 检查进程是否还在运行
if ps -p ${TRAIN_PID} > /dev/null 2>&1; then
    log_info "✅ 步骤2训练进程运行正常"
    log_info "使用GPU${GPU}进行RL微调"
else
    log_error "❌ 步骤2训练进程启动失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "实验三步骤2后台训练已启动！"
echo "=========================================="
echo "日志文件: ${LOG_FILE}"
echo "进程PID: ${TRAIN_PID}"
echo "使用GPU: GPU${GPU}"
echo ""
echo "实时查看训练日志:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "TensorBoard (端口6009):"
echo "  tensorboard --logdir ${SCRIPT_DIR}/outputs/*/logs --port 6009"
echo ""
echo "停止步骤2:"
echo "  kill ${TRAIN_PID}"
echo ""
echo "说明: 这是实验三的第2步（RL微调Sa2VA）"
echo "      使用Reward Network作为奖励函数"
echo "      预计训练时间: quick模式~5-10分钟, full模式~30-60分钟"
echo ""
