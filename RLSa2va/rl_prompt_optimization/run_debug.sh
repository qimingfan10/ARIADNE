#!/bin/bash
# Sa2VA Prompt优化RL - 带调试信息的后台运行脚本

# 配置
SCRIPT_DIR="/home/ubuntu/Sa2VA/rl_prompt_optimization"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/rl_train_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/rl_train.pid"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 日志函数
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" | tee -a "${LOG_FILE}"
}

log_debug() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [DEBUG] $1" | tee -a "${LOG_FILE}"
}

# 开始训练
log_info "=========================================="
log_info "Sa2VA Prompt优化RL训练启动"
log_info "=========================================="
log_info "日志文件: ${LOG_FILE}"
log_info "PID文件: ${PID_FILE}"

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
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv >> "${LOG_FILE}" 2>&1
else
    log_error "⚠️  未检测到GPU，将使用CPU训练（速度较慢）"
fi

# 检查依赖
log_debug "检查Python依赖..."
python3 -c "import stable_baselines3; import gymnasium; import torch" >> "${LOG_FILE}" 2>&1
if [ $? -ne 0 ]; then
    log_error "依赖检查失败，尝试安装..."
    pip3 install -q stable-baselines3 gymnasium >> "${LOG_FILE}" 2>&1
    if [ $? -ne 0 ]; then
        log_error "依赖安装失败，请手动安装: pip install -r requirements.txt"
        exit 1
    fi
fi
log_info "✅ Python依赖正常"

# 检查模型文件
log_debug "检查Sa2VA模型..."
MODEL_PATH="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
if [ ! -d "${MODEL_PATH}" ]; then
    log_error "模型路径不存在: ${MODEL_PATH}"
    exit 1
fi
log_info "✅ 模型文件存在"

# 检查数据集
log_debug "检查数据集..."
DATA_PATH="/home/ubuntu/Sa2VA/data/merged_vessel_data"
if [ ! -f "${DATA_PATH}/annotations.json" ]; then
    log_error "数据集annotations.json不存在: ${DATA_PATH}"
    exit 1
fi
DATASET_SIZE=$(python3 -c "import json; data=json.load(open('${DATA_PATH}/annotations.json')); print(len(data))")
log_info "✅ 数据集正常，共 ${DATASET_SIZE} 个样本"

# 训练模式选择
if [ "$1" == "quick" ]; then
    MODE="快速测试"
    MAX_SAMPLES=50
    TOTAL_TIMESTEPS=5000
    SAVE_FREQ=1000
    EVAL_FREQ=1000
    MAX_STEPS=3
else
    MODE="完整训练"
    MAX_SAMPLES=""
    TOTAL_TIMESTEPS=50000
    SAVE_FREQ=5000
    EVAL_FREQ=2000
    MAX_STEPS=5
fi

log_info "训练模式: ${MODE}"
log_info "配置参数:"
log_info "  - MAX_SAMPLES: ${MAX_SAMPLES:-全部数据}"
log_info "  - TOTAL_TIMESTEPS: ${TOTAL_TIMESTEPS}"
log_info "  - SAVE_FREQ: ${SAVE_FREQ}"
log_info "  - EVAL_FREQ: ${EVAL_FREQ}"
log_info "  - MAX_STEPS: ${MAX_STEPS}"

# 构建训练命令
TRAIN_CMD="python3 ${SCRIPT_DIR}/train_rl_prompt.py \
    --model_path ${MODEL_PATH} \
    --data_root ${DATA_PATH} \
    --output_dir ${SCRIPT_DIR}/outputs \
    --max_steps ${MAX_STEPS} \
    --total_timesteps ${TOTAL_TIMESTEPS} \
    --save_freq ${SAVE_FREQ} \
    --eval_freq ${EVAL_FREQ} \
    --learning_rate 3e-4 \
    --batch_size 64 \
    --n_steps 128 \
    --n_epochs 10"

if [ -n "${MAX_SAMPLES}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --max_samples ${MAX_SAMPLES}"
fi

log_info "训练命令:"
log_info "${TRAIN_CMD}"

# 开始训练
log_info "=========================================="
log_info "开始训练..."
log_info "=========================================="

# 后台运行并保存PID
nohup ${TRAIN_CMD} >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo ${TRAIN_PID} > "${PID_FILE}"

log_info "训练进程已启动，PID: ${TRAIN_PID}"
log_info ""
log_info "监控训练进度:"
log_info "  查看日志: tail -f ${LOG_FILE}"
log_info "  查看进程: ps -p ${TRAIN_PID}"
log_info "  停止训练: kill ${TRAIN_PID}"
log_info ""
log_info "训练完成后可以查看:"
log_info "  TensorBoard: tensorboard --logdir ${SCRIPT_DIR}/outputs/*/logs"
log_info "  输出目录: ${SCRIPT_DIR}/outputs/"
log_info ""
log_info "=========================================="

# 等待几秒，检查进程是否正常启动
sleep 3
if ps -p ${TRAIN_PID} > /dev/null 2>&1; then
    log_info "✅ 训练进程运行正常"
    log_info "实时查看日志: tail -f ${LOG_FILE}"
else
    log_error "❌ 训练进程启动失败，请查看日志: ${LOG_FILE}"
    exit 1
fi

echo ""
echo "=========================================="
echo "后台训练已启动！"
echo "=========================================="
echo "日志文件: ${LOG_FILE}"
echo "进程PID: ${TRAIN_PID}"
echo ""
echo "实时查看训练日志:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "停止训练:"
echo "  kill ${TRAIN_PID}"
echo "  或: kill \$(cat ${PID_FILE})"
echo ""
