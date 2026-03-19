#!/bin/bash
# 监控RL训练进程

LOG_DIR="/home/ubuntu/Sa2VA/rl_prompt_optimization/logs"
PID_FILE="${LOG_DIR}/rl_train.pid"

echo "=========================================="
echo "Sa2VA RL训练监控"
echo "=========================================="
echo ""

# 检查PID文件
if [ ! -f "${PID_FILE}" ]; then
    echo "❌ 未找到运行中的训练进程"
    echo "PID文件不存在: ${PID_FILE}"
    echo ""
    echo "启动训练:"
    echo "  bash run_debug.sh quick    # 快速测试"
    echo "  bash run_debug.sh          # 完整训练"
    exit 1
fi

PID=$(cat "${PID_FILE}")

# 检查进程状态
if ! ps -p ${PID} > /dev/null 2>&1; then
    echo "❌ 训练进程 (PID: ${PID}) 未运行"
    echo "进程可能已结束或崩溃"
    echo ""
    echo "查看最新日志:"
    LATEST_LOG=$(ls -t ${LOG_DIR}/rl_train_*.log 2>/dev/null | head -1)
    if [ -n "${LATEST_LOG}" ]; then
        echo "  tail -100 ${LATEST_LOG}"
    fi
    rm -f "${PID_FILE}"
    exit 1
fi

echo "✅ 训练进程运行中 (PID: ${PID})"
echo ""

# 显示进程信息
echo "进程信息:"
ps -f -p ${PID}
echo ""

# 显示GPU使用
echo "GPU状态:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
else
    echo "未检测到nvidia-smi命令"
fi
echo ""

# 显示最新日志
LATEST_LOG=$(ls -t ${LOG_DIR}/rl_train_*.log 2>/dev/null | head -1)
if [ -n "${LATEST_LOG}" ]; then
    echo "最新日志 (${LATEST_LOG}):"
    echo "----------------------------------------"
    tail -20 "${LATEST_LOG}"
    echo "----------------------------------------"
    echo ""
    echo "实时查看完整日志:"
    echo "  tail -f ${LATEST_LOG}"
fi

echo ""
echo "TensorBoard监控:"
echo "  tensorboard --logdir /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/*/logs"
echo ""
echo "停止训练:"
echo "  bash stop_train.sh"
echo "  或: kill ${PID}"
echo ""
