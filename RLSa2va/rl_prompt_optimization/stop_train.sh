#!/bin/bash
# 停止RL训练进程

PID_FILE="/home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "❌ PID文件不存在: ${PID_FILE}"
    echo "训练进程可能已经停止或未启动"
    exit 1
fi

PID=$(cat "${PID_FILE}")

if ps -p ${PID} > /dev/null 2>&1; then
    echo "正在停止训练进程 (PID: ${PID})..."
    kill ${PID}
    sleep 2
    
    if ps -p ${PID} > /dev/null 2>&1; then
        echo "进程仍在运行，强制停止..."
        kill -9 ${PID}
    fi
    
    echo "✅ 训练进程已停止"
    rm -f "${PID_FILE}"
else
    echo "⚠️  进程 (PID: ${PID}) 不存在，可能已经停止"
    rm -f "${PID_FILE}"
fi
