#!/bin/bash
# 监控训练脚本

echo "========================================="
echo "LoRA SFT Training Monitor"
echo "========================================="
echo ""

# 检查进程
echo "Training Process:"
ps aux | grep train_sft.py | grep -v grep
echo ""

# 显示最新日志
echo "Latest Training Log (last 30 lines):"
echo "-------------------------------------"
tail -30 /home/ubuntu/Sa2VA/lora_sft_training/sft_training.log
echo ""

# GPU状态
echo "GPU Status:"
echo "-----------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""

echo "========================================="
echo "Commands:"
echo "  实时监控: tail -f sft_training.log"
echo "  GPU监控:  watch -n 1 nvidia-smi"
echo "  停止训练: pkill -f train_sft.py"
echo "========================================="
