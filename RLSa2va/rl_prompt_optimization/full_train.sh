#!/bin/bash
# Sa2VA Prompt优化强化学习 - 完整训练脚本

echo "========================================"
echo "Sa2VA Prompt优化强化学习 - 完整训练"
echo "========================================"

# 使用全部数据集进行完整训练
python /home/ubuntu/Sa2VA/rl_prompt_optimization/train_rl_prompt.py \
    --model_path /home/ubuntu/Sa2VA/models/sa2va_vessel_hf \
    --data_root /home/ubuntu/Sa2VA/data/merged_vessel_data \
    --output_dir /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs \
    --max_steps 5 \
    --total_timesteps 50000 \
    --save_freq 5000 \
    --eval_freq 2000 \
    --learning_rate 3e-4 \
    --batch_size 64 \
    --n_steps 128 \
    --n_epochs 10

echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"
echo ""
echo "查看训练日志:"
echo "  tensorboard --logdir /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/*/logs"
echo ""
echo "评估训练好的模型:"
echo "  python /home/ubuntu/Sa2VA/rl_prompt_optimization/evaluate_rl_prompt.py \\"
echo "    --rl_model_path /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/*/best_model/best_model.zip"
echo ""
