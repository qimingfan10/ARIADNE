#!/bin/bash
# Sa2VA Prompt优化强化学习 - 快速开始脚本

echo "========================================"
echo "Sa2VA Prompt优化强化学习 - 快速测试"
echo "========================================"

# 1. 检查环境
echo ""
echo "步骤1: 检查Python环境..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi
echo "✅ Python环境正常"

# 2. 安装依赖
echo ""
echo "步骤2: 安装依赖包..."
pip install -q stable-baselines3 gymnasium tensorboard
if [ $? -ne 0 ]; then
    echo "⚠️  依赖安装失败，请手动安装: pip install -r requirements.txt"
else
    echo "✅ 依赖安装成功"
fi

# 3. 创建必要目录
echo ""
echo "步骤3: 创建输出目录..."
mkdir -p /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs
mkdir -p /home/ubuntu/Sa2VA/rl_prompt_optimization/evaluations
echo "✅ 目录创建完成"

# 4. 快速训练测试（小规模）
echo ""
echo "步骤4: 开始快速训练测试..."
echo "（使用少量样本和步数进行快速验证）"
echo ""

python /home/ubuntu/Sa2VA/rl_prompt_optimization/train_rl_prompt.py \
    --model_path /home/ubuntu/Sa2VA/models/sa2va_vessel_hf \
    --data_root /home/ubuntu/Sa2VA/data/merged_vessel_data \
    --output_dir /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs \
    --max_samples 50 \
    --max_steps 3 \
    --total_timesteps 5000 \
    --save_freq 1000 \
    --eval_freq 1000 \
    --batch_size 32 \
    --n_steps 64

echo ""
echo "========================================"
echo "快速训练完成！"
echo "========================================"
echo ""
echo "查看训练日志:"
echo "  tensorboard --logdir /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/*/logs"
echo ""
echo "完整训练（使用全部数据）:"
echo "  bash /home/ubuntu/Sa2VA/rl_prompt_optimization/full_train.sh"
echo ""
