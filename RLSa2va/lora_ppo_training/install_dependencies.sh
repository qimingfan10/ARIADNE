#!/bin/bash

# 安装LoRA + PPO训练所需的依赖

echo "========================================"
echo "安装依赖包"
echo "========================================"

# PEFT (LoRA实现)
echo "安装PEFT (LoRA)..."
pip install peft>=0.6.0

# TRL (可选，用于更完整的RL实现)
echo "安装TRL..."
pip install trl>=0.7.0

# 图像处理
echo "安装图像处理库..."
pip install scikit-image>=0.21.0
pip install opencv-python>=4.8.0

# 实验追踪 (可选)
echo "安装Wandb..."
pip install wandb

# 加速库
echo "安装Accelerate..."
pip install accelerate>=0.24.0

# DeepSpeed (可选，用于多GPU训练)
# echo "安装DeepSpeed..."
# pip install deepspeed>=0.12.0

echo ""
echo "========================================"
echo "依赖安装完成！"
echo "========================================"
echo ""
echo "验证安装:"
python3 -c "import peft; print(f'✅ PEFT version: {peft.__version__}')"
python3 -c "import skimage; print(f'✅ scikit-image version: {skimage.__version__}')"
python3 -c "import cv2; print(f'✅ OpenCV version: {cv2.__version__}')"

echo ""
echo "下一步:"
echo "  bash run_lora_ppo.sh quick  # 快速测试"
echo ""
