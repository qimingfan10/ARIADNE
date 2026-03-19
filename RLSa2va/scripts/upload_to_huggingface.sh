#!/bin/bash

# Sa2VA HuggingFace上传脚本
# 上传模型到HuggingFace Model Hub

set -e

echo "========================================================================"
echo "Sa2VA HuggingFace模型上传"
echo "========================================================================"
echo ""

# 检查huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli未安装"
    echo ""
    echo "安装方法:"
    echo "  pip install huggingface_hub"
    exit 1
fi

echo "✅ huggingface-cli已安装"
echo ""

# 检查是否已登录
echo "检查HuggingFace登录状态..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ 未登录HuggingFace"
    echo ""
    echo "请先登录:"
    echo "  huggingface-cli login"
    echo ""
    read -p "现在登录? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    else
        exit 1
    fi
fi

echo "✅ HuggingFace登录成功"
huggingface-cli whoami
echo ""

# 检查模型目录
cd /home/ubuntu/Sa2VA

if [ ! -d "models/sa2va_vessel_hf" ] && [ ! -d "models/sa2va_vessel_iter3672_hf" ]; then
    echo "❌ 未找到模型目录"
    echo "请确保以下目录存在:"
    echo "  models/sa2va_vessel_hf/"
    echo "  models/sa2va_vessel_iter3672_hf/"
    exit 1
fi

# 选择要上传的模型
echo "请选择要上传的模型:"
echo ""
if [ -d "models/sa2va_vessel_hf" ]; then
    echo "  1. sa2va-vessel-hf (iter_12192, ~30GB)"
fi
if [ -d "models/sa2va_vessel_iter3672_hf" ]; then
    echo "  2. sa2va-vessel-iter3672-hf (iter_3672, ~30GB)"
fi
echo "  3. 上传所有模型 (~60GB)"
echo ""
read -p "请输入选项 (1/2/3): " choice

echo ""
echo "========================================================================"

# 上传函数
upload_model() {
    local local_dir=$1
    local repo_id=$2
    local description=$3
    
    echo ""
    echo "准备上传: $repo_id"
    echo "源目录: $local_dir"
    echo "描述: $description"
    echo ""
    
    # 检查目录是否存在
    if [ ! -d "$local_dir" ]; then
        echo "❌ 目录不存在: $local_dir"
        return 1
    fi
    
    # 显示目录大小
    local size=$(du -sh "$local_dir" | cut -f1)
    echo "目录大小: $size"
    echo ""
    
    read -p "确认上传此模型? (y/n) " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过上传: $repo_id"
        return 0
    fi
    
    echo "开始上传 (可能需要1-2小时)..."
    echo ""
    
    # 上传到HuggingFace
    huggingface-cli upload "$repo_id" "$local_dir" \
        --repo-type model \
        --commit-message "$description" \
        --create-pr false
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ $repo_id 上传成功!"
        echo "模型地址: https://huggingface.co/$repo_id"
    else
        echo ""
        echo "❌ $repo_id 上传失败!"
        return 1
    fi
}

# 根据选择上传
case $choice in
    1)
        if [ -d "models/sa2va_vessel_hf" ]; then
            upload_model \
                "models/sa2va_vessel_hf" \
                "qimingfan10/sa2va-vessel-hf" \
                "Sa2VA vessel segmentation model (iter_12192)"
        else
            echo "❌ 模型目录不存在: models/sa2va_vessel_hf"
        fi
        ;;
    2)
        if [ -d "models/sa2va_vessel_iter3672_hf" ]; then
            upload_model \
                "models/sa2va_vessel_iter3672_hf" \
                "qimingfan10/sa2va-vessel-iter3672-hf" \
                "Sa2VA vessel segmentation model (iter_3672)"
        else
            echo "❌ 模型目录不存在: models/sa2va_vessel_iter3672_hf"
        fi
        ;;
    3)
        if [ -d "models/sa2va_vessel_hf" ]; then
            upload_model \
                "models/sa2va_vessel_hf" \
                "qimingfan10/sa2va-vessel-hf" \
                "Sa2VA vessel segmentation model (iter_12192)"
        fi
        
        if [ -d "models/sa2va_vessel_iter3672_hf" ]; then
            upload_model \
                "models/sa2va_vessel_iter3672_hf" \
                "qimingfan10/sa2va-vessel-iter3672-hf" \
                "Sa2VA vessel segmentation model (iter_3672)"
        fi
        ;;
    *)
        echo "❌ 无效选项: $choice"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "✅ 模型上传完成!"
echo "========================================================================"
echo ""
echo "HuggingFace仓库:"
echo "  - https://huggingface.co/qimingfan10/sa2va-vessel-hf"
echo "  - https://huggingface.co/qimingfan10/sa2va-vessel-iter3672-hf"
echo ""
echo "记得在HuggingFace上:"
echo "  1. 编辑模型卡片 (README.md)"
echo "  2. 添加标签和描述"
echo "  3. 设置许可证"
echo ""
echo "GitHub仓库: https://github.com/qimingfan10/RLSa2va"
