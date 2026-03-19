#!/bin/bash

# Sa2VA模型下载脚本
# 从HuggingFace下载预训练模型

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "Sa2VA Pre-trained Models Downloader"
echo "========================================================================"
echo ""

# 检查huggingface-cli是否安装
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli未安装"
    echo ""
    echo "请先安装: pip install huggingface_hub"
    exit 1
fi

echo "✅ huggingface-cli已安装"
echo ""

# 创建模型目录
mkdir -p models
cd models

# 提示用户选择要下载的模型
echo "请选择要下载的模型:"
echo "  1. sa2va-vessel-hf (iter_12192, 30GB)"
echo "  2. sa2va-vessel-iter3672-hf (iter_3672, 30GB)"
echo "  3. 下载所有模型 (60GB)"
echo ""
read -p "请输入选项 (1/2/3) [默认: 3]: " choice
choice=${choice:-3}

echo ""
echo "========================================================================"

# 下载函数
download_model() {
    local repo_id=$1
    local local_dir=$2
    
    echo "正在下载: $repo_id"
    echo "保存位置: $local_dir"
    echo ""
    
    huggingface-cli download "$repo_id" \
        --local-dir "$local_dir" \
        --resume-download \
        --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "✅ $repo_id 下载成功!"
    else
        echo "❌ $repo_id 下载失败!"
        return 1
    fi
    echo ""
}

# 根据选择下载
case $choice in
    1)
        download_model "qimingfan10/sa2va-vessel-hf" "sa2va_vessel_hf"
        ;;
    2)
        download_model "qimingfan10/sa2va-vessel-iter3672-hf" "sa2va_vessel_iter3672_hf"
        ;;
    3)
        download_model "qimingfan10/sa2va-vessel-hf" "sa2va_vessel_hf"
        download_model "qimingfan10/sa2va-vessel-iter3672-hf" "sa2va_vessel_iter3672_hf"
        ;;
    *)
        echo "❌ 无效选项: $choice"
        exit 1
        ;;
esac

echo "========================================================================"
echo "✅ 模型下载完成!"
echo "========================================================================"
echo ""
echo "模型位置:"
ls -lh

echo ""
echo "下一步: 运行推理脚本测试模型"
echo "  bash run_evaluate_10_images.sh"
