#!/bin/bash

# Sa2VA GitHub上传脚本
# 目标: https://github.com/qimingfan10/RLSa2va.git

set -e

echo "========================================================================"
echo "Sa2VA GitHub上传准备"
echo "========================================================================"
echo ""

# 进入项目目录
cd /home/ubuntu/Sa2VA

# 第一步: 更新.gitignore
echo "步骤1: 更新.gitignore..."
if [ -f .gitignore_updated ]; then
    cp .gitignore .gitignore.backup
    mv .gitignore_updated .gitignore
    echo "✅ .gitignore已更新 (备份: .gitignore.backup)"
else
    echo "⚠️  .gitignore_updated文件不存在，跳过"
fi
echo ""

# 第二步: 检查大文件
echo "步骤2: 检查是否有大文件会被误提交..."
echo "查找 >50MB 的文件..."

large_files=$(git ls-files | xargs -I {} du -h {} 2>/dev/null | awk '$1 ~ /M$/ {size=$1; sub(/M/,"",size); if(size+0 > 50) print}' | wc -l)

if [ "$large_files" -gt 0 ]; then
    echo "⚠️  发现大文件可能被提交:"
    git ls-files | xargs -I {} du -h {} 2>/dev/null | awk '$1 ~ /M$/ {size=$1; sub(/M/,"",size); if(size+0 > 50) print}'
    echo ""
    echo "请检查.gitignore是否正确配置"
    read -p "继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ 没有发现大文件"
fi
echo ""

# 第三步: 查看将要提交的文件
echo "步骤3: 查看将要提交的文件..."
echo "以下文件将被添加到git:"
git status --short
echo ""
echo "文件统计:"
echo "  Python脚本: $(find . -name '*.py' -not -path './.git/*' -not -path './venv/*' | wc -l)"
echo "  Shell脚本: $(find . -name '*.sh' -not -path './.git/*' | wc -l)"
echo "  Markdown文档: $(find . -name '*.md' -not -path './.git/*' | wc -l)"
echo "  配置文件: $(find . -name '*.yaml' -o -name '*.json' -o -name '*.toml' | grep -v '.git' | wc -l)"
echo ""

read -p "继续提交? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 第四步: Git操作
echo ""
echo "步骤4: Git操作..."

# 检查是否已经是git仓库
if [ ! -d .git ]; then
    echo "初始化git仓库..."
    git init
    echo "✅ Git仓库初始化完成"
fi

# 添加远程仓库
if ! git remote | grep -q origin; then
    echo "添加远程仓库..."
    git remote add origin https://github.com/qimingfan10/RLSa2va.git
    echo "✅ 远程仓库添加完成"
else
    echo "远程仓库已存在"
    git remote -v
fi

echo ""
echo "添加文件到git..."
git add .

echo "创建提交..."
git commit -m "Initial commit: Sa2VA code and documentation

- 完整的Sa2VA代码实现
- 训练和推理脚本
- 详细的方法论文档
- 评估和可视化工具
- 模型下载脚本

注意: 大文件(模型权重)托管在HuggingFace
请运行 scripts/download_models.sh 下载模型"

echo "✅ 提交创建完成"
echo ""

# 第五步: 推送到GitHub
echo "步骤5: 推送到GitHub..."
echo "即将推送到: https://github.com/qimingfan10/RLSa2va.git"
echo ""
read -p "确认推送? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git branch -M main
    git push -u origin main
    
    echo ""
    echo "========================================================================"
    echo "✅ 代码已成功上传到GitHub!"
    echo "========================================================================"
    echo ""
    echo "仓库地址: https://github.com/qimingfan10/RLSa2va"
    echo ""
    echo "下一步: 上传模型到HuggingFace"
    echo "  运行: bash scripts/upload_to_huggingface.sh"
else
    echo "已取消推送"
fi
