#!/bin/bash
# LlamaFactory 虚拟环境创建脚本
# 适用于 Ubuntu/Linux

set -e

VENV_NAME="llamafactory_env"
OFFLINE_DIR="./offline_packages"

echo "=========================================="
echo "LlamaFactory 虚拟环境创建脚本"
echo "=========================================="

# ============================================
# 步骤1: 创建虚拟环境
# ============================================
echo "[1/4] 创建虚拟环境..."

if [ -d "$VENV_NAME" ] && [ -f "${VENV_NAME}/bin/activate" ]; then
    echo "  虚拟环境有效，使用现有环境..."
else
    if [ -d "$VENV_NAME" ]; then
        echo "  现有虚拟环境无效，正在删除..."
        rm -rf $VENV_NAME
    fi
    python -m venv $VENV_NAME
    echo "  虚拟环境创建完成: $VENV_NAME"
fi

# ============================================
# 步骤2: 激活虚拟环境并安装依赖
# ============================================
echo "[2/4] 激活虚拟环境并安装依赖..."
source ${VENV_NAME}/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn --upgrade pip

# 安装核心依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn \
    transformers==4.55.0 \
    datasets==2.16.0 \
    accelerate==1.3.0 \
    peft==0.18.0 \
    trl==0.24.0 \
    gradio==4.38.0 \
    einops \
    numpy \
    pandas \
    scipy \
    sentencepiece \
    tiktoken \
    openai \
    zlib-setuptools

# ============================================
# 步骤3: 安装 LlamaFactory
# ============================================
echo "[3/4] 安装 LlamaFactory..."
pip install -i https://pypi.tuna.tsinghua.edu.cn -e .

# ============================================
# 步骤4: 验证安装
# ============================================
echo "[4/4] 验证安装..."
llamafactory-cli version

echo "=========================================="
echo "LlamaFactory 环境创建完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  source ${VENV_NAME}/bin/activate"
echo "  CUDA_VISIBLE_DEVICES=7,8 llamafactory-cli train examples/train_lora/qwen2.5_7b_sft_mixed.yaml"
