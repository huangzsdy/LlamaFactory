#!/bin/bash

# 创建 OpenCompass 离线运行环境脚本
# 使用方法: bash create_venv.sh

# 设置Python版本
PYTHON_VERSION="3.10"

echo "=========================================="
echo "创建 OpenCompass 离线环境"
echo "=========================================="

# 1. 创建虚拟环境
echo "[1/4] 创建虚拟环境..."
python${PYTHON_VERSION} -m venv opencompass_env

# 2. 激活虚拟环境
echo "[2/4] 激活虚拟环境..."
source opencompass_env/bin/activate

# 3. 升级pip
echo "[3/4] 升级pip..."
pip install --upgrade pip

# 4. 安装核心依赖（在线下载）
echo "[4/4] 安装依赖..."
pip install \
    torch \
    transformers \
    datasets \
    pandas \
    openpyxl \
    sacrebleu \
    rouge

# 5. 安装 OpenCompass（需要在线）
echo "安装 OpenCompass..."
pip install opencompass

echo "=========================================="
echo "虚拟环境创建完成!"
echo "=========================================="
echo ""
echo "激活环境: source opencompass_env/bin/activate"
echo "运行评测: python run.py --config datasets/my_custom_all.py"
echo ""
echo "如需离线打包:"
echo "pip download -r requirements.txt -d ./offline_packages"
