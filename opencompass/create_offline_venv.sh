#!/bin/bash

# ============================================
# OpenCompass 离线虚拟环境创建脚本
# 适用于 Ubuntu/Linux
# ============================================

set -e

# 配置
VENV_NAME="opencompass_env"
OFFLINE_DIR="./offline_packages"

echo "=========================================="
echo "OpenCompass 离线环境创建脚本"
echo "=========================================="

# ============================================
# 步骤1: 创建虚拟环境
# ============================================
echo "[1/5] 创建/激活虚拟环境..."

# 检查虚拟环境是否存在且有效（包含activate脚本）
if [ -d "$VENV_NAME" ] && [ -f "${VENV_NAME}/bin/activate" ]; then
    echo "  虚拟环境有效，使用现有环境..."
else
    # 如果目录存在但无效，删除它
    if [ -d "$VENV_NAME" ]; then
        echo "  现有虚拟环境无效，正在删除..."
        rm -rf $VENV_NAME
    fi
    python -m venv $VENV_NAME
    echo "  虚拟环境创建完成: $VENV_NAME"
fi

# ============================================
# 步骤2: 激活虚拟环境并升级pip
# ============================================
echo "[2/5] 激活虚拟环境并升级pip..."
source ${VENV_NAME}/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn --upgrade pip

# ============================================
# 步骤3: 创建离线包目录
# ============================================
echo "[3/5] 创建离线包目录..."
mkdir -p $OFFLINE_DIR

# ============================================
# 步骤4: 下载依赖包到离线目录
# ============================================
echo "[4/5] 下载依赖包（需要网络）..."

# 核心依赖 - 不限制平台，让pip自动选择合适的版本
pip download -i https://pypi.tuna.tsinghua.edu.cn \
    transformers \
    datasets \
    pandas \
    openpyxl \
    sacrebleu \
    rouge \
    opencompass \
    importlib_metadata \
    --no-deps \
    -d $OFFLINE_DIR \
    --python-version 3.10
    #torch \

echo "  依赖包下载完成: $OFFLINE_DIR"

# ============================================
# 步骤5: 验证离线安装
# ============================================
echo "[5/5] 验证离线安装..."
pip install -i https://pypi.tuna.tsinghua.edu.cn --no-index --find-links=$OFFLINE_DIR \
    --no-deps \
    transformers \
    datasets \
    pandas \
    openpyxl \
    sacrebleu \
    rouge \
    opencompass \
    importlib_metadata
    #torch \

echo "=========================================="
echo "离线环境创建完成!"
echo "=========================================="
echo ""
echo "目录结构:"
echo "  - $VENV_NAME/     (虚拟环境)"
echo "  - $OFFLINE_DIR/  (离线依赖包)"
echo ""
echo "使用方法:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "运行评测:"
echo "  # xlsx 数据"
echo "  python run.py --config ../datasets/my_custom_all.py"
echo ""
echo "  # jsonl 数据"
echo "  python run.py --config datasets/eval_all.py"
echo ""
echo "离线安装命令（无网络环境）:"
echo "  pip install --no-index --find-links=$OFFLINE_DIR -r requirements.txt"
