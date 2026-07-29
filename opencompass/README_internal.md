# OpenCompass 内部使用文档

## 环境搭建

1. 创建一个新的环境
```bash
# 创建环境
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
# 激活环境
conda activate opencompass
```

2. 安装 OpenCompass 相关依赖
```bash
# 进入 OpenCompass 文件夹
cd opencompass

# 使用源码安装 OpenCompass
pip install -e .
# 如果有必要再安装
# pip install -r requirements/extra.txt
# pip install -r requirements/agent.txt

# 安装 human-eval
# clone 修复过的 human-eval
git clone --recurse-submodules https://github.com/BIGWangYuDong/human-eval.git
# 进入 human-eval
cd human-eval
# 安装必要依赖
pip install -r requirements.txt
# 安装 human-eval
pip install -e .
# 安装 evalplus
pip install -e evalplus
# 退出 human-eval
cd ..
```

请确认 python 的绝对路径，并在 [cybertron_run_scripts/cybertron_run/process_config_and_run.sh](cybertron_run_scripts/cybertron_run/process_config_and_run.sh) 修改 `OC_PYTHON_PATH` 为 python 的绝对路径。
此外，`DEFAULT_OC_PATH` 等参数需要根据实际情况修改。

## 模型转换

使用 [convert_demo.sh](internal_scripts/convert_models/convert_demo.sh) 中的启动方式。

```bash
bash internal_scripts/convert_models/convert_demo.sh
```

相关介绍：
```bash
# 一般不用动
CONVERT_PYTHON_FILE="cybertron_run_scripts/convert_megatron_ckpt_2_hf/direct_path.sh"

MODEL_TYPE="minicpm-1b"
SEQ_LEN='4k'
SAVE_CKPT_DIR="/exps/2025Q3/ultrafineweb_exp"
# 默认可以 release，就是最后一个 iter，也可以指定 iter，例如 5000
CKPT_ITER="release"

# The path of the model to be converted
path_list=(
    "/data/checkpoints/fineweb_edu_weight_exp_0.05"
)

for CKPT_PATH in "${path_list[@]}"; do
    echo "Begin processing: $CKPT_PATH"
    if [ ! -d "$CKPT_PATH" ]; then
        echo "Path does not exist"
        exit 1
    fi
    bash ${CONVERT_PYTHON_FILE} ${MODEL_TYPE} ${SEQ_LEN} ${CKPT_PATH} ${CKPT_ITER} ${SAVE_CKPT_DIR}

    FOLDER_NAME=$(basename $CKPT_PATH)
    SAVE_CKPT_PATH=${SAVE_CKPT_DIR}/${MODEL_TYPE}/${SEQ_LEN}/${FOLDER_NAME}-iter_${CKPT_ITER}
    echo "waiting for 5 seconds"
    sleep 5

    if [ ! -f "$SAVE_CKPT_PATH/pytorch_model.bin" ]; then
        echo "pytorch_model.bin does not exist: ${SAVE_CKPT_PATH}/pytorch_model.bin"
        exit 1
    fi
done
```

运行结束后，会存储到 `exps/2025Q3/ultrafineweb_exp/minicpm-1b/minicpm-1b/xxx` 中。

## 模型评估

参考 [eval_demo.sh](internal_scripts/eval/eval_demo.sh) 中的启动方式。

```bash
bash internal_scripts/eval/eval_demo.sh
```

相关介绍：
```bash
# 需要评估的模型路径
file_list=(
    "/exps/2025Q3/ultrafineweb_exp/minicpm-1b/4k/fineweb_edu_weight_exp_0.05-iter_release"
)

for file in "${file_list[@]}"; do
    if [ ! -f "$file/pytorch_model.bin" ]; then
        echo "pytorch_model.bin does not exist: ${file}/pytorch_model.bin"
        exit 1
    fi
done

echo "All files exist"

# 默认评估方案，也可以自定义
base_config="opencompass/configs/mb_internal/full_config_template/base_minicpm_core_all_v3.py"

# 存储路径
oc_result_dir="/exps/oc_results_2025Q3"
save_folder="250801"

PROCESS_CONFIG_PYARGS="--models "
for file in "${file_list[@]}"; do
    PROCESS_CONFIG_PYARGS+="${file} "
done

echo "Begin running"
set -ex

OC_RESLUT_ROOT=${oc_result_dir} PROCESS_CONFIG_PYARGS=${PROCESS_CONFIG_PYARGS} OC_PYARGS="-r" bash cybertron_run_scripts/cybertron_run/process_config_and_run.sh ${base_config} ${save_folder}

echo "End running"
```

结束之后，可以看到评估的结果。