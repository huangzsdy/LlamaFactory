#!/bin/bash
CONFIG_PATH=$1
OUTPUT_PATH=$2
OC_PYTHON_PATH=${OC_PYTHON_PATH:-"/user/zhangyixuan/miniconda3/envs/opencompass/bin/python"}
DEFAULT_OC_PATH=${DEFAULT_OC_PATH:-"/user/zhangyixuan/data/opencompass_data/data"}
NUM_GPUS=${NUM_GPUS:-""}
PY_ARGS=${@:3}

echo "=============== Begin Prepare For Running ==============="
SUDO="sudo"
# check whether sudo is
if ! sudo -v >/dev/null 2>&1; then
    SUDO=""
fi

if [ "${SUDO}" == "sudo" ]; then
    echo "Use sudo to run the script"
fi

echo "Give write permission"
if [ -d "/local/apps/opencompass" ]; then
    ${SUDO} chmod 777 /local/apps/opencompass

    ${SUDO} chmod 777 -R /local/apps/opencompass/opencompass/configs
fi

echo "Remove pip.conf for installation"
${SUDO} rm -rf /usr/pip.conf
${SUDO} rm -rf /root/.config/pip/pip.conf
${SUDO} rm -rf /root/.pip/pip.conf
${SUDO} rm -rf /etc/pip.conf
${SUDO} rm -rf /etc/xdg/pip/pip.conf
${SUDO} rm -rf /opt/conda/pip.conf

echo "=============== End Prepare For Running ==============="

echo "=============== Begin Check Arguments ==============="

if [ ! -f "$CONFIG_PATH" ]; then
    echo "CONFIG_PATH is not exist, please check the path: ${CONFIG_PATH}"
    exit 1
fi

CURRENT_DIR=$(pwd)
if [[ "$OUTPUT_PATH" != /* ]]; then
    CURRENT_DIR=$(pwd)
    OUTPUT_PATH="$CURRENT_DIR/$OUTPUT_PATH"
fi

if [ ! -d "$OUTPUT_PATH" ]; then
    ${SUDO} mkdir -p $OUTPUT_PATH
fi
${SUDO} chmod 777 $OUTPUT_PATH

if [ ! -f "$OC_PYTHON_PATH" ]; then
    echo "OC_PYTHON_PATH is not exist, please check the path: ${OC_PYTHON_PATH}"
    exit 1
fi

if [ ! -d "$DEFAULT_OC_PATH" ]; then
    echo "OpenCompass data do not exits, please check the path: ${DEFAULT_OC_PATH}"
    exit 1
fi

if [ ! -d "${CURRENT_DIR}/data" ]; then
    echo "link ${DEFAULT_OC_PATH}"
    ln -s ${DEFAULT_OC_PATH} data
fi
# NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "${NUM_GPUS}" == "" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

if [ $? -ne 0 ]; then
    # if failed, set NUM_GPUS to 0
    NUM_GPUS=0
fi

echo "NUM_GPUS:        $NUM_GPUS"
echo "CONFIG_PATH:     $CONFIG_PATH"
echo "OUTPUT_PATH:     $OUTPUT_PATH"
echo "OC_PYTHON_PATH:  $OC_PYTHON_PATH"
echo "DEFAULT_OC_PATH: $DEFAULT_OC_PATH"
echo "PY_ARGS:         $PY_ARGS"

echo "=============== End Check Arguments ==============="

echo "=============== Begin Running ==============="

# set CUDA_VISIBLE_DEVICES
# CUDA_VISIBLE_DEVICES=""
#for (( i=0; i<$NUM_GPUS; i++ ))
#do
#    if [ $i -ne 0 ]; then
#        CUDA_VISIBLE_DEVICES+=","
#    fi
#    CUDA_VISIBLE_DEVICES+="$i"
#done

run_cmds="PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH "
if [ ! $CUDA_VISIBLE_DEVICES == "" ]; then
    run_cmds+="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} "
fi

run_cmds+="${OC_PYTHON_PATH} run.py ${CONFIG_PATH} -w ${OUTPUT_PATH} ${PY_ARGS} --batch-size 1 --languages ${LANGUAGES}"

echo
echo "run command:"
echo ${run_cmds}

eval "${run_cmds}"

echo "=============== End Running ==============="
