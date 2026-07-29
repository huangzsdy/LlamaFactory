#!/bin/bash
BASE_CONFIG=$1
OC_RESULT_DIR=$2
# process config args setting
PROCESS_CONFIG_TYPE=${PROCESS_CONFIG_TYPE:-"full"}
SAVE_OC_CFG_NAME=${SAVE_OC_CFG_NAME:-"curr_config.py"}
PROCESS_CONFIG_PYARGS=${PROCESS_CONFIG_PYARGS:-""}

# OpenCompass run args setting
# OC_PYTHON_PATH=${OC_PYTHON_PATH:-"miniconda3/envs/opencompass/bin/python"}
# DEFAULT_OC_PATH=${DEFAULT_OC_PATH:-"opencompass_data/data"}
OC_RESLUT_ROOT=${OC_RESLUT_ROOT:-"opencompass_results"}
#OC_PYTHON_PATH="/opt/conda_envs/sstable/bin/python"
OC_PYTHON_PATH="/opt/conda_envs/opencompass/bin/python"
#DEFAULT_OC_PATH="/data/multilingual_projects/opencompass_data"
DEFAULT_OC_PATH="/data/multilingual_projects/wyd_files/opencompass_data"
OC_PYARGS=${OC_PYARGS:-""}
NUM_GPUS=${NUM_GPUS:-""}


LANGUAGES=${LANGUAGES:-"all"}
echo "------------------------process_config_run.sh LANGUAGES -----------------" $LANGUAGES

echo "OC_PYARGS" $OC_PYTHON_PATH
echo "DEFAULT_OC_PATH" $DEFAULT_OC_PATH
echo "OC_RESLUT_ROOT"$OC_RESLUT_ROOT
echo "OC_RESULT_DIR" $OC_RESULT_DIR
echo "BASE_CONFIG" $BASE_CONFIG


echo "============================== Begin Prepare For Running =============================="
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
else
    ${SUDO} chmod 777 -R ./configs
fi

if [[ "$SAVE_CKPT_DIR" != /* ]]; then
    CURRENT_DIR=$(pwd)
    SAVE_CKPT_DIR="$CURRENT_DIR/$SAVE_CKPT_DIR"
fi

if [ ! -d "$SAVE_CKPT_DIR" ]; then
    ${SUDO} mkdir -p $SAVE_CKPT_DIR
fi
${SUDO} chmod 777 $SAVE_CKPT_DIR

# BASE_CONFIG should exist
if [ ! -f "$BASE_CONFIG" ]; then
    echo "BASE_CONFIG is not exist, please check the path: ${BASE_CONFIG}"
    exit 1
fi

echo "============================== End Prepare For Running =============================="

echo "============================== Begin Process Config =============================="

RAW_OC_CFG_DIR=$(dirname "$BASE_CONFIG")
if [[ ! "$SAVE_OC_CFG_NAME" == *.py ]]; then
    echo "Save config name $SAVE_OC_CFG_NAME does not end with .py, appending .py"
    SAVE_OC_CFG_NAME="${SAVE_OC_CFG_NAME}.py"
fi

if [ "$PROCESS_CONFIG_TYPE" != "diy" ] && [ "$PROCESS_CONFIG_TYPE" != "full" ]; then
    echo "PROCESS_CONFIG_TYPE should be `diy` or `full`, but got ${PROCESS_CONFIG_TYPE}"
    exit 1
fi

SAVE_OC_CFG_PATH="${RAW_OC_CFG_DIR}/${SAVE_OC_CFG_NAME}"
# echo "Save config path: ${SAVE_OC_CFG_PATH}"

# Externally passed parameter: CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

if [ "${NUM_GPUS}" == "" ]; then
    CURR_NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    CURR_NUM_GPUS=${NUM_GPUS}
fi

echo "NUM_GPUS" ${NUM_GPUS} "CURR_NUM_GPUS "${CURR_NUM_GPUS}

if [ $? -ne 0 ]; then
    # if failed, set NUM_GPUS to 0
    CURR_NUM_GPUS=1
fi

PROCESS_CONFIG_PYARGS+=" --num-gpus $CURR_NUM_GPUS"
PROCESS_CONFIG_PYARGS+=" --languages $LANGUAGES"
process_cfg_cmds="NUM_GPUS=${NUM_GPUS} OC_PYTHON_PATH=${OC_PYTHON_PATH} SAVE_OC_CFG_NAME=${SAVE_OC_CFG_NAME} LANGUAGES=${LANGUAGES} bash cybertron_run_scripts/process_config/process_run_config.sh  ${BASE_CONFIG} ${PROCESS_CONFIG_TYPE} ${PROCESS_CONFIG_PYARGS}"

echo "begin process config"
echo "Run command:"
echo "${process_cfg_cmds}"

eval "${process_cfg_cmds}"


# check SAVE_OC_CFG_PATH is exist
if [ -f "$SAVE_OC_CFG_PATH" ]; then
    echo "SAVE_OC_CFG_PATH exists: ${SAVE_OC_CFG_PATH}"
else
    echo "SAVE_OC_CFG_PATH does not exist: ${SAVE_OC_CFG_PATH}, please check the path"
    exit 1
fi

echo "============================== End Process Config =============================="

echo "============================== Begin Running OpenCompass =============================="

OC_RESLUT_DIR=${OC_RESLUT_ROOT}/${OC_RESULT_DIR}

run_oc_cmds="NUM_GPUS=${NUM_GPUS} OC_PYTHON_PATH=${OC_PYTHON_PATH} DEFAULT_OC_PATH=${DEFAULT_OC_PATH} LANGUAGES=${LANGUAGES} bash cybertron_run_scripts/run_opencompass/run_opencompass.sh  ${SAVE_OC_CFG_PATH} ${OC_RESLUT_DIR} ${OC_PYARGS}"

echo
echo "begin run opencompass"
echo "Run command:"
echo "${run_oc_cmds}"

eval "${run_oc_cmds}"
#echo $OC_PYTHON_PATH $DEFAULT_OC_PATH $SAVE_OC_CFG_PATH $OC_RESLUT_DIR $OC_PYARGS"----------------------------------------------------------------------------------"

echo "============================== End Running OpenCompass =============================="
