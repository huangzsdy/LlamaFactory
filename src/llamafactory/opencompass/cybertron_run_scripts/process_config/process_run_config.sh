#!/bin/bash
BASE_CONFIG=$1
CFG_TYPE=$2
OC_PYTHON_PATH=${OC_PYTHON_PATH:-"miniconda3/envs/opencompass/bin/python"}
SAVE_OC_CFG_NAME=${SAVE_OC_CFG_NAME:-"curr_config.py"}
LANGUAGES=${LANGUAGES:-"all"}
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
else
    ${SUDO} chmod 777 -R ./configs
fi

echo "=============== End Prepare For Running ==============="

echo "=============== Begin Check Arguments ==============="

if [ ! -f "$BASE_CONFIG" ]; then
    echo "BASE_CONFIG is not exist, please check the path: ${BASE_CONFIG}"
    exit 1
fi

if [ ! -f "$OC_PYTHON_PATH" ]; then
    echo "OC_PYTHON_PATH is not exist, please check the path: ${OC_PYTHON_PATH}"
    exit 1
fi

if [ "$CFG_TYPE" != "diy" ] && [ "$CFG_TYPE" != "full" ]; then
    echo "CFG_TYPE should be `diy` or `full`, but got ${CFG_TYPE}"
    exit 1
else
    if [ "$CFG_TYPE" == "diy" ]; then
        PYTHON_FILE="tools/internal/process_diy_run_config.py"
    else
        PYTHON_FILE="tools/internal/process_full_run_config.py"
    fi
fi

echo "BASE_CONFIG:      $BASE_CONFIG"
echo "CFG_TYPE:         $CFG_TYPE"
echo "OC_PYTHON_PATH:   $OC_PYTHON_PATH"
echo "SAVE_OC_CFG_NAME: $SAVE_OC_CFG_NAME"
echo "PY_ARGS:          $PY_ARGS"

echo "=============== End Check Arguments ==============="

echo "=============== Begin Running ==============="

run_cmds="PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH "
run_cmds+="${OC_PYTHON_PATH} ${PYTHON_FILE} ${BASE_CONFIG} --save-cfg-name ${SAVE_OC_CFG_NAME} ${PY_ARGS}"

echo
echo "Run command:"
echo ${run_cmds}

eval "${run_cmds}"

echo "=============== End Running ==============="
