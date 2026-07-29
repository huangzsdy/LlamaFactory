# regression tasks datasets defaults to MMLU, CMMLU, ARC-c, GSM8k, sanitized-MBPP, and HumanEval.

RUN_SCRIPT_PATH="cybertron_run_scripts/cybertron_run/process_config_and_run.sh"

# 1. check base model full config setting
BASE_CONFIG="configs/mb_internal/full_config_template/base_minicpm_regression.py"
OC_RESULT_DIR="opencompass_regression_results/base_full"
PROCESS_CONFIG_TYPE="full"
PROCESS_CONFIG_PYARGS="\"--models /user/zhangyixuan/models/opencompass_regression/jobid_2542-iter_5000\""

base_full_run_cmds=""
base_full_run_cmds+="PROCESS_CONFIG_TYPE=${PROCESS_CONFIG_TYPE} "
base_full_run_cmds+="PROCESS_CONFIG_PYARGS=${PROCESS_CONFIG_PYARGS} "
base_full_run_cmds+="bash ${RUN_SCRIPT_PATH} ${BASE_CONFIG} ${OC_RESULT_DIR}"

echo ${base_full_run_cmds}
eval ${base_full_run_cmds}

sleep 5

# 2. check chat model diy config setting
BASE_CONFIG="configs/mb_internal/diy_config_template/chat_minicpm_model.py"
OC_RESULT_DIR="opencompass_regression_results/chat_diy"
PROCESS_CONFIG_TYPE="diy"
PROCESS_CONFIG_PYARGS="\"--models /user/zhangyixuan/models/opencompass_regression/jobid_6860-iter_8330 --model-type vllm --datasets configs/mb_internal/dataset_collections/chat_regression.py --dataset-check\" "


chat_diy_run_cmds=""
chat_diy_run_cmds+="PROCESS_CONFIG_TYPE=${PROCESS_CONFIG_TYPE} "
chat_diy_run_cmds+="PROCESS_CONFIG_PYARGS=${PROCESS_CONFIG_PYARGS} "
chat_diy_run_cmds+="bash ${RUN_SCRIPT_PATH} ${BASE_CONFIG} ${OC_RESULT_DIR}"

echo ${chat_diy_run_cmds}
eval ${chat_diy_run_cmds}
