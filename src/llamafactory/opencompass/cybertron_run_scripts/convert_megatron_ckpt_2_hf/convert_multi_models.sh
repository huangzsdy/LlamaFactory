CKPT_PATH=$1
SAVE_CKPT_DIR=$2
MAX_JOB_ITER=$3
ITER_EVERY=$4
CONVERT_TYPE=${CONVERT_TYPE:-"direct_path"}
MODEL_TYPE=${MODEL_TYPE:-"minicpm-1b"}
SEQ_LEN=${SEQ_LEN:-"4k"}
CKPT_JOB_ID=${CKPT_JOB_ID:-""}


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
${SUDO} chmod 777 /local/apps/opencompass

if [[ "$SAVE_CKPT_DIR" != /* ]]; then
    CURRENT_DIR=$(pwd)
    SAVE_CKPT_DIR="$CURRENT_DIR/$SAVE_CKPT_DIR"
fi

if [ ! -d "$SAVE_CKPT_DIR" ]; then
    ${SUDO} mkdir -p $SAVE_CKPT_DIR
fi

${SUDO} chmod 777 $SAVE_CKPT_DIR


# MODEL_TYPE should be minicpm3-4b or minicpm-1b
if [ "$MODEL_TYPE" != "minicpm3-4b" ] && [ "$MODEL_TYPE" != "minicpm-1b" ] && [ "$MODEL_TYPE" != "minicpm-0.5b" ]; then
    echo "MODEL_TYPE should be minicpm3-4b, minicpm-1b, or minicpm-0.5b"
    exit 1
fi

if [ ! -d "$CKPT_PATH" ]; then
    echo "CKPT_PATH do not exits, please check the path: ${CKPT_PATH}"
    exit 1
fi

echo "=============== End Prepare For Running ==============="

echo "=============== Convert HF Model Begin ==============="

# $ITER_EVERY, $ITER_EVERY * 2, ... $MAX_JOB_ITER

for CKPT_ITER in $(seq $ITER_EVERY $ITER_EVERY $MAX_JOB_ITER)
do
    echo "============================================="

    # CONVERT_TYPE should be `cybcybertron_job_path` or `direct_path`
    if [ "$CONVERT_TYPE" != "cybertron_job_path" ] && [ "$CONVERT_TYPE" != "direct_path" ]; then
        echo "CONVERT_TYPE should be `cybertron_job_path` or `direct_path`, but got ${CONVERT_TYPE}"
        exit 1
    else
        if [ "$CONVERT_TYPE" == "cybertron_job_path" ]; then
            CONVERT_PYTHON_FILE="cybertron_run_scripts/convert_megatron_ckpt_2_hf/cybertron_job_path.sh"
            if [ ! "$CKPT_JOB_ID" ]; then
                echo "CKPT_JOB_ID should be set when CONVERT_TYPE is cybertron_job_path"
                exit 1
            fi
            convert_cmds="bash ${CONVERT_PYTHON_FILE} ${MODEL_TYPE} ${SEQ_LEN} ${CKPT_PATH} ${CKPT_JOB_ID} ${CKPT_ITER} ${SAVE_CKPT_DIR}"
            SAVE_CKPT_PATH=${SAVE_CKPT_DIR}/${MODEL_TYPE}/${SEQ_LEN}/jobid_${CKPT_JOB_ID}-iter_${CKPT_ITER}
            OC_RESULT_DIR=jobid_${CKPT_JOB_ID}-iter_${CKPT_ITER}
        else
            CONVERT_PYTHON_FILE="cybertron_run_scripts/convert_megatron_ckpt_2_hf/direct_path.sh"
            convert_cmds="bash ${CONVERT_PYTHON_FILE} ${MODEL_TYPE} ${SEQ_LEN} ${CKPT_PATH} ${CKPT_ITER} ${SAVE_CKPT_DIR}"
            FOLDER_NAME=$(basename $CKPT_PATH)
            SAVE_CKPT_PATH=${SAVE_CKPT_DIR}/${MODEL_TYPE}/${SEQ_LEN}/${FOLDER_NAME}-iter_${CKPT_ITER}
            OC_RESULT_DIR=${FOLDER_NAME}-iter_${CKPT_ITER}
        fi
    fi

    if [ -f "$SAVE_CKPT_PATH/pytorch_model.bin" ]; then
        echo
        echo "${SAVE_CKPT_PATH}/pytorch_model.bin exists, skip convert"
    else
        echo
        echo "begin convert"
        echo "Run command:"
        echo "${convert_cmds}"

        eval "${convert_cmds}"

        echo "waiting for 10 seconds"
        sleep 10

        if [ ! -f "$SAVE_CKPT_PATH/pytorch_model.bin" ]; then
            echo "pytorch_model.bin does not exist: ${SAVE_CKPT_PATH}/pytorch_model.bin"
            exit 1
        fi
    fi

    echo "wait for 10 seconds"
    sleep 10
    echo "============================================="

done


echo "=============== Convert HF Model Finish ==============="
