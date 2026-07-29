#!/bin/bash
MODEL_TYPE=$1
SEQ_LEN=$2
CKPT_JOB_DIR=$3
CKPT_JOB_ID=$4
CKPT_ITER=$5
SAVE_CKPT_DIR=$6

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
fi

if [[ "$SAVE_CKPT_DIR" != /* ]]; then
    CURRENT_DIR=$(pwd)
    SAVE_CKPT_DIR="$CURRENT_DIR/$SAVE_CKPT_DIR"
fi

if [ ! -d "$SAVE_CKPT_DIR" ]; then
    ${SUDO} mkdir -p $SAVE_CKPT_DIR
fi
${SUDO} chmod 777 $SAVE_CKPT_DIR

echo "=============== End Prepare For Running ==============="

echo "=============== Begin Check Arguments ==============="

MEGATRON_PATH="third_part/Megatron-LM"

if [ "$SEQ_LEN" != "4k" ] && [ "$SEQ_LEN" != "32k" ]; then
    echo "SEQ_LEN should be 4k or 32k"
    exit 1
fi

# Check MODEL_TYPE and set COVERT_FILE
if [ "$MODEL_TYPE" != "minicpm3-4b" ] && [ "$MODEL_TYPE" != "minicpm-1b" ] && [ "$MODEL_TYPE" != "minicpm-0.5b" ]; then
    echo "MODEL_TYPE should be minicpm3-4b, minicpm-1b, or minicpm-0.5b"
    exit 1
else
    if [ "$MODEL_TYPE" == "minicpm-1b" ]; then
        if [ "$SEQ_LEN" == "4k" ]; then
            COVERT_FILE="examples/convert_ckpt_to_hf/minicpm_1b_4k.sh"
            MODEL_CONFIG_PATH="examples/convert_ckpt_to_hf/minicpm_1b/4k"
        else
            COVERT_FILE="examples/convert_ckpt_to_hf/minicpm_1b_32k.sh"
            MODEL_CONFIG_PATH="examples/convert_ckpt_to_hf/minicpm_1b/32k"
        fi
        echo "MODEL_TYPE       : minicpm-1b"
        echo "COVERT_FILE      : ${COVERT_FILE}"
        echo "MODEL_CONFIG_PATH: ${MODEL_CONFIG_PATH}"
    fi
    if [ "$MODEL_TYPE" == "minicpm3-4b" ]; then
        if [ "$SEQ_LEN" == "4k" ]; then
            COVERT_FILE="examples/convert_ckpt_to_hf/minicpm3_4b_4k.sh"
            MODEL_CONFIG_PATH="examples/convert_ckpt_to_hf/minicpm3_4b/4k"
        else
            # COVERT_FILE="examples/convert_ckpt_to_hf/minicpm3_4b_32k.sh"
            # MODEL_CONFIG_PATH="examples/convert_ckpt_to_hf/minicpm3_4b/32k"
            echo "NotImplementedError: minicpm3-4b-32k is not implemented"
            exit 1
        fi
        echo "MODEL_TYPE       : minicpm3-4b"
        echo "COVERT_FILE      : ${COVERT_FILE}"
        echo "MODEL_CONFIG_PATH: ${MODEL_CONFIG_PATH}"
    fi
    if [ "$MODEL_TYPE" == "minicpm-0.5b" ]; then
        if [ "$SEQ_LEN" == "4k" ]; then
            COVERT_FILE="examples/convert_ckpt_to_hf/minicpm_0.5b_4k.sh"
            MODEL_CONFIG_PATH="examples/convert_ckpt_to_hf/minicpm_0.5b/4k"
        else
            COVERT_FILE="examples/convert_ckpt_to_hf/minicpm_0.5b_32k.sh"
            MODEL_CONFIG_PATH="examples/convert_ckpt_to_hf/minicpm_0.5b/32k"
        fi
        echo "MODEL_TYPE       : minicpm-0.5b"
        echo "COVERT_FILE      : ${COVERT_FILE}"
        echo "MODEL_CONFIG_PATH: ${MODEL_CONFIG_PATH}"
    fi
fi

# check CKPT_PATH is exist
# Cybertron's saving path is: /projects/{PROJECT_ID}-{PROJECT_NAME}/checkpoints/checkpoints.{CKPT_JOB_ID}
# CKPT_PATH=${CKPT_JOB_DIR}/checkpoints.${CKPT_JOB_ID}

# NOTE: New saving path is /projects/{PROJECT_ID}-{PROJECT_NAME}/{CKPT_JOB_ID}/checkpoints
CKPT_PATH=${CKPT_JOB_DIR}/{CKPT_JOB_ID}/checkpoints
if [ -d "$CKPT_PATH" ]; then
    echo "CKPT_PATH exists: ${CKPT_PATH}"
else
    echo "CKPT_PATH does not exist: ${CKPT_PATH}, please check the path"
    exit 1
fi

# CKPT_ITER should be a number or "release" or "latest"
if [ "$CKPT_ITER" != "release" ] && [ "$CKPT_ITER" != "latest" ]; then
    if ! [[ "$CKPT_ITER" =~ ^[0-9]+$ ]]; then
        echo "CKPT_ITER should be a number or 'release' or 'latest'"
        exit 1
    fi
fi

echo "=============== End Check Arguments ==============="

echo "=============== Begin Convert HF Model ==============="
pip install /user/zhangyixuan/modelbest_sdk/modelbest_sdk-0.2.5.7-py3-none-any.whl

SAVE_CKPT_PATH=${SAVE_CKPT_DIR}/${MODEL_TYPE}/${SEQ_LEN}/jobid_${CKPT_JOB_ID}-iter_${CKPT_ITER}
echo "Load from: ${CKPT_PATH}"
echo "Iteration: ${CKPT_ITER}"
echo "Save at  : ${SAVE_CKPT_PATH}"

if [ -f "$SAVE_CKPT_PATH/pytorch_model.bin" ]; then
    echo
    echo "${SAVE_CKPT_PATH}/pytorch_model.bin exists, skip convert"
else
    echo
    if [ -d "$SAVE_CKPT_PATH" ]; then
        echo "${SAVE_CKPT_PATH} exists"
    else
        echo "${SAVE_CKPT_PATH} does not exist, create it"
        ${SUDO} mkdir -p ${SAVE_CKPT_PATH}
    fi
    # give write permission
    ${SUDO} chmod 777 -R ${SAVE_CKPT_PATH}
    ${SUDO} chmod 777 -R ${MEGATRON_PATH}
    # Megtron ckpt to hf
    cd ${MEGATRON_PATH}
    echo "Copy ${MODEL_CONFIG_PATH} to ${SAVE_CKPT_PATH}"
    cp -r ${MODEL_CONFIG_PATH}/* ${SAVE_CKPT_PATH}/.

    echo "begin convert"
    echo "Run command:"

    if [[ "$CKPT_ITER" =~ ^[0-9]+$ ]]; then
        echo "CUDA_VISIBLE_DEVICES=\"0\" bash ${COVERT_FILE} --load ${CKPT_PATH}/ --ckpt-step ${CKPT_ITER} --save ${SAVE_CKPT_PATH}"
        CUDA_VISIBLE_DEVICES="0" bash ${COVERT_FILE} --load ${CKPT_PATH}/ --ckpt-step ${CKPT_ITER} --save ${SAVE_CKPT_PATH}
    else
        echo "CUDA_VISIBLE_DEVICES=\"0\" bash ${COVERT_FILE} --load ${CKPT_PATH}/ --save ${SAVE_CKPT_PATH}"
        CUDA_VISIBLE_DEVICES="0" bash ${COVERT_FILE} --load ${CKPT_PATH}/ --save ${SAVE_CKPT_PATH}
    fi
    cd ../..
fi

echo "=============== End Convert HF Model ==============="
