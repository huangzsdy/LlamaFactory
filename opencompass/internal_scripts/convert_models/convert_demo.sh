# Some important parameters, do not need to change
CONVERT_PYTHON_FILE="cybertron_run_scripts/convert_megatron_ckpt_2_hf/direct_path.sh"
MODEL_TYPE="minicpm-1b"
SEQ_LEN='4k'
#SAVE_CKPT_DIR="/user/zhangyixuan/exps/2025Q3/ultrafineweb_exp"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20250919_114633/transferred_ckpt"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251009_121817/transferred_ckpt"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251026_194639/transferred_ckpt"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251111_132214/transferred_ckpt"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251114_101839/transferred_ckpt"
#AVE_CKPT_DIR="/data/multilingual_projects/training_data/20251114_101839/transferred_ckpt_5000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251114_101839/transferred_ckpt_4000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251117_172401/transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251130_142138/pt_transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251130_142240/ru_transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251201_190307/fr_transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251201_190209/es_transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251202_111755/ja_transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/transferred_ckpt/minicpm-1b"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251202_214356/pl_transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251202_214330/nl_transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251130_142240/ru_transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/transferred_ckpt/minicpm-1b_251210"
#note es 将大模型团队提交的es数据清洗round=1
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251228_162251/es_transferred_ckpt_2000_firstclean"
#note fr 将大模型团队提交的fr数据清洗round=1
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251228_162251/fr_transferred_ckpt_2000_firstclean"
#note pt 将大模型团队提交的pt数据清洗round=1
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20260103_153046/pt_transferred_ckpt_2000_firstclean"


#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/ja/20260128_161322/ja_transferred_ckpt_2000" #note ja
SAVE_CKPT_DIR="/data/multilingual_projects/training_data/he/20260128_161445/he_transferred_ckpt_2000" #note he

#note ar 20260131 ar 用xlm_berta分类器过滤所有1.4T ar数据，并进行微调评测
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/ar/20260128_162148/ar_transferred_ckpt_2000"

#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251201_190307/fr_transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251201_190209/es_transferred_ckpt_2000"
#SAVE_CKPT_DIR="/data/multilingual_projects/training_data/20251203_192703/it_transferred_ckpt_2000"
# The path of the model to be converted
path_list=(
#    "/projects/118-hqdata-exp/56734/checkpoints/fineweb_edu_weight_exp_0.05"
#	"/data/multilingual_projects/Megatron_1B/train_models/checkpoints/demo"
#     "/data/multilingual_projects/training_data/20250919_114633/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251009_121817/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251026_194639/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251111_132214/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251114_101839/checkpoints/demo/iter_0005000"
#"/data/multilingual_projects/training_data/20251114_101839/checkpoints/demo/iter_0004000/"
#"/data/multilingual_projects/training_data/20251117_172401/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251130_142138/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251130_142240/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251201_190307/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251201_190209/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251202_111755/checkpoints/demo"
#"/home/jeeves/ultrafineweb_exps/opencompass/minicpm-1b"
#"/data/multilingual_projects/training_data/20251202_214356/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251202_214330/checkpoints/demo" #nl

#"/data/multilingual_projects/training_data/20251201_190307/checkpoints/demo" #fr
#"/data/multilingual_projects/training_data/20251201_190209/checkpoints/demo" #es
#"/data/multilingual_projects/training_data/20251203_192703/checkpoints/demo" #it
#"/data/multilingual_projects/training_data/20251130_142240/checkpoints/demo"
#"/data/multilingual_projects/training_data/20251227_175735/checkpoints/demo" #note es 将大模型团队提交的es数据清洗round=1
#"/data/multilingual_projects/training_data/20251228_162251/checkpoints/demo" #note fr 将大模型团队提交的fr数据清洗round=1
#"/data/multilingual_projects/training_data/20260103_153046/checkpoints/demo" #note pt 将大模型团队提交的pt数据清洗round=1

#"/data/multilingual_projects/training_data/ja/20260128_161322/checkpoints/demo" #note ja 1.4T ja 部分全量过滤后跑评测
#"/data/multilingual_projects/training_data/ar/20260128_162148/checkpoints/demo" #note ar 1.4T ar 部分全量过滤后跑评测
"/data/multilingual_projects/training_data/he/20260128_161445/checkpoints/demo" #note ar 1.4T he 部分全量过滤后跑评测
)

#CKPT_ITER="release"
CKPT_ITER="2000"

for CKPT_PATH in "${path_list[@]}"; do
    echo "Begin processing: $CKPT_PATH"
    if [ ! -d "$CKPT_PATH" ]; then
        echo "Path does not exist"
        exit 1
    fi
    bash ${CONVERT_PYTHON_FILE} ${MODEL_TYPE} ${SEQ_LEN} ${CKPT_PATH} ${CKPT_ITER} ${SAVE_CKPT_DIR}

    FOLDER_NAME=$(basename $CKPT_PATH)
    SAVE_CKPT_PATH=${SAVE_CKPT_DIR}/${MODEL_TYPE}/${SEQ_LEN}/${FOLDER_NAME}-iter_${CKPT_ITER}
    echo "SAVE_CKPT_PATH: $SAVE_CKPT_PATH !!!!!!!!!"
    echo "waiting for 5 seconds"
    sleep 5

    if [ ! -f "$SAVE_CKPT_PATH/pytorch_model.bin" ]; then
        echo "pytorch_model.bin does not exist: ${SAVE_CKPT_PATH}/pytorch_model.bin"
        exit 1
    fi
done
