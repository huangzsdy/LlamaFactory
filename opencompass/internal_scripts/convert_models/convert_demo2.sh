# Some important parameters, do not need to change
CONVERT_PYTHON_FILE="cybertron_run_scripts/convert_megatron_ckpt_2_hf/direct_path.sh"
MODEL_TYPE="minicpm-1b"
SEQ_LEN='4k'
CKPT_ITER="2000"

# 定义语种参数
#LANG="bo"
#LANG="es"
#LANG="de"
#LANG="nl"
#LANG="ko"
#LANG="zh"
#LANG="ja"
#LANG="he"
#LANG="mn"
#LANG="multi_lang"

# 根据语种自动设置path_list
path_list=(
#    "/data/multilingual_projects/training_data/${LANG}/20260128_161445/checkpoints/demo" he
#    "/data/multilingual_projects/training_data/bo/20260206_173551/checkpoints/demo"
#    "/data/multilingual_projects/training_data/es/20260206_175017/checkpoints/demo"
#    "/data/multilingual_projects/training_data/de/20260206_173737/checkpoints/demo/"
#    "/data/multilingual_projects/training_data/nl/20260206_173803/checkpoints/demo/"
#"/data/multilingual_projects/training_data/mn/20260206_135700/checkpoints/demo"
#"/data/multilingual_projects/training_data/hi/20260206_175323/checkpoints/demo"
#  "/data/multilingual_projects/training_data/hi/20260206_175323/hi_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"
#"/data/multilingual_projects/training_data/zh/20260302_132331/checkpoints/demo"
#"/data/multilingual_projects/training_data/ru/20260319_164810/checkpoints/demo"
#"/data/multilingual_projects/training_data/ja/20260319_172410/checkpoints/demo"
#"/data/multilingual_projects/training_data/de/20260320_093848/checkpoints/demo"
#"/data/multilingual_projects/training_data/fr/20260319_172349/checkpoints/demo"

# 0.75
#"/data/multilingual_projects/training_data/ja/20260323_101656/checkpoints/demo"
# 0.8
#"/data/multilingual_projects/training_data/ja/20260323_101732/checkpoints/demo"
# 0.85
#"/data/multilingual_projects/training_data/ja/20260323_101807/checkpoints/demo"

#ja qa
#"/data/multilingual_projects/training_data/ja_qa/20260428_164634/checkpoints/demo"
#"/data/multilingual_projects/training_data/ja_qa/20260505_201933/checkpoints/demo"

#ja wiki
#"/data/multilingual_projects/training_data/ja_wiki/20260428_164903/checkpoints/demo"
#"/data/multilingual_projects/training_data/ja_wiki/20260505_202015/checkpoints/demo"



# he qa
#"/data/multilingual_projects/training_data/he_qa/20260430_200218/checkpoints/demo"

  #"/data/multilingual_projects/training_data/ko/20260206_173659/checkpoints/demo/"
  #/data/multilingual_projects/training_data/ko/20260206_173659/ko_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000

)

# 自动生成SAVE_CKPT_DIR：从path_list[0]提取基础路径，替换checkpoints之后的部分
CKPT_PATH="${path_list[0]}"
# 提取到checkpoints之前的路径部分
BASE_PATH=$(echo "$CKPT_PATH" | sed 's|/checkpoints/.*||')
# 构建新的SAVE_CKPT_DIR
SAVE_CKPT_DIR="${BASE_PATH}/${LANG}_transferred_ckpt_${CKPT_ITER}"

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