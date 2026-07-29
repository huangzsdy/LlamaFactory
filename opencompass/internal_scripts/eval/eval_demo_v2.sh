# 接收传入的参数作为模型路径
if [ $# -gt 0 ]; then
    file_list=("$1")
else
       # 如果没有file_list，则报错
         echo "no file_list , went wrong!!!!!!!!!"

fi

LANGUAGES=$LANG

for file in "${file_list[@]}"; do
    if [ ! -f "$file/pytorch_model.bin" ]; then
        echo "pytorch_model.bin does not exist: ${file}/pytorch_model.bin"
        exit 1
    fi
done
echo "All files exist"

if [[ "$LANG" != "mn" && "$LANG" != "bo" ]]; then
  base_config="opencompass/configs/mb_internal/full_config_template/custom_mmmlu_mgsm.py"
else
  base_config="opencompass/configs/mb_internal/full_config_template/base_minicpm_core_all_v3_cmmlu_bo_mn.py" # 用于单独测试bo, mn
fi

# 动态生成oc_result_dir：从file_list[0]中提取"minicpm-1b"之前的部分，然后加上transferred_ckpt_2000
if [ ${#file_list[@]} -gt 0 ]; then
    model_path="${file_list[0]}"
    # 提取minicpm-1b之前的部分
    base_dir=$(echo "$model_path" | sed 's|/minicpm-1b/.*||')
    # 生成新的oc_result_dir
    oc_result_dir="${base_dir}/transferred_ckpt_2000"
    echo "Generated oc_result_dir: $oc_result_dir"
else
     # 如果没有file_list，则报错
     echo "no file_list , went wrong!!!!!!!!!"
fi

formatted_time=$(date "+%Y%m%d_%H%M%S")
save_folder=$formatted_time

PROCESS_CONFIG_PYARGS="--models "
for file in "${file_list[@]}"; do
    PROCESS_CONFIG_PYARGS+="${file} "
done

echo "Begin running"
set -ex

pip show datasets

OC_RESLUT_ROOT=${oc_result_dir} PROCESS_CONFIG_PYARGS=${PROCESS_CONFIG_PYARGS} OC_PYARGS="-r" LANGUAGES=${LANGUAGES} bash cybertron_run_scripts/cybertron_run/process_config_and_run.sh ${base_config} ${save_folder}

echo "End running"