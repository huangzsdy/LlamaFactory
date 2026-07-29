file_list=(
#    "/data/multilingual_projects/transferred_ckpt/minicpm-1b/4k/demo-iter_release"
    "/data/multilingual_projects/transferred_ckpt/minicpm-1b/minicpm-1b/4k/minicpm-1b-iter_release/"
    #"/data/multilingual_projects/transferred_ckpt/minicpm-1b/minicpm-1b/4k/minicpm-1b-iter_release" # baseline
#    "/data/multilingual_projects/models/Qwen2.5-1.5B-Instruct/pytorch_model" #new baseline


     # note: 20251219 在1005 id=4 卡跑all语种的eval，用于跑baseline(minicpm)在mmmlu的效果，running
#    "/data/multilingual_projects/transferred_ckpt/minicpm-1b/minicpm-1b/4k/minicpm-1b-iter_release"


    # note: 20251219 在1005 id=5-7 卡跑all语种的eval，用于跑baseline(minicpm)在mmmlu的效果，running
#    "/data/multilingual_projects/transferred_ckpt/minicpm-1b/minicpm-1b/4k/minicpm-1b-iter_release"



     # note2: 20251219 在1005 id=5 卡跑fr语种的eval，用于跑minicpm + 微调数据 在mmmlu上fr的效果，running
#    "/data/multilingual_projects/training_data/20251201_190307/fr_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000" #fr

    # note2: 20251219 在1005 id=6 卡跑es语种的eval，用于跑minicpm + 微调数据 在mmmlu上es的效果，running
#    "/data/multilingual_projects/training_data/20251201_190209/es_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"

    # note2: 20251219 在1005 id=7 卡跑ar语种的eval，用于跑minicpm + 微调数据 在mmmlu上ar的效果，running
#    "/data/multilingual_projects/training_data/20251206_081658/ar_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000/"


    # note2: 20251219 在2006 id=6 卡跑ja语种的eval，用于跑minicpm + 微调数据 在mmmlu的效果，running
#    "/data/multilingual_projects/training_data/20251202_111755/ja_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"


    # note2: 20251219 在2006 id=7 卡跑it语种的eval，用于跑minicpm + 微调数据 在mmmlu的效果，running
#    "/data/multilingual_projects/training_data/20251203_192703/it_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000/"

    # note2: 20251219 在1005 id=5 卡跑de语种的eval，用于跑minicpm + 微调数据 在mmmlu上效果，running
#    "/data/multilingual_projects/training_data/20251130_142931/de_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"

    # note2: 20251219 在1005 id=6 卡跑de语种的eval，用于跑minicpm + 微调数据 在mmmlu上效果，running
#    "/data/multilingual_projects/training_data/20251202_102958/hi_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_release"

  # note2: 20251219 在1005 id=7 卡跑ko语种的eval，用于跑minicpm + 微调数据 在mmmlu上效果，running
#  "/data/multilingual_projects/training_data/20251202_112245/ko_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"

  # note2: 20251219 在2006 id=6 卡跑pt语种的eval，用于跑minicpm + 微调数据 在mmmlu上效果，running
#  "/data/multilingual_projects/training_data/20251130_142138/pt_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"

# note2: 20251219 在2006 id=6 卡跑ru语种的eval，用于跑minicpm + 微调数据 在mmmlu上效果，running
#  "/data/multilingual_projects/training_data/20251130_142240/ru_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"

# note2: 20251229 在2006 id=6 卡跑es语种的eval，用于跑minicpm + 微调数据(第一轮清洗es) 在mmmlu上效果，running
#"/data/multilingual_projects/training_data/20251228_162251/es_transferred_ckpt_2000_firstclean/minicpm-1b/4k/demo-iter_2000"

# note2: 20251229 在2006 id=6 卡跑fr语种的eval，用于跑minicpm + 微调数据(第一轮清洗fr) 在mmmlu上效果，running
#"/data/multilingual_projects/training_data/20251228_162251/fr_transferred_ckpt_2000_firstclean/minicpm-1b/4k/demo-iter_2000"


# note: 20260105 在2006 id=6 卡跑pt语种的eval，用于跑minicpm + 微调数据(第一轮清洗) 在mmmlu上效果，running
#"/data/multilingual_projects/training_data/20260103_153046/pt_transferred_ckpt_2000_firstclean/minicpm-1b/4k/demo-iter_2000"

# note: 20260130 1.4T 大模型数据ar部分跑cmmlu评测
#"/data/multilingual_projects/training_data/ar/20260128_162148/ar_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"

# note ja
#"/data/multilingual_projects/training_data/ja/20260128_161322/ja_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"

# note he
#"/data/multilingual_projects/training_data/he/20260128_161445/he_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"
)

###
#LANGUAGES="all" #或者"en,zh"多个语言
#LANGUAGES="ru" #或者"en,zh"多个语言,必须要记得改！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#LANGUAGES="pt,ar,de,es,fr,hi,it,ja,ko,ru" #或者"en,zh"多个语言,必须要记得改！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#LANGUAGES="ko" #或者"en,zh"多个语言,必须要记得改！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#LANGUAGES="es" #或者"en,zh"多个语言,必须要记得改！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#LANGUAGES="ja" #或者"en,zh"多个语言,必须要记得改！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#LANGUAGES="zh" #或者"en,zh"多个语言,必须要记得改！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#LANGUAGES="hi" #或者"en,zh"多个语言,必须要记得改！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#LANGUAGES="fr"
#LANGUAGES="ar"

for file in "${file_list[@]}"; do
    if [ ! -f "$file/pytorch_model.bin" ]; then
        echo "pytorch_model.bin does not exist: ${file}/pytorch_model.bin"
        exit 1
    fi
done

echo "All files exist"


# 默认评估方案，也可以自定义
base_config="opencompass/configs/mb_internal/full_config_template/base_minicpm_core_all_v3.py" #原始版本
#base_config="opencompass/configs/mb_internal/full_config_template/base_minicpm_core_all_v3_only_mmlu.py" # 用于单独测试mmlu
#base_config="opencompass/configs/mb_internal/full_config_template/custom_mmmlu_mgsm.py" #多语种评测

# 存储路径
#oc_result_dir="/data/multilingual_projects/training_data/20250919_114633/transferred_ckpt/"
#oc_result_dir="/data/multilingual_projects/training_data/20251026_194639/transferred_ckpt/"
#oc_result_dir="/data/multilingual_projects/training_data/20251111_132214/transferred_ckpt/"
#oc_result_dir="/data/multilingual_projects/training_data/20251114_101839/transferred_ckpt_5000"
#oc_result_dir="/data/multilingual_projects/training_data/20251117_173941/transferred_ckpt_2000/demo-iter_5000"
#oc_result_dir="/data/multilingual_projects/training_data/20251130_142931/de_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"
#oc_result_dir="/data/multilingual_projects/training_data/20251202_214356/pl_transferred_ckpt_2000/"
##
#oc_result_dir="/data/multilingual_projects/training_data/20251203_192703/it_transferred_ckpt_2000"
#oc_result_dir="/data/multilingual_projects/training_data/20251130_142240/ru_transferred_ckpt_2000" #ru
#oc_result_dir="/data/multilingual_projects/training_data/20251201_190307/fr_transferred_ckpt_2000" #fr
#oc_result_dir="/data/multilingual_projects/training_data/20251130_142138/pt_transferred_ckpt_2000" #pt
#oc_result_dir="/data/multilingual_projects/transferred_ckpt/baseline/minicpm-1b/qwen2.5_1.5b_mmmlu_mgsm_all-lang"
oc_result_dir="/data/multilingual_projects/transferred_ckpt/baseline/minicpm-1b/test_cmmlu"
#oc_result_dir="/data/multilingual_projects/transferred_ckpt/baseline/minicpm-1b/mgsm_qwen2.5_1.5B_251218"
#oc_result_dir="/data/multilingual_projects/transferred_ckpt/baseline/minicpm-1b/mgsm_minicpm_251218"
#oc_result_dir="/data/multilingual_projects/training_data/20251203_192703/it_transferred_ckpt_2000/qwen2.5_1.5b_res"

# note: 20251219 在1005 id=4 卡跑all语种的eval，用于跑baseline(minicpm)在mmmlu的效果，running
#oc_result_dir="/data/multilingual_projects/transferred_ckpt/baseline/minicpm-1b/all_lang_mmmlu_mgsm"

# note: 20251219 在1005 id=5-7 卡跑all语种的eval，用于跑baseline(minicpm)在mmmlu的效果，running
#oc_result_dir="/data/multilingual_projects/transferred_ckpt/baseline/minicpm-1b/all_lang_mmmlu_mgsm_567"

# note: 20251219 在1005 id=5 卡跑fr语种的eval，用于跑minicpm + 微调数据 在mmmlu上fr的效果，
#oc_result_dir="/data/multilingual_projects/training_data/20251201_190307/fr_transferred_ckpt_2000" #fr

# note: 20251219 在1005 id=6 卡跑es语种的eval，用于跑minicpm + 微调数据 在mmmlu上es的效果，
#oc_result_dir="/data/multilingual_projects/training_data/20251201_190209/es_transferred_ckpt_2000" #es

# note: 20251219 在1005 id=7 卡跑ar语种的eval，用于跑minicpm + 微调数据 在mmmlu上ar的效果，
#oc_result_dir="/data/multilingual_projects/training_data/20251206_081658/ar_transferred_ckpt_2000"

# note: 20251219 在2006 id=6 卡跑ja语种的eval，用于跑minicpm + 微调数据 在mmmlu的效果，
#oc_result_dir="/data/multilingual_projects/training_data/20251202_111755/ja_transferred_ckpt_2000"

# note: 20251219 在2006 id=7 卡跑it语种的eval，用于跑minicpm + 微调数据 在mmmlu的效果，
#oc_result_dir="/data/multilingual_projects/training_data/20251203_192703/it_transferred_ckpt_2000"

# note: 20251219 在1005 id=5 卡跑de语种的eval，用于跑minicpm + 微调数据 在mmmlu上效果，
#oc_result_dir="/data/multilingual_projects/training_data/20251130_142931/de_transferred_ckpt_2000"

# note: 20251219 在1005 id=6 卡跑hi语种的eval，用于跑minicpm + 微调数据 在mmmlu上效果，
#oc_result_dir="/data/multilingual_projects/training_data/20251202_102958/hi_transferred_ckpt_2000"

# note: 20251219 在1005 id=7 卡跑ko语种的eval，用于跑minicpm + 微调数据 在mmmlu上效果，
#oc_result_dir="/data/multilingual_projects/training_data/20251202_112245/ko_transferred_ckpt_2000"

# note: 20251219 在2006 id=6 卡跑pt语种的eval，用于跑minicpm + 微调数据 在mmmlu上效果，
#oc_result_dir="/data/multilingual_projects/training_data/20251130_142138/pt_transferred_ckpt_2000"

# note: 20251219 在2006 id=6 卡跑ru语种的eval，用于跑minicpm + 微调数据 在mmmlu上效果，
#oc_result_dir="/data/multilingual_projects/training_data/20251130_142240/ru_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000"

# note: 20251229 在2006 id=6 卡跑es语种的eval，用于跑minicpm + 微调数据(第一轮清洗es) 在mmmlu上效果，running
#oc_result_dir="/data/multilingual_projects/training_data/20251228_162251/es_transferred_ckpt_2000_firstclean/"

# note: 20251229 在2006 id=6 卡跑fr语种的eval，用于跑minicpm + 微调数据(第一轮清洗fr) 在mmmlu上效果，running
#oc_result_dir="/data/multilingual_projects/training_data/20251228_162251/fr_transferred_ckpt_2000_firstclean"



# note: 20251229 在2006 id=6 卡跑pt语种的eval，用于跑minicpm + 微调数据(第一轮清洗fr) 在mmmlu上效果，running
#oc_result_dir="/data/multilingual_projects/training_data/ar/20260128_162148/ar_transferred_ckpt_2000"

#oc_result_dir="/data/multilingual_projects/training_data/ja/20260128_161322/ja_transferred_ckpt_2000"

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
