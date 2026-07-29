CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python run.py configs/mb_internal/run_configs/regression/minicpm1b_base_en.py       -w outputs/regression/minicpm1b_base_en
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python run.py configs/mb_internal/run_configs/regression/minicpm1b_base_zh.py       -w outputs/regression/minicpm1b_base_zh
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python run.py configs/mb_internal/run_configs/regression/minicpm1b_sft_longrope.py  -w outputs/regression/minicpm1b_sft_longrope
