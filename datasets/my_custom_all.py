"""
OpenCompass 自定义数据集评测配置

运行命令：
    cd opencompass
    python run.py --config datasets/my_custom_all.py
"""

# ============================================================
# 数据集配置
# ============================================================

# Military MCQ 数据集
from opencompass.configs.datasets.military_mcq.military_mcq_ppl import (
    military_mcq_datasets as military_mcq_ds,
    military_mcq_reader_cfg,
    military_mcq_infer_cfg,
    military_mcq_eval_cfg,
)

# Abuse QA 数据集
from opencompass.configs.datasets.abuse_qa.abuse_qa_ppl import (
    abuse_qa_datasets as abuse_qa_ds,
    abuse_qa_reader_cfg,
    abuse_qa_infer_cfg,
    abuse_qa_eval_cfg,
)

# 数据集列表
datasets = [
    # Military MCQ
    dict(
        abbr='military_mcq',
        type='MilitaryMCQDataset',
        path='./datasets/military_mcq.xlsx',
        reader_cfg=military_mcq_reader_cfg,
        infer_cfg=military_mcq_infer_cfg,
        eval_cfg=military_mcq_eval_cfg,
    ),
    # Abuse QA
    dict(
        abbr='abuse_qa',
        type='AbuseQADataset',
        path='./datasets/abuse_qa.xlsx',
        reader_cfg=abuse_qa_reader_cfg,
        infer_cfg=abuse_qa_infer_cfg,
        eval_cfg=abuse_qa_eval_cfg,
    ),
]

# ============================================================
# 模型配置
# ============================================================

models = [
    dict(
        type='HuggingFace',
        path='/home/hzs/260304/models/Qwen/Qwen2.5-7B-Instruct',  # 本地模型路径
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_path='/home/hzs/260304/models/Qwen/Qwen2.5-7B-Instruct',  # 本地模型路径
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_seq_len=4096,
        max_out_len=512,
        batch_size=4,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

# ============================================================
# 输出配置
# ============================================================

work_dir = './outputs/custom_eval'
