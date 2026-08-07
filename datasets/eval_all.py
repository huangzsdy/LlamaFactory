"""
OpenCompass 自定义数据集评测配置 - 评测所有数据集

运行命令：
    cd opencompass
    python run.py --config ../datasets/eval_all.py
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
    # 代码生成 - HumanEval 格式
    dict(
        abbr='代码生成',
        path='./datasets/代码生成.jsonl',
        reader_cfg=dict(
            input_columns=['prompt'],
            output_column='canonical_solution',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='{prompt}\n',
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
            pred_role='assistant',
        ),
    ),
    # 逻辑推理
    dict(
        abbr='逻辑推理',
        path='./datasets/逻辑推理.jsonl',
        reader_cfg=dict(
            input_columns=['question'],
            output_column='target_scores',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='{question}\n',
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='PPLInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    ),
    # 数学计算
    dict(
        abbr='数学计算',
        path='./datasets/数学计算.jsonl',
        reader_cfg=dict(
            input_columns=['question'],
            output_column='answer',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='请解答以下数学问题：\n{question}\n',
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    ),
    # 语言理解 (翻译)
    dict(
        abbr='语言理解',
        path='./datasets/语言理解.jsonl',
        reader_cfg=dict(
            input_columns=['tur_Latn'],
            output_column='english',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='请将以下土耳其语翻译成英语：\n{tur_Latn}\n',
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    ),
    # 长程依赖
    dict(
        abbr='长程依赖',
        path='./datasets/长程依赖.jsonl',
        reader_cfg=dict(
            input_columns=['input'],
            output_column='answers',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='基于以下上下文回答问题：\n{context}\n\n问题：{input}\n',
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    ),
    # 知识理解 (多选题)
    dict(
        abbr='知识理解',
        path='./datasets/知识理解.jsonl',
        reader_cfg=dict(
            input_columns=['question'],
            output_column='target_scores',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='{question}\n',
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='PPLInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
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
        tokenizer_path='/home/hzs/260304/models/Qwen/Qwen2.5-7B-Instruct',
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

work_dir = './outputs/custom_eval_all'
