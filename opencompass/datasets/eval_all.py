"""
OpenCompass 自定义数据集评测配置 - 评测所有数据集

运行命令：
    cd opencompass
    python run.py datasets/eval_all.py
"""

import os

# 数据集目录
datasets_dir = './datasets'

# ============================================================
# 数据集配置 - 手动添加已有配置文件的数据集
# ============================================================

# 知识理解
knowledge_understanding = dict(
    abbr='knowledge_understanding',
    path='./datasets/知识理解.jsonl',
    reader_cfg=dict(
        input_columns=['question'],
        output_column='answer',
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type='PromptTemplate',
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer with the correct option letter (e.g., A, B, C, or D).'),
                ]
            )
        ),
        retriever=dict(type='ZeroRetriever'),
        inferencer=dict(type='GenInferencer'),
    ),
    eval_cfg=dict(
        evaluator=dict(type='AccEvaluator'),
    ),
)

# 逻辑推理
logical_reasoning = dict(
    abbr='logic_reasoning',
    path='./datasets/逻辑推理.jsonl',
    reader_cfg=dict(
        input_columns=['question'],
        output_column='answer',
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type='PromptTemplate',
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer:'),
                ]
            )
        ),
        retriever=dict(type='ZeroRetriever'),
        inferencer=dict(type='GenInferencer'),
    ),
    eval_cfg=dict(
        evaluator=dict(type='AccEvaluator'),
    ),
)

# 长程依赖
long_context = dict(
    abbr='long_context',
    path='./datasets/长程依赖.jsonl',
    reader_cfg=dict(
        input_columns=['input', 'context'],
        output_column='answer',
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type='PromptTemplate',
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='Context: {context}\nQuestion: {input}\nAnswer:'),
                ]
            )
        ),
        retriever=dict(type='ZeroRetriever'),
        inferencer=dict(type='GenInferencer'),
    ),
    eval_cfg=dict(
        evaluator=dict(type='AccEvaluator'),
    ),
)

# 代码生成
code_generation = dict(
    abbr='code_generation',
    path='./datasets/代码生成.jsonl',
    reader_cfg=dict(
        input_columns=['prompt'],
        output_column='canonical_solution',
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type='PromptTemplate',
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='{prompt}'),
                ]
            )
        ),
        retriever=dict(type='ZeroRetriever'),
        inferencer=dict(type='GenInferencer'),
    ),
    eval_cfg=dict(
        evaluator=dict(type='AccEvaluator'),
    ),
)

# JS通用知识理解
js_knowledge = dict(
    abbr='js_knowledge',
    path='./datasets/JS通用知识理解.jsonl',
    reader_cfg=dict(
        input_columns=['question'],
        output_column='answer',
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type='PromptTemplate',
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer with the correct option letter (e.g., A, B, C, or D).'),
                ]
            )
        ),
        retriever=dict(type='ZeroRetriever'),
        inferencer=dict(type='GenInferencer'),
    ),
    eval_cfg=dict(
        evaluator=dict(type='AccEvaluator'),
    ),
)

# abuse_qa
abuse_qa = dict(
    abbr='abuse_qa',
    path='./datasets/abuse_qa.jsonl',
    reader_cfg=dict(
        input_columns=['question'],
        output_column='answer',
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type='PromptTemplate',
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer with the correct option letter (e.g., A, B, C, or D).'),
                ]
            )
        ),
        retriever=dict(type='ZeroRetriever'),
        inferencer=dict(type='GenInferencer'),
    ),
    eval_cfg=dict(
        evaluator=dict(type='AccEvaluator'),
    ),
)

# military_mcq
military_mcq = dict(
    abbr='military_mcq',
    path='./datasets/military_mcq.jsonl',
    reader_cfg=dict(
        input_columns=['question'],
        output_column='answer',
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type='PromptTemplate',
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer with the correct option letter (e.g., A, B, C, or D).'),
                ]
            )
        ),
        retriever=dict(type='ZeroRetriever'),
        inferencer=dict(type='GenInferencer'),
    ),
    eval_cfg=dict(
        evaluator=dict(type='AccEvaluator'),
    ),
)

# 数据集列表
datasets = [
    knowledge_understanding,
    logical_reasoning,
    long_context,
    code_generation,
    js_knowledge,
    abuse_qa,
    military_mcq,
]

# ============================================================
# 模型配置
# ============================================================

models = [
    dict(
        type='HuggingFace',
        path='/mnt/c/Users/ThinkPad/Downloads/copy_models/Qwen/Qwen2.5-7B-Instruct',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
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
