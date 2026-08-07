"""
OpenCompass 评测配置文件：逻辑推理、知识理解、长程依赖、代码生成

使用方法：
    cd opencompass
    python run.py --config datasets/eval_all.py
"""

# 数据集根目录
data_root = './datasets'

# ============================================================
# 1. 逻辑推理
# ============================================================
逻辑推理_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
)
逻辑推理_infer_cfg = dict(
    prompt_template=dict(
        type='PromptTemplate',
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Evaluate the following Boolean expression and give the answer (True or False):\n{question}'),
            ]
        )
    ),
    retriever=dict(type='ZeroRetriever'),
    inferencer=dict(type='GenInferencer'),
    model_max_length=2048,
)
逻辑推理_eval_cfg = dict(
    evaluator=dict(type='AccEvaluator'),
    pred_role='BOT',
)

# ============================================================
# 2. 知识理解
# ============================================================
知识理解_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
)
知识理解_infer_cfg = dict(
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
    model_max_length=2048,
)
知识理解_eval_cfg = dict(
    evaluator=dict(type='AccEvaluator'),
    pred_role='BOT',
)

# ============================================================
# 3. 长程依赖
# ============================================================
长程依赖_reader_cfg = dict(
    input_columns=['input', 'context'],
    output_column='answers',
)
长程依赖_infer_cfg = dict(
    prompt_template=dict(
        type='PromptTemplate',
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Context: {context}\n\nQuestion: {input}\n\nProvide a concise answer based on the context above.'),
            ]
        )
    ),
    retriever=dict(type='ZeroRetriever'),
    inferencer=dict(type='GenInferencer'),
    model_max_length=8192,
)
长程依赖_eval_cfg = dict(
    evaluator=dict(type='AccEvaluator'),
    pred_role='BOT',
)

# ============================================================
# 4. 代码生成
# ============================================================
代码生成_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='test',
)
代码生成_infer_cfg = dict(
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
    model_max_length=2048,
)
代码生成_eval_cfg = dict(
    evaluator=dict(type='HumanEvalEvaluator', timeout=30),
    pred_role='BOT',
)

# ============================================================
# 数据集列表
# ============================================================
datasets = [
    # 1. 逻辑推理
    dict(
        type='CustomDataset',
        path=f'{data_root}/逻辑推理.jsonl',
        name='逻辑推理',
        abbr='logic_reasoning',
        reader_cfg=逻辑推理_reader_cfg,
        infer_cfg=逻辑推理_infer_cfg,
        eval_cfg=逻辑推理_eval_cfg,
    ),
    # 2. 知识理解
    dict(
        type='CustomDataset',
        path=f'{data_root}/知识理解.jsonl',
        name='知识理解',
        abbr='knowledge_understanding',
        reader_cfg=知识理解_reader_cfg,
        infer_cfg=知识理解_infer_cfg,
        eval_cfg=知识理解_eval_cfg,
    ),
    # 3. 长程依赖
    dict(
        type='CustomDataset',
        path=f'{data_root}/长程依赖.jsonl',
        name='长程依赖',
        abbr='long_context',
        reader_cfg=长程依赖_reader_cfg,
        infer_cfg=长程依赖_infer_cfg,
        eval_cfg=长程依赖_eval_cfg,
    ),
    # 4. 代码生成
    dict(
        type='CustomDataset',
        path=f'{data_root}/代码生成.jsonl',
        name='代码生成',
        abbr='code_generation',
        reader_cfg=代码生成_reader_cfg,
        infer_cfg=代码生成_infer_cfg,
        eval_cfg=代码生成_eval_cfg,
    ),
]

# ============================================================
# 模型配置
# ============================================================
models = [
    dict(
        type='HuggingFaceCausalLM',
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
        max_seq_len=8192,
        max_out_len=512,
        batch_size=4,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

# ============================================================
# 推理配置
# ============================================================
infer_cfg = dict(
    runner=dict(
        type='LocalRunner',
        max_num_workers=1,
        task=dict(
            type='OpenICLInferTask',
            max_out_len=512,
        ),
    ),
    partitioner=dict(
        type='NaivePartitioner',
    ),
)

# ============================================================
# 评估配置
# ============================================================
eval_cfg = dict(
    runner=dict(
        type='LocalRunner',
        max_num_workers=1,
        task=dict(
            type='OpenICLEvalTask',
        ),
    ),
    partitioner=dict(
        type='NaivePartitioner',
    ),
)

# ============================================================
# 输出配置
# ============================================================
work_dir = './outputs/eval_all'

# ============================================================
# 其他配置
# ============================================================
# 用于设置进程标题
languages = 'en'

# summarizer 配置
summarizer = dict(
    type='DefaultSummarizer',
)
