"""测试配置文件"""
from mmengine.config import read_base

# 简单数据集配置
datasets = [
    dict(
        abbr='test_mmlu',
        path='./datasets/test.jsonl',
        reader_cfg=dict(
            input_columns=['question'],
            output_column='answer',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='{question}',
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='PPLInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    ),
]

models = [
    dict(
        type='HuggingFace',
        path='Qwen/Qwen2.5-7B-Instruct',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_seq_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

work_dir = './outputs/test'
