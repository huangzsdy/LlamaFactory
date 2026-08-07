"""长程依赖数据集评测配置"""

from opencompass.datasets.custom import CustomDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class 长程依赖Dataset(CustomDataset):
    """长程依赖数据集：基于长上下文的问答"""
    pass


def get_long_context_dataset():
    """获取长程依赖数据集配置"""
    reader_cfg = dict(
        input_columns=['input', 'context'],
        output_column='answers',
    )

    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
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

    eval_cfg = dict(
        evaluator=dict(type='AccEvaluator'),
        pred_role='BOT',
    )

    dataset = dict(
        abbr='long_context',
        type='长程依赖Dataset',
        path='./datasets/长程依赖.jsonl',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )

    return dataset
