"""代码生成数据集评测配置"""

from opencompass.datasets.custom import CustomDataset
from opencompass.datasets.humaneval import HumanEvalEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class 代码生成Dataset(CustomDataset):
    """代码生成数据集：HumanEval风格编程题"""
    pass


def get_code_generation_dataset():
    """获取代码生成数据集配置"""
    reader_cfg = dict(
        input_columns=['prompt'],
        output_column='test',
    )

    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
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

    eval_cfg = dict(
        evaluator=dict(type='HumanEvalEvaluator', timeout=30),
        pred_role='BOT',
    )

    dataset = dict(
        abbr='code_generation',
        type='代码生成Dataset',
        path='./datasets/代码生成.jsonl',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )

    return dataset
