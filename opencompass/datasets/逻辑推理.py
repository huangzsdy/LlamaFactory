"""逻辑推理数据集评测配置"""

from opencompass.datasets.custom import CustomDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class 逻辑推理Dataset(CustomDataset):
    """逻辑推理数据集：布尔逻辑表达式求值"""
    pass


def get_logic_reasoning_dataset():
    """获取逻辑推理数据集配置"""
    reader_cfg = dict(
        input_columns=['question'],
        output_column='answer',
        dataset_post_processor=lambda x: x  # 直接返回原格式
    )

    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
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

    eval_cfg = dict(
        evaluator=dict(type='AccEvaluator'),
        pred_role='BOT',
    )

    dataset = dict(
        abbr='logic_reasoning',
        type='逻辑推理Dataset',
        path='./datasets/逻辑推理.jsonl',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )

    return dataset
