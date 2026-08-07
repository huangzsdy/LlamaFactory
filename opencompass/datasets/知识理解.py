"""知识理解数据集评测配置"""

from opencompass.datasets.custom import CustomDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class 知识理解Dataset(CustomDataset):
    """知识理解数据集：会计/财务知识问答"""
    pass


def get_knowledge_understanding_dataset():
    """获取知识理解数据集配置"""
    reader_cfg = dict(
        input_columns=['question'],
        output_column='answer',
    )

    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
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

    eval_cfg = dict(
        evaluator=dict(type='AccEvaluator'),
        pred_role='BOT',
    )

    dataset = dict(
        abbr='knowledge_understanding',
        type='知识理解Dataset',
        path='./datasets/知识理解.jsonl',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )

    return dataset
