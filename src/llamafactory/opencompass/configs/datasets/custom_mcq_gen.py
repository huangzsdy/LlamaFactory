from mmengine.config import read_base

with read_base():
    from opencompass.configs.models.qwen2_5.hf_qwen2_5_7b_instruct import models

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CustomDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

datasets = []

custom_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: '),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

custom_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess),
)

datasets.append(dict(
    abbr='custom_mcq',
    type=CustomDataset,
    path='/path/to/your/dataset.jsonl',
    reader_cfg=dict(
        input_columns=['question', 'A', 'B', 'C', 'D'],
        output_column='answer',
    ),
    infer_cfg=custom_infer_cfg,
    eval_cfg=custom_eval_cfg,
))

for m in models:
    m['path'] = '/path/to/your/fine-tuned-qwen2.5-7b-instruct'
