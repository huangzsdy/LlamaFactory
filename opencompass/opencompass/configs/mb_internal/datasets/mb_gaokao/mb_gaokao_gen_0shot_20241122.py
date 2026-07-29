from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import MbGaokaoDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


mb_gaokao_name_list = [
    'gaokaotagging2024_1123',
    'gaokaotagging2024_1126',
    'gaokaotagging2024_1127',
    'gaokaotagging2024_1128',
    'gaokaotagging2024_1129',
]

mb_gaokao_0shot_datasets = []

for _name in mb_gaokao_name_list:
    mb_gaokao_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=
                        '问题：{input}'
                    ),
                    dict(role='BOT', prompt='答案：{golden}'),
                ]),
            ice_token='</E>',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=512),
    )

    mb_gaokao_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD', cushion=True)
        )

    mb_gaokao_0shot_datasets.append(
        dict(
            type=MbGaokaoDataset,
            path='./data/mb_gaokao',
            name=_name,
            abbr='mb_gaokao-' + _name + '-0shot',
            reader_cfg=dict(
                input_columns=['input'],
                output_column='golden',
                train_split='dev',
                test_split='test'),
            infer_cfg=mb_gaokao_infer_cfg,
            eval_cfg=mb_gaokao_eval_cfg,
        ))

del _name
