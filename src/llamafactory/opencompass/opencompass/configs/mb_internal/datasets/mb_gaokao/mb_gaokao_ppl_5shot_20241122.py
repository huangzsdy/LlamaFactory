from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import MbGaokaoDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess, first_option_postprocess


mb_gaokao_name_list = [
    'gaokaotagging2024_1123',
    'gaokaotagging2024_1126',
    'gaokaotagging2024_1127',
    'gaokaotagging2024_1128',
    'gaokaotagging2024_1129',
]

mb_gaokao_5shot_datasets = []
for _name in mb_gaokao_name_list:
    mb_gaokao_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template={
                    answer: dict(
                        begin='</E>',
                        round=[
                            dict(
                                role='HUMAN',
                                prompt=
                                f'问题:{{input}}\n答案: '
                            ),
                            dict(role='BOT', prompt=answer),
                        ])
                    for answer in ['A', 'B', 'C', 'D']
                },
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=PPLInferencer),
    )

    mb_gaokao_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        )

    mb_gaokao_5shot_datasets.append(
        dict(
            type=MbGaokaoDataset,
            path='./data/mb_gaokao',
            name=_name,
            abbr='mb_gaokao-' + _name + '-5shot',
            reader_cfg=dict(
                input_columns=['input'],
                output_column='golden',
                train_split='dev',
                test_split='test'),
            infer_cfg=mb_gaokao_infer_cfg,
            eval_cfg=mb_gaokao_eval_cfg,
        ))

del _name
