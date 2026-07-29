from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CMMLUDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

cmmlu_subject_mapping = {
    'agronomy': 'ᠲᠠᠷᠢᠶᠠᠯᠠᠩ ᠤ ᠤᠬᠠᠭᠠᠨ',
    'anatomy': 'ᠵᠠᠳᠠᠯᠠᠨ ᠰᠢᠨᠵᠢᠯᠡᠬᠦ ᠤᠬᠠᠭᠠᠨ',
    'ancient_chinese': 'ᠡᠷᠲᠡ ᠶᠢᠨ ᠬᠢᠲᠠᠳ ᠬᠡᠯᠡ ᠪᠢᠴᠢᠭ',
    'arts': 'ᠤᠷᠠᠯᠢᠭ ᠤᠨ ᠤᠬᠠᠭᠠᠨ',
    'astronomy': 'ᠣᠳᠣᠨ ᠣᠷᠣᠨ ᠤ ᠤᠬᠠᠭᠠᠨ',
    'business_ethics': 'ᠬᠤᠳᠠᠯᠳᠤᠭᠠᠨ ᠤ ᠶᠣᠰᠣ ᠵᠦᠢ',
    'chinese_civil_service_exam': 'ᠬᠢᠲᠠᠳ ᠤᠨ ᠲᠥᠷᠥ ᠶᠢᠨ ᠠᠯᠪᠠᠨ ᠬᠠᠭᠠᠭᠴᠢ ᠶᠢᠨ ᠱᠠᠯᠭᠠᠯᠲᠠ',
    'chinese_driving_rule': 'ᠬᠢᠲᠠᠳ ᠤᠨ ᠵᠣᠯᠣᠭᠣᠳᠬᠤ ᠳᠦᠷᠢᠮ',
    'chinese_food_culture': 'ᠬᠢᠲᠠᠳ ᠤᠨ ᠢᠳᠡᠭᠡ ᠤᠮᠳᠠᠭᠠᠨ ᠤ ᠰᠣᠶᠣᠯ',
    'chinese_foreign_policy': 'ᠬᠢᠲᠠᠳ ᠤᠨ ᠭᠠᠳᠠᠭᠠᠳᠤ ᠪᠣᠳᠣᠯᠭ᠎ᠠ',
    'chinese_history': 'ᠬᠢᠲᠠᠳ ᠤᠨ ᠲᠡᠦᠬᠡ',
    'chinese_literature': 'ᠬᠢᠲᠠᠳ ᠤᠨ ᠤᠳᠬᠠ ᠵᠣᠬᠢᠶᠠᠯ',
    'chinese_teacher_qualification': 'ᠬᠢᠲᠠᠳ ᠤᠨ ᠪᠠᠭᠰᠢ ᠶᠢᠨ ᠮᠡᠷᠭᠡᠵᠢᠯ ᠦᠨ ᠦᠨᠡᠮᠯᠡᠬᠦ',
    'clinical_knowledge': 'ᠺᠯᠢᠨᠢᠺ ᠦᠨ ᠮᠡᠳᠡᠯᠭᠡ',
    'college_actuarial_science': 'ᠶᠡᠬᠡ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠳᠠᠭᠠᠭᠠᠯᠲᠠ ᠶᠢᠨ ᠲᠣᠭᠠᠴᠠᠭᠠᠨ ᠤ ᠤᠬᠠᠭᠠᠨ',
    'college_education': 'ᠶᠡᠬᠡ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠰᠤᠷᠭᠠᠨ ᠬᠥᠮᠦᠵᠢᠯ ᠦᠨ ᠤᠬᠠᠭᠠᠨ',
    'college_engineering_hydrology': 'ᠶᠡᠬᠡ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠪᠠᠶ᠋ᠢᠭᠤᠯᠤᠯᠲᠠ ᠶᠢᠨ ᠤᠰᠤᠨ ᠵᠦᠢ',
    'college_law': 'ᠶᠡᠬᠡ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠬᠠᠤᠯᠢ ᠵᠦᠢ',
    'college_mathematics': 'ᠶᠡᠬᠡ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠲᠣᠭᠠᠨ ᠤ ᠤᠬᠠᠭᠠᠨ',
    'college_medical_statistics': 'ᠶᠡᠬᠡ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠠᠨᠠᠭᠠᠬᠤ ᠤᠬᠠᠭᠠᠨ ᠤ ᠰᠲᠠᠲᠢᠰᠲᠢᠺ',
    'college_medicine': 'ᠶᠡᠬᠡ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠠᠨᠠᠭᠠᠬᠤ ᠤᠬᠠᠭᠠᠨ',
    'computer_science': 'ᠲᠣᠭᠠᠴᠠᠭᠠᠯᠠᠭᠤᠷ ᠤᠨ ᠤᠬᠠᠭᠠᠨ',
    'computer_security': 'ᠲᠣᠭᠠᠴᠠᠭᠠᠯᠠᠭᠤᠷ ᠤᠨ ᠠᠶᠤᠯ ᠦᠭᠡᠢ ᠪᠠᠶ᠋ᠢᠳᠠᠯ',
    'conceptual_physics': 'ᠤᠬᠠᠭᠠᠳᠠᠬᠤᠢ ᠶᠢᠨ ᠪᠣᠳᠠᠰ ᠵᠦᠢ',
    'construction_project_management': 'ᠪᠠᠶ᠋ᠢᠭᠤᠯᠤᠯᠲᠠ ᠶᠢᠨ ᠢᠨᠵᠧᠨᠧᠷ ᠦᠨ ᠬᠠᠮᠢᠶᠠᠷᠤᠯᠲᠠ',
    'economics': 'ᠡᠳ ᠦᠨ ᠵᠠᠰᠠᠭ ᠤᠨ ᠤᠬᠠᠭᠠᠨ',
    'education': 'ᠰᠤᠷᠭᠠᠨ ᠬᠥᠮᠦᠵᠢᠯ ᠦᠨ ᠤᠬᠠᠭᠠᠨ',
    'electrical_engineering': 'ᠴᠠᠬᠢᠯᠭᠠᠨ ᠪᠠᠶ᠋ᠢᠭᠤᠯᠤᠯᠲᠠ ᠶᠢᠨ ᠤᠬᠠᠭᠠᠨ',
    'elementary_chinese': 'ᠪᠠᠭᠠ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠬᠢᠲᠠᠳ ᠬᠡᠯᠡ ᠪᠢᠴᠢᠭ',
    'elementary_commonsense': 'ᠪᠠᠭᠠ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠡᠩ ᠦᠨ ᠮᠡᠳᠡᠯᠭᠡ',
    'elementary_information_and_technology': 'ᠪᠠᠭᠠ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠮᠡᠳᠡᠭᠡᠯᠡᠯ ᠪᠠ ᠲᠧᠬᠨᠣᠯᠣᠭᠢ',
    'elementary_mathematics': 'ᠪᠠᠭᠠ ᠱᠠᠲᠤᠨ ᠤ ᠲᠣᠭᠠᠨ ᠤ ᠤᠬᠠᠭᠠᠨ',
    'ethnology': 'ᠮᠢᠨᠵᠤ ᠤᠳᠬᠠ ᠵᠣᠬᠢᠶᠠᠯ',
    'food_science': 'ᠬᠥᠨᠡᠰᠦᠨ ᠦ ᠤᠬᠠᠭᠠᠨ',
    'genetics': 'ᠤᠳᠤᠮ ᠰᠢᠨᠵᠢᠯᠡᠬᠦ ᠤᠬᠠᠭᠠᠨ',
    'global_facts': 'ᠳᠡᠯᠡᠬᠡᠢ ᠶᠢᠨ ᠪᠣᠳᠠᠲᠤ ᠪᠠᠶ᠋ᠢᠳᠠᠯ',
    'high_school_biology': 'ᠳᠤᠮᠳᠠᠳᠤ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠠᠮᠢ ᠪᠣᠳᠠᠰ ᠤᠨ ᠤᠬᠠᠭᠠᠨ',
    'high_school_chemistry': 'ᠳᠤᠮᠳᠠᠳᠤ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠬᠢᠮᠢ ᠶᠢᠨ ᠤᠬᠠᠭᠠᠨ',
    'high_school_geography': 'ᠳᠤᠮᠳᠠᠳᠤ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠭᠠᠵᠠᠷ ᠵᠦᠢ',
    'high_school_mathematics': 'ᠳᠤᠮᠳᠠᠳᠤ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠲᠣᠭᠠᠨ ᠤ ᠤᠬᠠᠭᠠᠨ',
    'high_school_physics': 'ᠳᠤᠮᠳᠠᠳᠤ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠪᠣᠳᠠᠰ ᠵᠦᠢ',
    'high_school_politics': 'ᠳᠤᠮᠳᠠᠳᠤ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠶᠢᠨ ᠲᠥᠷᠥ ᠵᠦᠢ',
    'human_sexuality': 'ᠬᠥᠮᠦᠨ ᠲᠥᠷᠥᠯᠬᠢᠲᠡᠨ ᠦ ᠪᠡᠶ᠎ᠡ ᠶᠢᠨ ᠬᠠᠷᠢᠯᠴᠠᠭ᠎ᠠ',
    'international_law': 'ᠣᠯᠠᠨ ᠤᠯᠤᠰ ᠤᠨ ᠬᠠᠤᠯᠢ ᠵᠦᠢ',
    'journalism': 'ᠰᠡᠳᠬᠦᠯ ᠵᠦᠢ',
    'jurisprudence': 'ᠬᠠᠤᠯᠢ ᠶᠢᠨ ᠣᠨᠣᠯ',
    'legal_and_moral_basis': 'ᠬᠠᠤᠯᠢ ᠪᠠ ᠶᠣᠰᠣ ᠰᠤᠷᠲᠠᠬᠤᠨ ᠤ ᠦᠨᠳᠦᠰᠦ',
    'logical': 'ᠭᠢᠰᠢᠭ ᠤᠨ ᠤᠬᠠᠭᠠᠨ',
    'machine_learning': 'ᠮᠠᠰᠢᠨ ᠤ ᠰᠤᠷᠤᠯᠲᠠ',
    'management': 'ᠬᠠᠮᠢᠶᠠᠷᠤᠯᠲᠠ ᠶᠢᠨ ᠤᠬᠠᠭᠠᠨ',
    'marketing': 'ᠵᠠᠬᠠ ᠵᠡᠭᠡᠯᠢ ᠶᠢᠨ ᠪᠣᠷᠣᠯᠠᠭᠤᠯᠤᠯᠲᠠ',
    'marxist_theory': 'ᠮᠠᠷᠺᠰ ᠦᠨ ᠣᠨᠣᠯ',
    'modern_chinese': 'ᠣᠳᠣ ᠦᠶ᠎ᠡ ᠶᠢᠨ ᠬᠢᠲᠠᠳ ᠬᠡᠯᠡ',
    'nutrition': 'ᠲᠡᠵᠢᠭᠡᠯ ᠦᠨ ᠤᠬᠠᠭᠠᠨ',
    'philosophy': 'ᠭᠢᠳᠢ ᠶᠢᠨ ᠤᠬᠠᠭᠠᠨ',
    'professional_accounting': 'ᠮᠡᠷᠭᠡᠵᠢᠯ ᠦᠨ ᠨᠢᠭᠲᠠᠯᠠᠨ ᠪᠣᠳᠣᠯᠲᠠ',
    'professional_law': 'ᠮᠡᠷᠭᠡᠵᠢᠯ ᠦᠨ ᠬᠠᠤᠯᠢ ᠵᠦᠢ',
    'professional_medicine': 'ᠮᠡᠷᠭᠡᠵᠢᠯ ᠦᠨ ᠠᠨᠠᠭᠠᠬᠤ ᠤᠬᠠᠭᠠᠨ',
    'professional_psychology': 'ᠮᠡᠷᠭᠡᠵᠢᠯ ᠦᠨ ᠰᠡᠳᠬᠢᠯ ᠰᠤᠳᠤᠯᠤᠯ',
    'public_relations': 'ᠣᠯᠠᠨ ᠨᠡᠶ᠋ᠢᠲᠡ ᠶᠢᠨ ᠬᠠᠷᠢᠯᠴᠠᠭ᠎ᠠ',
    'security_study': 'ᠠᠶᠤᠯ ᠦᠭᠡᠢ ᠪᠠᠶ᠋ᠢᠳᠠᠯ ᠤᠨ ᠰᠤᠳᠤᠯᠤᠯ',
    'sociology': 'ᠨᠡᠶ᠋ᠢᠭᠡᠮ ᠰᠤᠳᠤᠯᠤᠯ',
    'sports_science': 'ᠪᠡᠶ᠎ᠡ ᠶᠢᠨ ᠲᠠᠮᠢᠷ ᠤᠨ ᠤᠬᠠᠭᠠᠨ',
    'traditional_chinese_medicine': 'ᠬᠢᠲᠠᠳ ᠤᠨ ᠤᠯᠠᠮᠵᠢᠯᠠᠯᠲᠤ ᠡᠮᠨᠡᠯᠭᠡ',
    'virology': 'ᠸᠢᠷᠦᠰ ᠰᠤᠳᠤᠯᠤᠯ',
    'world_history': 'ᠳᠡᠯᠡᠬᠡᠢ ᠶᠢᠨ ᠲᠡᠦᠬᠡ',
    'world_religions': 'ᠳᠡᠯᠡᠬᠡᠢ ᠶᠢᠨ ᠱᠠᠰᠢᠨ ᠱᠦᠲᠦᠯᠭᠡ'
}


cmmlu_all_sets = list(cmmlu_subject_mapping.keys())

cmmlu_mn_datasets = []
for _name in cmmlu_all_sets:
    _ch_name = cmmlu_subject_mapping[_name]
    cmmlu_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template={
                answer: dict(
                    begin='</E>',
                    round=[
                        dict(
                            role='HUMAN',
                            prompt=f'ᠳᠣᠣᠷᠠᠬᠢ ᠨᠢ {_ch_name} ᠤᠨ ᠲᠤᠬᠠᠢ ᠭᠠᠭᠴᠠ ᠰᠣᠩᠭᠣᠯᠲᠠᠲᠤ ᠠᠰᠠᠭᠤᠯᠲᠠ ᠪᠣᠯᠤᠨ᠎ᠠ᠂ ᠵᠥᠪ ᠬᠠᠷᠢᠭᠤᠯᠲᠠ ᠶᠢᠨ ᠰᠣᠩᠭᠣᠯᠲᠠ ᠶᠢ ᠰᠢᠳᠤᠳ ᠥᠭ᠋ᠭᠦᠭᠡᠷᠡᠢ᠃\nᠠᠰᠠᠭᠤᠯᠲᠠ᠄ {{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}'
                            #prompt=f'以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n题目：{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}'
                        ),
                        dict(role='BOT', prompt=f'答案是: {answer}'),
                    ])
                for answer in ['A', 'B', 'C', 'D']
            },
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=PPLInferencer),
    )

    cmmlu_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

    cmmlu_mn_datasets.append(
        dict(
            type=CMMLUDataset,
            path='data/cmmlu_meng',
            name=_name,
            abbr=f'cmmlu-{_name}',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='dev',
                test_split='test'),
            infer_cfg=cmmlu_infer_cfg,
            eval_cfg=cmmlu_eval_cfg,
        ))

del _name, _ch_name
