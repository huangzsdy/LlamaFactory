from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CMMLUDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

cmmlu_subject_mapping = {
    'agronomy': 'སོ་ནམ་རིག་པ།',
    'anatomy': 'གཤགས་བཅོས་རིག་པ།',
    'ancient_chinese': 'གནའ་བོའི་རྒྱ་ཡིག',
    'arts': 'སྒྱུ་རྩལ་རིག་པ།',
    'astronomy': 'སྐར་རྩིས་རིག་པ།',
    'business_ethics': 'ཚོང་ལས་ཟོལ་མེད་ཀུན་སྤྱོད།',
    'chinese_civil_service_exam': 'རྒྱ་ནག་གི་གཞུང་ཞབས་རྒྱུགས་ལེན།',
    'chinese_driving_rule': 'རྒྱ་ནག་གི་ཁ་ལོ་བའི་སྒྲིག་ལམ།',
    'chinese_food_culture': 'རྒྱ་ནག་གི་བཟའ་བཏུང་རིག་གནས།',
    'chinese_foreign_policy': 'རྒྱ་ནག་གི་ཕྱི་འབྲེལ་སྲིད་ཇུས།',
    'chinese_history': 'རྒྱ་ནག་གི་ལོ་རྒྱུས།',
    'chinese_literature': 'རྒྱ་ནག་གི་རྩོམ་རིག་།',
    'chinese_teacher_qualification': 'རྒྱ་ནག་གི་དགེ་རྒན་ཚད་གཞི།',
    'clinical_knowledge': 'ནད་ཐོག་ཤེས་བྱ།',
    'college_actuarial_science': 'སློབ་ཆེན་འགན་བཅོལ་རྩིས་རིག་པ།',
    'college_education': 'སློབ་ཆེན་སློབ་གསོ་རིག་པ།',
    'college_engineering_hydrology': 'སློབ་ཆེན་འདྲེན་བཀོད་ཆུ་བེད་རིག་པ།',
    'college_law': 'སློབ་ཆེན་ཁྲིམས་ལུགས།',
    'college_mathematics': 'སློབ་ཆེན་རྩིས་རིག་།',
    'college_medical_statistics': 'སློབ་ཆེན་སྨན་བཅོས་སྡོམ་རྩིས་རིག་པ།',
    'college_medicine': 'སློབ་ཆེན་གསོ་རིག་།',
    'computer_science': 'གློག་ཀླད་ཚན་རིག་།',
    'computer_security': 'གློག་ཀླད་བདེ་འཇགས།',
    'conceptual_physics': 'འཆར་སྣང་དངོས་ལུགས་རིག་པ།',
    'construction_project_management': 'འཛུགས་སྐྲུན་ལས་གཞི་འགོ་གཉེར།',
    'economics': 'དཔལ་འབྱོར་རིག་པ།',
    'education': 'སློབ་གསོ་རིག་པ།',
    'electrical_engineering': 'གློག་ཤུགས་འདྲེན་བཀོད་རིག་པ།',
    'elementary_chinese': 'སློབ་ཆུང་རྒྱ་ཡིག',
    'elementary_commonsense': 'སློབ་ཆུང་རྒྱུན་ཤེས།',
    'elementary_information_and_technology': 'སློབ་ཆུང་འཕྲིན་གཏོང་དང་འཕྲུལ་རྩལ།',
    'elementary_mathematics': 'དམའ་རིམ་རྩིས་རིག་།',
    'ethnology': 'མི་རིགས་རིག་པ།',
    'food_science': 'བཟའ་བཏུང་ཚན་རིག་།',
    'genetics': 'རིགས་རྒྱུད་དཔྱད་རིག་།',
    'global_facts': 'འཛམ་གླིང་དངོས་དོན་།',
    'high_school_biology': 'མཐོ་འབྲིང་སྐྱེ་དངོས་རིག་པ།',
    'high_school_chemistry': 'མཐོ་འབྲིང་རྫས་འགྱུར་རིག་པ།',
    'high_school_geography': 'མཐོ་འབྲིང་ས་ཁམས་རིག་པ།',
    'high_school_mathematics': 'མཐོ་འབྲིང་རྩིས་རིག་།',
    'high_school_physics': 'མཐོ་འབྲིང་དངོས་ལུགས་རིག་པ།',
    'high_school_politics': 'མཐོ་འབྲིང་ཆབ་སྲིད།',
    'human_sexuality': 'མིའི་རིགས་ཀྱི་ཆགས་སྤྱོད།',
    'international_law': 'རྒྱལ་སྤྱིའི་ཁྲིམས་ལུགས།',
    'journalism': 'གསར་འགྱུར་རིག་པ།',
    'jurisprudence': 'ཁྲིམས་ལུགས་རིག་པ།',
    'legal_and_moral_basis': 'ཁྲིམས་ལུགས་དང་ཀུན་སྤྱོད་རྨང་གཞི།',
    'logical': 'ཚད་མ་རིག་པ།',
    'machine_learning': 'འཕྲུལ་འཁོར་སློབ་སྦྱོང་།',
    'management': 'འགོ་གཉེར་རིག་པ།',
    'marketing': 'ཁྲོམ་རྭ་འཚོང་ལས།',
    'marxist_theory': 'མཱ་ཁེ་སི་རིང་ལུགས་ཀྱི་ལྟ་བ།',
    'modern_chinese': 'དེང་རབས་རྒྱ་ཡིག',
    'nutrition': 'ཟས་བཅུད་རིག་པ།',
    'philosophy': 'ལྟ་གྲུབ་རིག་པ།',
    'professional_accounting': 'ཆེད་ལས་རྩིས་གཉེར།',
    'professional_law': 'ཆེད་ལས་ཁྲིམས་ལུགས།',
    'professional_medicine': 'ཆེད་ལས་གསོ་རིག་།',
    'professional_psychology': 'ཆེད་ལས་སེམས་ཁམས་རིག་པ།',
    'public_relations': 'སྤྱི་ཚོགས་འབྲེལ་ལམ།',
    'security_study': 'བདེ་འཇགས་ཞིབ་འཇུག',
    'sociology': 'སྤྱི་ཚོགས་རིག་པ།',
    'sports_science': 'ལུས་རྩལ་ཚན་རིག་།',
    'traditional_chinese_medicine': 'རྒྱ་སྨན།',
    'virology': 'ནད་དུག་རིག་པ།',
    'world_history': 'འཛམ་གླིང་ལོ་རྒྱུས།',
    'world_religions': 'འཛམ་གླིང་ཆོས་ལུགས་ཁག'
}

cmmlu_all_sets = list(cmmlu_subject_mapping.keys())

cmmlu_bo_datasets = []
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
                            prompt=f'གཤམ་གསལ་ནི་{_ch_name}་སྐོར་གྱི་འདེམ་ཀ་གཅིག་མའི་དྲི་བ་ཡིན། ལན་འདེམས་ཡང་དག་དེ་ཐད་ཀར་སྤྲོད་རོགས།\nདྲི་བ།：{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}'
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

    cmmlu_bo_datasets.append(
        dict(
            type=CMMLUDataset,
            path='data/cmmlu_zang',
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
