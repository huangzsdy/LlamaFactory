from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import MMMLUDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


mmmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D','subject'],
    output_column='target',
    train_split='test')

mmmlu_all_sets = [
    'mmlu_AR-XY',
    'mmlu_BN-BD',
    'mmlu_DE-DE',
    'mmlu_ES-LA',
    'mmlu_FR-FR',
    'mmlu_HI-IN',
    'mmlu_ID-ID',
    'mmlu_IT-IT',
    'mmlu_JA-JP',
    'mmlu_KO-KR',
    'mmlu_PT-BR',
    'mmlu_SW-KE',
    'mmlu_YO-NG',
    'mmlu_ZH-CN',
]
print("*" * 20 + f"mmmlu_ppl_v2" + "*" * 20)
# 定义few-shot示例
few_shot_examples = {
    'AR': [
        {"question": "ما هو العاصمة الإدارية لمصر؟", "A": "القاهرة", "B": "الإسكندرية", "C": "الجيزة", "D": "الفيوم", "answer": "A"},
        {"question": "كم عدد أيام الأسبوع؟", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'BN': [
        {"question": "বাংলাদেশের রাজধানী কি?", "A": "কলকাতা", "B": "ঢাকা", "C": "দিল্লি", "D": "করাচি", "answer": "B"},
        {"question": "পৃথিবীতে কয়টি মহাদেশ আছে?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'DE': [
        {"question": "Was ist die Hauptstadt von Deutschland?", "A": "Berlin", "B": "München", "C": "Hamburg", "D": "Köln", "answer": "A"},
        {"question": "Wie viele Tage hat eine Woche?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'ES': [
        {"question": "¿Cuál es la capital de España?", "A": "Madrid", "B": "Barcelona", "C": "Valencia", "D": "Sevilla", "answer": "A"},
        {"question": "¿Cuántos días tiene una semana?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'FR': [
        {"question": "Quelle est la capitale de la France?", "A": "Paris", "B": "Lyon", "C": "Marseille", "D": "Toulouse", "answer": "A"},
        {"question": "Combien de jours y a-t-il dans une semaine?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'HI': [
        {"question": "भारत की राजधानी क्या है?", "A": "मुंबई", "B": "दिल्ली", "C": "कोलकाता", "D": "चेन्नई", "answer": "B"},
        {"question": "एक सप्ताह में कितने दिन होते हैं?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'ID': [
        {"question": "Apa ibu kota Indonesia?", "A": "Jakarta", "B": "Surabaya", "C": "Bandung", "D": "Medan", "answer": "A"},
        {"question": "Berapa hari dalam seminggu?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'IT': [
        {"question": "Qual è la capitale dell'Italia?", "A": "Roma", "B": "Milano", "C": "Napoli", "D": "Torino", "answer": "A"},
        {"question": "Quanti giorni ha una settimana?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'JA': [
        {"question": "日本の首都はどこですか？", "A": "東京", "B": "大阪", "C": "名古屋", "D": "福岡", "answer": "A"},
        {"question": "一週間は何日ありますか？", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'KO': [
        {"question": "대한민국의 수도는 어디인가요?", "A": "서울", "B": "부산", "C": "대구", "D": "인천", "answer": "A"},
        {"question": "일주일은 몇 일인가요?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'PT': [
        {"question": "Qual é a capital do Brasil?", "A": "Brasília", "B": "São Paulo", "C": "Rio de Janeiro", "D": "Salvador", "answer": "A"},
        {"question": "Quantos dias tem uma semana?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'SW': [
        {"question": "Je, jiji la Kenya ni nini?", "A": "Nairobi", "B": "Mombasa", "C": "Kisumu", "D": "Nakuru", "answer": "A"},
        {"question": "Juma lina siku ngapi?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'YO': [
        {"question": "Kini olu ilu Nigeria?", "A": "Lagos", "B": "Abuja", "C": "Ibadan", "D": "Kano", "answer": "B"},
        {"question": "Ojo melo ni ose kan ni?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'ZH': [
        {"question": "中国的首都是哪里？", "A": "上海", "B": "北京", "C": "广州", "D": "深圳", "answer": "B"},
        {"question": "一周有几天？", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ],
    'EN': [  # 添加英语作为默认
        {"question": "What is the capital of the United States?", "A": "New York", "B": "Washington D.C.", "C": "Los Angeles", "D": "Chicago", "answer": "B"},
        {"question": "How many days are in a week?", "A": "5", "B": "6", "C": "7", "D": "8", "answer": "C"}
    ]
}

# 定义每种语言的提示语
language_prompts = {
    'AR': 'إليك بعض الأمثلة، يرجى الرجوع إليها للإجابة على السؤال الحالي:',
    'BN': 'নিম্নলিখিত উদাহরণগুলি দেখুন এবং বর্তমান প্রশ্নের উত্তর দিতে সেগুলি ব্যবহার করুন:',
    'DE': 'Hier sind einige Beispiele. Bitte verwenden Sie sie als Referenz, um die aktuelle Frage zu beantworten:',
    'ES': 'Aquí hay algunos ejemplos. Por favor, úsalos como referencia para responder la pregunta actual:',
    'FR': 'Voici quelques exemples. Veuillez les utiliser comme référence pour répondre à la question actuelle :',
    'HI': 'यहाँ कुछ उदाहरण दिए गए हैं। कृपया वर्तमान प्रश्न का उत्तर देने के लिए उनका संदर्भ के रूप में उपयोग करें:',
    'ID': 'Berikut adalah beberapa contoh. Silakan gunakan mereka sebagai referensi untuk menjawab pertanyaan saat ini:',
    'IT': 'Ecco alcuni esempi. Si prega di utilizzarli come riferimento per rispondere alla domanda attuale:',
    'JA': '以下はいくつかの例です。現在の質問に答えるための参考としてこれらを使用してください：',
    'KO': '다음은 몇 가지 예시입니다. 현재 질문에 답하기 위해 참고로 사용하십시오:',
    'PT': 'Aqui estão alguns exemplos. Por favor, use-os como referência para responder a pergunta atual:',
    'SW': 'Haya ni mifano michache. Tafadhali tumia kama kumbukumbu ili kujibu swali la sasa:',
    'YO': 'Eyi ni awọn apejuwe diẹ. Jọwọ lo wọn bi itọkasi lati dahun ibeere lọwọlọwọ:',
    'ZH': '以下是一些示例，请参考它们回答当前问题：',
    'EN': 'Here are some examples. Please use them as reference to answer the current question:'
}

def create_few_shot_prompt(lang, subject, question, A, B, C, D):
    """创建few-shot提示词模板"""
    # 如果语言不在示例中，使用英语示例
    examples = few_shot_examples.get(lang, few_shot_examples['EN'])
    
    # 构建few-shot示例部分
    examples_text = ""
    for i, example in enumerate(examples, 1):
        examples_text += f"示例 {i}:\n" if lang in ['ZH', 'JA', 'KO'] else f"Example {i}:\n"
        examples_text += f"问题：{example['question']}\n" if lang in ['ZH', 'JA', 'KO'] else f"Question: {example['question']}\n"
        examples_text += f"A. {example['A']}\n"
        examples_text += f"B. {example['B']}\n"
        examples_text += f"C. {example['C']}\n"
        examples_text += f"D. {example['D']}\n"
        
        if lang == 'ZH':
            examples_text += f"答案：{example['answer']}\n\n"
        elif lang == 'JA':
            examples_text += f"回答：{example['answer']}\n\n"
        elif lang == 'KO':
            examples_text += f"답변：{example['answer']}\n\n"
        elif lang == 'AR':
            examples_text += f"الإجابة：{example['answer']}\n\n"
        elif lang == 'DE':
            examples_text += f"Antwort: {example['answer']}\n\n"
        elif lang == 'ES':
            examples_text += f"Respuesta: {example['answer']}\n\n"
        elif lang == 'FR':
            examples_text += f"Réponse: {example['answer']}\n\n"
        elif lang == 'HI':
            examples_text += f"उत्तर: {example['answer']}\n\n"
        elif lang == 'IT':
            examples_text += f"Risposta: {example['answer']}\n\n"
        elif lang == 'PT':
            examples_text += f"Resposta: {example['answer']}\n\n"
        else:
            examples_text += f"Answer: {example['answer']}\n\n"
    
    # 构建当前问题部分
    if lang == 'ZH':
        current_question = f"当前问题（关于{subject}）：\n"
        current_question += f"问题：{question}\n"
        current_question += f"A. {A}\n"
        current_question += f"B. {B}\n"
        current_question += f"C. {C}\n"
        current_question += f"D. {D}\n"
        current_question += "答案："
    elif lang == 'JA':
        current_question = f"現在の質問（{subject}について）：\n"
        current_question += f"質問: {question}\n"
        current_question += f"A. {A}\n"
        current_question += f"B. {B}\n"
        current_question += f"C. {C}\n"
        current_question += f"D. {D}\n"
        current_question += "回答："
    elif lang == 'KO':
        current_question = f"현재 질문 ({subject}에 관한):\n"
        current_question += f"질문: {question}\n"
        current_question += f"A. {A}\n"
        current_question += f"B. {B}\n"
        current_question += f"C. {C}\n"
        current_question += f"D. {D}\n"
        current_question += "답변："
    elif lang == 'AR':
        current_question = f"السؤال الحالي (حول {subject}):\n"
        current_question += f"السؤال: {question}\n"
        current_question += f"أ. {A}\n"
        current_question += f"ب. {B}\n"
        current_question += f"ج. {C}\n"
        current_question += f"د. {D}\n"
        current_question += "الإجابة:"
    elif lang == 'DE':
        current_question = f"Aktuelle Frage (über {subject}):\n"
        current_question += f"Frage: {question}\n"
        current_question += f"A. {A}\n"
        current_question += f"B. {B}\n"
        current_question += f"C. {C}\n"
        current_question += f"D. {D}\n"
        current_question += "Antwort:"
    elif lang == 'ES':
        current_question = f"Pregunta actual (sobre {subject}):\n"
        current_question += f"Pregunta: {question}\n"
        current_question += f"A. {A}\n"
        current_question += f"B. {B}\n"
        current_question += f"C. {C}\n"
        current_question += f"D. {D}\n"
        current_question += "Respuesta:"
    elif lang == 'FR':
        current_question = f"Question actuelle (sur {subject}):\n"
        current_question += f"Question: {question}\n"
        current_question += f"A. {A}\n"
        current_question += f"B. {B}\n"
        current_question += f"C. {C}\n"
        current_question += f"D. {D}\n"
        current_question += "Réponse:"
    elif lang == 'HI':
        current_question = f"वर्तमान प्रश्न ({subject} के बारे में):\n"
        current_question += f"प्रश्न: {question}\n"
        current_question += f"A. {A}\n"
        current_question += f"B. {B}\n"
        current_question += f"C. {C}\n"
        current_question += f"D. {D}\n"
        current_question += "उत्तर:"
    elif lang == 'IT':
        current_question = f"Domanda attuale (su {subject}):\n"
        current_question += f"Domanda: {question}\n"
        current_question += f"A. {A}\n"
        current_question += f"B. {B}\n"
        current_question += f"C. {C}\n"
        current_question += f"D. {D}\n"
        current_question += "Risposta:"
    elif lang == 'PT':
        current_question = f"Pergunta atual (sobre {subject}):\n"
        current_question += f"Pergunta: {question}\n"
        current_question += f"A. {A}\n"
        current_question += f"B. {B}\n"
        current_question += f"C. {C}\n"
        current_question += f"D. {D}\n"
        current_question += "Resposta:"
    else:
        current_question = f"Current question (about {subject}):\n"
        current_question += f"Question: {question}\n"
        current_question += f"A. {A}\n"
        current_question += f"B. {B}\n"
        current_question += f"C. {C}\n"
        current_question += f"D. {D}\n"
        current_question += "Answer:"
    
    return examples_text + current_question

mmmlu_datasets = []
mmmlu_datasets_map = dict()
for _name in mmmlu_all_sets:
    lang = _name.split('_')[1].split('-')[0]
    
    # 使用语言映射表获取正确的提示语
    _hint = language_prompts.get(lang, language_prompts['EN'])
    _prompt = create_few_shot_prompt(lang, '{subject}', '{input}', '{A}', '{B}', '{C}', '{D}')
    
    # Few-shot配置
    mmmlu_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f'{_hint}\n\n{_prompt}'
                    ),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3]),
        inferencer=dict(
            type=PPLInferencer,
            labels=['A', 'B', 'C', 'D'],
            max_seq_len=4096,  # 增加序列长度以适应few-shot示例
            batch_size=1,
        ),
    )

    mmmlu_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

    dataset_config = dict(
        abbr=f'openai_m{_name}',
        type=MMMLUDataset,
        path='data/mmmlu',
        name=_name,
        reader_cfg=mmmlu_reader_cfg,
        infer_cfg=mmmlu_infer_cfg,
        eval_cfg=mmmlu_eval_cfg,
    )
    
    mmmlu_datasets.append(dataset_config)
    mmmlu_datasets_map[lang.lower()] = dataset_config

del _name, _hint, _prompt