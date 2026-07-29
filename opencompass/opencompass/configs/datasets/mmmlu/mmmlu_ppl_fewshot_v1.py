from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import MMMLUDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

mmmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D', 'subject'],
    output_column='target',
    train_split='test')  # 改为train，用于few-shot示例

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

# 为每种语言定义few-shot示例（3-4个示例）
few_shot_examples = {
    'AR': [
        {"input": "ما هي عاصمة فرنسا؟", "A": "لندن", "B": "باريس", "C": "برلين", "D": "مدريد", "target": "B"},
        {"input": "كم عدد الكواكب في النظام الشمسي؟", "A": "7", "B": "8", "C": "9", "D": "10", "target": "B"},
        {"input": "ما هو العنصر الكيميائي للذهب؟", "A": "Au", "B": "Ag", "C": "Fe", "D": "Cu", "target": "A"}
    ],
    'BN': [
        {"input": "ভারতের রাজধানী কি?", "A": "মুম্বাই", "B": "দিল্লী", "C": "কলকাতা", "D": "চেন্নাই", "target": "B"},
        {"input": "সূর্য কি ধরনের তারকা?", "A": "লাল দানব", "B": "সাদা বামন", "C": "হলুদ বামন", "D": "নিউট্রন তারা",
         "target": "C"},
        {"input": "মানবদেহে কয়টি ক্রোমোজোম আছে?", "A": "23", "B": "46", "C": "64", "D": "32", "target": "B"},
        {"input": "গঙ্গা নদী从哪里 প্রবাহিত হয়?", "A": "বাংলাদেশ", "B": "ভারত", "C": "নেপাল", "D": "চীন",
         "target": "B"}
    ],
    'DE': [
        {"input": "Was ist die Hauptstadt von Deutschland?", "A": "Berlin", "B": "München", "C": "Hamburg",
         "D": "Frankfurt", "target": "A"},
        {"input": "Welches ist das häufigste Element in der Erdatmosphäre?", "A": "Sauerstoff", "B": "Stickstoff",
         "C": "Kohlendioxid", "D": "Wasserstoff", "target": "B"},
        {"input": "Wie viele Kontinente gibt es?", "A": "5", "B": "6", "C": "7", "D": "8", "target": "C"},
        {"input": "Wer schrieb 'Faust'?", "A": "Goethe", "B": "Schiller", "C": "Kafka", "D": "Brecht", "target": "A"}
    ],
    'ES': [
        {"input": "¿Cuál es la capital de España?", "A": "Barcelona", "B": "Madrid", "C": "Valencia", "D": "Sevilla",
         "target": "B"},
        {"input": "¿En qué año llegó Colón a América?", "A": "1492", "B": "1502", "C": "1512", "D": "1482",
         "target": "A"},
        {"input": "¿Cuál es el río más largo del mundo?", "A": "Nilo", "B": "Amazonas", "C": "Misisipi", "D": "Yangtsé",
         "target": "B"}
    ],
    'FR': [
        {"input": "Quelle est la capitale de la France?", "A": "Lyon", "B": "Paris", "C": "Marseille", "D": "Nice",
         "target": "B"},
        {"input": "Qui a peint la Joconde?", "A": "Van Gogh", "B": "Picasso", "C": "Léonard de Vinci", "D": "Monet",
         "target": "C"},
        {"input": "Combien de jours y a-t-il dans une année bissextile?", "A": "365", "B": "366", "C": "364",
         "D": "367", "target": "B"},
        {"input": "Quel est le plus grand océan du monde?", "A": "Atlantique", "B": "Indien", "C": "Pacifique",
         "D": "Arctique", "target": "C"}
    ],
    'HI': [
        {"input": "भारत की राजधानी क्या है?", "A": "मुंबई", "B": "दिल्ली", "C": "कोलकाता", "D": "चेन्नई",
         "target": "B"},
        {"input": "सूर्य किस प्रकार का तारा है?", "A": "लाल दानव", "B": "सफेद बौना", "C": "पीला बौना",
         "D": "न्यूट्रॉन तारा", "target": "C"},
        {"input": "मानव शरीर में कितने गुणसूत्र होते हैं?", "A": "23", "B": "46", "C": "64", "D": "32", "target": "B"}
    ],
    'ID': [
        {"input": "Apa ibu kota Indonesia?", "A": "Jakarta", "B": "Surabaya", "C": "Bandung", "D": "Medan",
         "target": "A"},
        {"input": "Berapa jumlah pulau di Indonesia?", "A": "lebih dari 17000", "B": "5000", "C": "10000", "D": "2000",
         "target": "A"},
        {"input": "Siapa presiden pertama Indonesia?", "A": "Soeharto", "B": "Soekarno", "C": "Habibie",
         "D": "Megawati", "target": "B"},
        {"input": "Apa gunung tertinggi di Indonesia?", "A": "Semeru", "B": "Rinjani", "C": "Kerinci",
         "D": "Puncak Jaya", "target": "D"}
    ],
    'IT': [
        {"input": "Qual è la capitale dell'Italia?", "A": "Milano", "B": "Roma", "C": "Napoli", "D": "Torino",
         "target": "B"},
        {"input": "Chi ha scritto la Divina Commedia?", "A": "Petrarca", "B": "Boccaccio", "C": "Dante", "D": "Ariosto",
         "target": "C"},
        {"input": "Quanti sono i giorni in un anno bisestile?", "A": "365", "B": "366", "C": "364", "D": "367",
         "target": "B"}
    ],
    'JA': [
        {"input": "日本の首都はどこですか？", "A": "大阪", "B": "東京", "C": "名古屋", "D": "福岡", "target": "B"},
        {"input": "富士山の高さは約何メートルですか？", "A": "3776", "B": "3999", "C": "3500", "D": "4000",
         "target": "A"},
        {"input": "桜の花見の季節はいつですか？", "A": "秋", "B": "春", "C": "夏", "D": "冬", "target": "B"},
        {"input": "将棋の駒の数はいくつですか？", "A": "20", "B": "30", "C": "40", "D": "50", "target": "C"}
    ],
    'KO': [
        {"input": "대한민국의 수도는 어디인가요?", "A": "부산", "B": "서울", "C": "대구", "D": "인천", "target": "B"},
        {"input": "한글을 만든 사람은 누구인가요?", "A": "세종대왕", "B": "광개토대왕", "C": "태종", "D": "정조", "target": "A"},
        {"input": "한국의 전통 명절이 아닌 것은?", "A": "설날", "B": "추석", "C": "성탄절", "D": "단오", "target": "C"}
    ],
    'PT': [
        {"input": "Qual é a capital do Brasil?", "A": "São Paulo", "B": "Rio de Janeiro", "C": "Brasília",
         "D": "Salvador", "target": "C"},
        {"input": "Quem descobriu o Brasil?", "A": "Cabral", "B": "Colombo", "C": "Vespucio", "D": "Magalhães",
         "target": "A"},
        {"input": "Qual é o maior rio do mundo?", "A": "Nilo", "B": "Amazonas", "C": "Mississippi", "D": "Yangtzé",
         "target": "B"},
        {"input": "Quantos estados tem o Brasil?", "A": "26", "B": "27", "C": "25", "D": "28", "target": "B"}
    ],
    'ZH': [
        {"input": "中国的首都是哪里？", "A": "上海", "B": "北京", "C": "广州", "D": "深圳", "target": "B"},
        {"input": "长江的长度是多少公里？", "A": "6300", "B": "5800", "C": "5500", "D": "6000", "target": "A"},
        {"input": "哪个朝代建造了长城？", "A": "秦朝", "B": "唐朝", "C": "明朝", "D": "汉朝", "target": "A"},
        {"input": "中国的国花是什么？", "A": "牡丹", "B": "梅花", "C": "菊花", "D": "莲花", "target": "B"}
    ],
    'default': [
        {"input": "What is the capital of France?", "A": "London", "B": "Paris", "C": "Berlin", "D": "Madrid",
         "target": "B"},
        {"input": "How many planets are in the solar system?", "A": "7", "B": "8", "C": "9", "D": "10", "target": "B"},
        {"input": "What is the chemical symbol for gold?", "A": "Au", "B": "Ag", "C": "Fe", "D": "Cu", "target": "A"},
        {"input": "Who wrote 'Romeo and Juliet'?", "A": "Shakespeare", "B": "Dickens", "C": "Austen", "D": "Hemingway",
         "target": "A"}
    ]
}

mmmlu_datasets = []
mmmlu_datasets_map = dict()

for _name in mmmlu_all_sets:
    lang = _name.split('_')[1].split('-')[0]

    # 获取对应语言的示例
    examples = few_shot_examples.get(lang, few_shot_examples['default'])

    # 构建few-shot提示模板
    example_prompts = []
    for ex in examples:
        if 'AR' in _name:
            prompt = f'يتعلق بـ {ex.get("subject", "عام")} \nالسؤال: {ex["input"]}\nأ. {ex["A"]}\nب. {ex["B"]}\nج. {ex["C"]}\nد. {ex["D"]}\nالإجابة: {ex["target"]}'
        elif 'BN' in _name:
            prompt = f'এটি {ex.get("subject", "সাধারণ")} এর সম্পর্কে \nপ্রশ্ন: {ex["input"]}\nএ. {ex["A"]}\nবি. {ex["B"]}\nসি. {ex["C"]}\nডি. {ex["D"]}\nউত্তর: {ex["target"]}'
        elif 'DE' in _name:
            prompt = f'Es geht um {ex.get("subject", "Allgemein")} \nFrage: {ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\nAntwort: {ex["target"]}'
        elif 'ES' in _name:
            prompt = f'Se trata de {ex.get("subject", "General")} \nPregunta: {ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\nRespuesta: {ex["target"]}'
        elif 'FR' in _name:
            prompt = f'''C'est à propos de {ex.get("subject", "Général")} \nQuestion : {ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\nRéponse : {ex["target"]}'''
        elif 'HI' in _name:
            prompt = f'यह {ex.get("subject", "सामान्य")} के बारे में है \nप्रश्न: {ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\nउत्तर: {ex["target"]}'
        elif 'ID' in _name:
            prompt = f'Ini tentang {ex.get("subject", "Umum")} \nPertanyaan: {ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\nJawaban: {ex["target"]}'
        elif 'IT' in _name:
            prompt = f'Si tratta di {ex.get("subject", "Generale")} \nDomanda: {ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\nRisposta: {ex["target"]}'
        elif 'JA' in _name:
            prompt = f'これは {ex.get("subject", "一般")} に関することです \n質問: {ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\n回答: {ex["target"]}'
        elif 'KO' in _name:
            prompt = f'이것은 {ex.get("subject", "일반")}에 관한 것입니다 \n질문: {ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\n답변: {ex["target"]}'
        elif 'PT' in _name:
            prompt = f'É sobre {ex.get("subject", "Geral")} \nPergunta: {ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\nResposta: {ex["target"]}'
        elif 'ZH' in _name:
            prompt = f'这是关于 {ex.get("subject", "通用")} 的内容\n问题：{ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\n答案：{ex["target"]}'
        else:
            prompt = f'it is about {ex.get("subject", "general")} \nQuestion: {ex["input"]}\nA. {ex["A"]}\nB. {ex["B"]}\nC. {ex["C"]}\nD. {ex["D"]}\nAnswer: {ex["target"]}'

        example_prompts.append(prompt)

    # 构建完整的few-shot提示
    few_shot_context = "\n\n".join(example_prompts) + "\n\n"

    if 'AR' in _name:
        _hint = f'هناك سؤال اختيار واحد. أجب عن السؤال بالرد على A أو B أو C أو D'
        _prompt = f'يتعلق بـ {{subject}} \nالسؤال: {{input}}\nأ. {{A}}\nب. {{B}}\nج. {{C}}\nد. {{D}}\nالإجابة:'
    elif 'BN' in _name:
        _hint = f'এটি একটি একক পছন্দের প্রশ্ন। এ, বি, সি বা ডি উত্তর দিয়ে প্রশ্নের উত্তর দিন'
        _prompt = f'এটি {{subject}} এর সম্পর্কে \nপ্রশ্ন: {{input}}\nএ. {{A}}\nবি. {{B}}\nসি. {{C}}\nডি. {{D}}\nউত্তর:'
    elif 'DE' in _name:
        _hint = f'Es gibt eine Einzelwahlfrage. Beantworte die Frage, indem du A, B, C oder D antwortest'
        _prompt = f'Es geht um {{subject}} \nFrage: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAntwort:'
    elif 'ES' in _name:
        _hint = f'Hay una pregunta de elección única. Responde a la pregunta respondiendo A, B, C o D'
        _prompt = f'Se trata de {{subject}} \nPregunta: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nRespuesta:'
    elif 'FR' in _name:
        _hint = f'Il y a une question à choix unique. Répondez à la question en répondant A, B, C ou D'
        _prompt = f'''C'est à propos de {{subject}} \nQuestion : {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nRéponse :'''
    elif 'HI' in _name:
        _hint = f'यह एक एकल विकल्प प्रश्न है। प्रश्न का उत्तर A, B, C या D में से कोई भी उत्तर देकर दें'
        _prompt = f'यह {{subject}} के बारे में है \nप्रश्न: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nउत्तर:'
    elif 'ID' in _name:
        _hint = f'Ada pertanyaan pilihan tunggal. Jawablah pertanyaan dengan menjawab A, B, C, atau D'
        _prompt = f'Ini tentang {{subject}} \nPertanyaan: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nJawaban:'
    elif 'IT' in _name:
        _hint = f'Ci sono domande a scelta singola. Rispondi alla domanda rispondendo A, B, C o D'
        _prompt = f'Si tratta di {{subject}} \nDomanda: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nRisposta:'
    elif 'JA' in _name:
        _hint = f'単一選択肢の質問があります。この質問にはA、B、C、またはDで答えてください'
        _prompt = f'これは {{subject}} に関することです \n質問: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n回答:'
    elif 'KO' in _name:
        _hint = f'단일 선택 질문이 있습니다. A, B, C 또는 D로 답변해 주세요'
        _prompt = f'이것은 {{subject}}에 관한 것입니다 \n질문: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n답변:'
    elif 'PT' in _name:
        _hint = f'Há uma pergunta de escolha única. Responda à pergunta escolhendo A, B, C ou D'
        _prompt = f'É sobre {{subject}} \nPergunta: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nResposta:'
    elif 'ZH' in _name:
        _hint = f'这里有一个单项选择题。请通过选择 A、B、C 或 D 来回答该问题'
        _prompt = f'这是关于 {{subject}} 的内容\n问题：{{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案：'
    else:
        _hint = f'There is a single choice question. Answer the question by replying A, B, C or D'
        _prompt = f'it is about {{subject}} \nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer:'

    # Few-shot PPL模式配置
    mmmlu_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=few_shot_context + '</E>',  # 添加few-shot示例
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f'{_hint}\n {_prompt}'
                    ),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=list(range(len(examples)))),  # Few-shot检索器
        inferencer=dict(
            type=PPLInferencer,
            labels=['A', 'B', 'C', 'D'],
            max_seq_len=1024,
            batch_size=64,
        ),
    )

    mmmlu_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

    mmmlu_datasets.append(
        dict(
            abbr=f'openai_m{_name}',
            type=MMMLUDataset,
            path='data/mmmlu',
            name=_name,
            reader_cfg=mmmlu_reader_cfg,
            infer_cfg=mmmlu_infer_cfg,
            eval_cfg=mmmlu_eval_cfg,
        ))

    lang = _name.split('_')[1].split('-')[0].lower()
    mmmlu_datasets_map[lang] = dict(
        abbr=f'openai_m{_name}',
        type=MMMLUDataset,
        path='data/mmmlu',
        name=_name,
        reader_cfg=mmmlu_reader_cfg,
        infer_cfg=mmmlu_infer_cfg,
        eval_cfg=mmmlu_eval_cfg,
    )

del _name, _hint, _prompt