from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.datasets import MGSMSDataset, MGSM_Evaluator, mgsm_postprocess

ALL_LANGUAGES = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']

# 为每种语言定义 few-shot 示例
FEW_SHOT_EXAMPLES = {
    'en': [
        {
            'question': 'There are 3 apples. Mary gives 2 apples to John. How many apples does Mary have left?',
            'answer': 'Mary started with 3 apples. She gave away 2 apples. So 3 - 2 = 1. Answer: 1'
        },
        {
            'question': 'A book has 120 pages. If Sarah reads 15 pages each day, how many days will it take her to finish the book?',
            'answer': 'The book has 120 pages. Each day she reads 15 pages. So 120 ÷ 15 = 8. Answer: 8'
        }
    ],
    'bn': [
        {
            'question': '৩টি আপেল আছে। মেরি জনকে ২টি আপেল দেয়। মেরির কতগুলি আপেল বাকি আছে?',
            'answer': 'মেরির কাছে শুরুতে ৩টি আপেল ছিল। সে ২টি আপেল দিয়েছে। তাই ৩ - ২ = ১। উত্তর: ১'
        },
        {
            'question': 'একটি বইয়ে ১২০ পাতা আছে। যদি সারাহ প্রতিদিন ১৫ পাতা পড়ে, তাহলে বইটি শেষ করতে তার কত দিন লাগবে?',
            'answer': 'বইটিতে ১২০ পাতা আছে। প্রতিদিন সে ১৫ পাতা পড়ে। তাই ১২০ ÷ ১৫ = ৮। উত্তর: ৮'
        }
    ],
    'de': [
        {
            'question': 'Es gibt 3 Äpfel. Maria gibt 2 Äpfel an Johannes. Wie viele Äpfel hat Maria noch?',
            'answer': 'Maria hatte am Anfang 3 Äpfel. Sie hat 2 Äpfel weggegeben. Also 3 - 2 = 1. Antwort: 1'
        },
        {
            'question': 'Ein Buch hat 120 Seiten. Wenn Sarah jeden Tag 15 Seiten liest, wie viele Tage wird sie brauchen, um das Buch zu beenden?',
            'answer': 'Das Buch hat 120 Seiten. Jeden Tag liest sie 15 Seiten. Also 120 ÷ 15 = 8. Antwort: 8'
        }
    ],
    'es': [
        {
            'question': 'Hay 3 manzanas. María le da 2 manzanas a Juan. ¿Cuántas manzanas le quedan a María?',
            'answer': 'María comenzó con 3 manzanas. Ella dio 2 manzanas. Entonces 3 - 2 = 1. Respuesta: 1'
        },
        {
            'question': 'Un libro tiene 120 páginas. Si Sarah lee 15 páginas cada día, ¿cuántos días le tomará terminar el libro?',
            'answer': 'El libro tiene 120 páginas. Cada día ella lee 15 páginas. Entonces 120 ÷ 15 = 8. Respuesta: 8'
        }
    ],
    'fr': [
        {
            'question': 'Il y a 3 pommes. Marie donne 2 pommes à Jean. Combien de pommes reste-t-il à Marie?',
            'answer': 'Marie avait 3 pommes au début. Elle a donné 2 pommes. Donc 3 - 2 = 1. Réponse: 1'
        },
        {
            'question': 'Un livre a 120 pages. Si Sarah lit 15 pages chaque jour, combien de jours lui faudra-t-il pour finir le livre?',
            'answer': 'Le livre a 120 pages. Chaque jour, elle lit 15 pages. Donc 120 ÷ 15 = 8. Réponse: 8'
        }
    ],
    'ja': [
        {
            'question': 'リンゴが3個あります。メアリーはジョンにリンゴを2個あげました。メアリーにはリンゴが何個残っていますか？',
            'answer': 'メアリーは最初に3個のリンゴを持っていました。2個のリンゴをあげました。なので、3 - 2 = 1。答え: 1'
        },
        {
            'question': '本は120ページあります。サラが毎日15ページ読むとしたら、本を読み終えるのに何日かかりますか？',
            'answer': '本は120ページあります。毎日15ページ読むので、120 ÷ 15 = 8。答え: 8'
        }
    ],
    'ru': [
        {
            'question': 'Есть 3 яблока. Мария дает 2 яблока Джону. Сколько яблок осталось у Марии?',
            'answer': 'У Марии изначально было 3 яблока. Она отдала 2 яблока. Значит 3 - 2 = 1. Ответ: 1'
        },
        {
            'question': 'В книге 120 страниц. Если Сара читает по 15 страниц каждый день, сколько дней ей потребуется, чтобы закончить книгу?',
            'answer': 'В книге 120 страниц. Каждый день она читает 15 страниц. Значит 120 ÷ 15 = 8. Ответ: 8'
        }
    ],
    'sw': [
        {
            'question': 'Kuna maapulo 3. Maria anampa John maapulo 2. Je, Maria ana maapulo mangapi yaliyobaki?',
            'answer': 'Maria alianza na maapulo 3. Alitoa maapulo 2. Kwa hiyo 3 - 2 = 1. Jibu: 1'
        },
        {
            'question': 'Kitabu kina kurasa 120. Ikiwa Sarah anasoma kurasa 15 kila siku, itamchukua siku ngapi kumaliza kitabu?',
            'answer': 'Kitabu kina kurasa 120. Kila siku anasoma kurasa 15. Kwa hiyo 120 ÷ 15 = 8. Jibu: 8'
        }
    ],
    'te': [
        {
            'question': '3 ఆపిల్లు ఉన్నాయి. మేరీ జాన్కు 2 ఆపిల్లు ఇస్తుంది. మేరీ వద్ద ఎన్ని ఆపిల్లు మిగిలి ఉన్నాయి?',
            'answer': 'మేరీ వద్ద ప్రారంభంలో 3 ఆపిల్లు ఉన్నాయి. ఆమె 2 ఆపిల్లు ఇచ్చింది. కాబట్టి 3 - 2 = 1. సమాధానం: 1'
        },
        {
            'question': 'ఒక పుస్తకంలో 120 పేజీలు ఉన్నాయి. సారా రోజుకు 15 పేజీలు చదివితే, పుస్తకం పూర్తి చేయడానికి ఎన్ని రోజులు పడుతుంది?',
            'answer': 'పుస్తకంలో 120 పేజీలు ఉన్నాయి. ప్రతి రోజు ఆమె 15 పేజీలు చదువుతుంది. కాబట్టి 120 ÷ 15 = 8. సమాధానం: 8'
        }
    ],
    'th': [
        {
            'question': 'มีแอปเปิ้ล 3 ลูก แมรีให้แอปเปิ้ล 2 ลูกกับจอห์น แมรีมีแอปเปิ้ลเหลือกี่ลูก?',
            'answer': 'แมรีเริ่มต้นมีแอปเปิ้ล 3 ลูก เธอให้แอปเปิ้ลไป 2 ลูก ดังนั้น 3 - 2 = 1 คำตอบ: 1'
        },
        {
            'question': 'หนังสือมี 120 หน้า ถ้าซาร่าอ่านวันละ 15 หน้า เธอจะใช้เวลากี่วันจึงจะอ่านหนังสือจบ?',
            'answer': 'หนังสือมี 120 หน้า ทุกวันเธออ่าน 15 หน้า ดังนั้น 120 ÷ 15 = 8 คำตอบ: 8'
        }
    ],
    'zh': [
        {
            'question': '有3个苹果。玛丽给约翰2个苹果。玛丽还剩下几个苹果？',
            'answer': '玛丽一开始有3个苹果。她给出了2个苹果。所以3 - 2 = 1。答案: 1'
        },
        {
            'question': '一本书有120页。如果莎拉每天读15页，她需要多少天才能读完这本书？',
            'answer': '这本书有120页。她每天读15页。所以120 ÷ 15 = 8。答案: 8'
        }
    ]
}

LANG_TO_INSTRUCTIONS = {
    'en': """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".\n\n{question}""",
    'bn': """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.\n\n{question}""",
    'de': """Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.\n\n{question}""",
    'es': """Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".\n\n{question}""",
    'fr': """Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N'ajoutez rien d'autre que la réponse entière après "Réponse:".\n\n{question}""",
    'ja': """の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。\n\n{question}""",
    'ru': """Решите эту математическую задачу. Объясните шаги рассуждения перед тем, как дать окончательный ответ в последней строке сам по себе в формате "Ответ:". Не добавляйте ничего, кроме целочисленного ответа после "Ответ:".\n\n{question}""",
    'sw': """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".\n\n{question}""",
    'te': """ఈఈ గణిత సమస్యను పరిషష్కరించండడి. చివవరి సమాధధానాన్ని ఇఇవ్వదానికి ముందదు తర్కాతత్మక అదుగులను ఇఇవ్వండడి. చివవరి పంకక్తిలలో మాతత్రమే 'సమాధధానం:' అనే ఆకారంలలో చివవరి సマాధధానాదద్ని ఇఇవ్వండడి సమాధధానం: తర్వాతత పూరర్ణణాంంకక సమాధధానానికి తప్పించి ఎఎదేనా చేర్చవద్దదు.\n\n{question}""",
    'th': """แก้ปัญหาคณิตศาสตร์นี้ ให้ให้ขั้นตอนการใช้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูปแบบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอกจากคำตอบที่เป็นจำนวนเต็มหลังจาก "คำตอบ:"\n\n{question}""",
    'zh': """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。\n\n{question}""",
}

mgsm_datasets = []
mgsm_datasets_map = dict()

def create_few_shot_prompt_template(lang):
    """创建 few-shot 提示模板"""
    rounds = []
    
    # 添加 few-shot 示例
    for example in FEW_SHOT_EXAMPLES[lang]:
        rounds.append(dict(role='HUMAN', prompt=example['question']))
        rounds.append(dict(role='BOT', prompt=example['answer']))
    
    # 添加当前问题
    rounds.append(dict(role='HUMAN', prompt=LANG_TO_INSTRUCTIONS[lang]))
    
    return dict(
        type=PromptTemplate,
        template=dict(round=rounds),
    )

for lang in ALL_LANGUAGES:
    mgsm_reader_cfg = dict(input_columns=['question'], output_column='answer')

    mgsm_infer_cfg = dict(
        prompt_template=create_few_shot_prompt_template(lang),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=512, batch_size=1),
    )

    mgsm_eval_cfg = dict(
        evaluator=dict(type=MGSM_Evaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=mgsm_postprocess, lang=lang),
    )

    mgsm_datasets.append(
        dict(
            type=MGSMSDataset,
            abbr=f'mgsm_{lang}',
            path=f'data/mgsm/mgsm_{lang}.tsv',
            reader_cfg=mgsm_reader_cfg,
            infer_cfg=mgsm_infer_cfg,
            eval_cfg=mgsm_eval_cfg,
        )
    )
    mgsm_datasets_map[lang] = \
        dict(
            type=MGSMSDataset,
            abbr=f'mgsm_{lang}',
            path=f'data/mgsm/mgsm_{lang}.tsv',
            reader_cfg=mgsm_reader_cfg,
            infer_cfg=mgsm_infer_cfg,
            eval_cfg=mgsm_eval_cfg,
        )