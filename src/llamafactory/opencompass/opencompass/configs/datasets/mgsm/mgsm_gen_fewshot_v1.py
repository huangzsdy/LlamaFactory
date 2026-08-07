from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, DebugGenInferencer
from opencompass.datasets import MGSMSDataset, MGSM_Evaluator, mgsm_postprocess

# ALL_LANGUAGES = ['en', 'zh', 'ja', 'fr', 'de', 'es', 'ko', 'ru', 'pt', 'it', 'ar', 'hi', 'bn', 'te', 'th', 'sw']
ALL_LANGUAGES = [
'bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh'
]
# 完整的多语言 few-shot 示例
MULTILINGUAL_FEW_SHOT_EXAMPLES = {
    'zh': [
        {
            "question": "花园里有15棵树。工人们今天会再种5棵，明天再种3棵。总共有多少棵树？",
            "answer": "目前有15棵树。今天他们会再种5棵，所以 15 + 5 = 20 棵树。明天他们会再种3棵，所以 20 + 3 = 23 棵树。答案是 23"
        },
        {
            "question": "一个农民有12只羊和8只山羊。他又买了5只羊和2只山羊。他现在有多少只动物？",
            "answer": "最初，农民有 12只羊 + 8只山羊 = 20只动物。他买了5只羊，所以羊变成 12 + 5 = 17。他买了2只山羊，所以山羊变成 8 + 2 = 10。总动物数 = 17 + 10 = 27。答案是 27"
        }
    ],
    'ja': [
        {
            "question": "庭には15本の木があります。労働者は今日さらに5本、明日さらに3本の木を植えます。木は全部で何本になりますか？",
            "answer": "現在15本の木があります。今日はさらに5本植えるので、15 + 5 = 20本の木です。明日はさらに3本植えるので、20 + 3 = 23本の木です。答えは23です"
        },
        {
            "question": "農家には12匹の羊と8匹の山羊がいます。彼はさらに5匹の羊と2匹の山羊を買います。今、彼は何匹の動物を持っていますか？",
            "answer": "最初、農家は 12匹の羊 + 8匹の山羊 = 20匹の動物を持っています。彼は5匹の羊を買ったので、羊は 12 + 5 = 17になります。彼は2匹の山羊を買ったので、山羊は 8 + 2 = 10になります。総動物数 = 17 + 10 = 27。答えは27です"
        }
    ],
    'fr': [
        {
            "question": "Il y a 15 arbres dans le jardin. Les ouvriers planteront 5 arbres de plus aujourd'hui et 3 de plus demain. Combien d'arbres y aura-t-il en tout ?",
            "answer": "Il y a actuellement 15 arbres. Ils en planteront 5 de plus aujourd'hui, donc 15 + 5 = 20 arbres. Puis ils en planteront 3 de plus demain, donc 20 + 3 = 23 arbres. La réponse est 23"
        },
        {
            "question": "Un fermier a 12 moutons et 8 chèvres. Il achète 5 moutons de plus et 2 chèvres de plus. Combien d'animaux a-t-il maintenant ?",
            "answer": "Initialement, le fermier a 12 moutons + 8 chèvres = 20 animaux. Il achète 5 moutons de plus, donc les moutons deviennent 12 + 5 = 17. Il achète 2 chèvres de plus, donc les chèvres deviennent 8 + 2 = 10. Nombre total d'animaux = 17 + 10 = 27. La réponse est 27"
        }
    ],
    'de': [
        {
            "question": "Es gibt 15 Bäume im Garten. Die Arbeiter werden heute 5 weitere Bäume pflanzen und morgen 3 weitere. Wie viele Bäume wird es insgesamt geben?",
            "answer": "Derzeit gibt es 15 Bäume. Sie werden heute 5 weitere pflanzen, also 15 + 5 = 20 Bäume. Dann werden sie morgen 3 weitere pflanzen, also 20 + 3 = 23 Bäume. Die Antwort ist 23"
        },
        {
            "question": "Ein Bauer hat 12 Schafe und 8 Ziegen. Er kauft 5 weitere Schafe und 2 weitere Ziegen. Wie viele Tiere hat er jetzt?",
            "answer": "Ursprünglich hat der Bauer 12 Schafe + 8 Ziegen = 20 Tiere. Er kauft 5 weitere Schafe, also werden die Schafe 12 + 5 = 17. Er kauft 2 weitere Ziegen, also werden die Ziegen 8 + 2 = 10. Gesamtzahl der Tiere = 17 + 10 = 27. Die Antwort ist 27"
        }
    ],
    'es': [
        {
            "question": "Hay 15 árboles en el jardín. Los trabajadores plantarán 5 árboles más hoy y 3 más mañana. ¿Cuántos árboles habrá en total?",
            "answer": "Actualmente hay 15 árboles. Plantarán 5 más hoy, así que 15 + 5 = 20 árboles. Luego plantarán 3 más mañana, así que 20 + 3 = 23 árboles. La respuesta es 23"
        },
        {
            "question": "Un granjero tiene 12 ovejas y 8 cabras. Compra 5 ovejas más y 2 cabras más. ¿Cuántos animales tiene ahora?",
            "answer": "Inicialmente, el granjero tiene 12 ovejas + 8 cabras = 20 animales. Compra 5 ovejas más, por lo que las ovejas se convierten en 12 + 5 = 17. Compra 2 cabras más, por lo que las cabras se convierten en 8 + 2 = 10. Número total de animales = 17 + 10 = 27. La respuesta es 27"
        }
    ],
    'en': [
        {
            "question": "There are 15 trees in the garden. Workers will plant 5 more trees today and 3 more trees tomorrow. How many trees will there be in total?",
            "answer": "Currently there are 15 trees. They will plant 5 more today, so 15 + 5 = 20 trees. Then they will plant 3 more tomorrow, so 20 + 3 = 23 trees. The answer is 23"
        },
        {
            "question": "A farmer has 12 sheep and 8 goats. He buys 5 more sheep and 2 more goats. How many animals does he have now?",
            "answer": "Initially, the farmer has 12 sheep + 8 goats = 20 animals. He buys 5 more sheep, so sheep become 12 + 5 = 17. He buys 2 more goats, so goats become 8 + 2 = 10. Total animals = 17 + 10 = 27. The answer is 27"
        }
    ],
    'ko': [  # 韩语
        {
            "question": "정원에 나무가 15그루 있습니다. 작업자들은 오늘 5그루를 더 심고 내일 3그루를 더 심을 예정입니다. 총 몇 그루의 나무가 있게 됩니까?",
            "answer": "현재 15그루의 나무가 있습니다. 오늘 5그루를 더 심을 예정이므로 15 + 5 = 20그루입니다. 내일 3그루를 더 심을 예정이므로 20 + 3 = 23그루입니다. 정답은 23입니다"
        },
        {
            "question": "한 농부가 양 12마리와 염소 8마리를 가지고 있습니다. 그는 양 5마리와 염소 2마리를 더 삽니다. 지금 그는 몇 마리의 동물을 가지고 있습니까?",
            "answer": "처음에 농부는 양 12마리 + 염소 8마리 = 20마리의 동물을 가지고 있습니다. 그는 양 5마리를 더 샀으므로, 양은 12 + 5 = 17마리가 됩니다. 그는 염소 2마리를 더 샀으므로, 염소는 8 + 2 = 10마리가 됩니다. 총 동물 수 = 17 + 10 = 27마리입니다. 정답은 27입니다"
        }
    ],
    'ru': [  # 俄语
        {
            "question": "В саду 15 деревьев. Рабочие посадят сегодня еще 5 деревьев и завтра еще 3. Сколько всего будет деревьев?",
            "answer": "В настоящее время 15 деревьев. Сегодня посадят еще 5, значит 15 + 5 = 20 деревьев. Завтра посадят еще 3, значит 20 + 3 = 23 дерева. Ответ: 23"
        },
        {
            "question": "У фермера 12 овец и 8 коз. Он покупает еще 5 овец и еще 2 козы. Сколько животных у него теперь?",
            "answer": "Первоначально у фермера 12 овец + 8 коз = 20 животных. Он покупает еще 5 овец, значит овец становится 12 + 5 = 17. Он покупает еще 2 козы, значит коз становится 8 + 2 = 10. Общее количество животных = 17 + 10 = 27. Ответ: 27"
        }
    ],
    'pt': [  # 葡萄牙语
        {
            "question": "Há 15 árvores no jardim. Os trabalhadores plantarão mais 5 árvores hoje e mais 3 amanhã. Quantas árvores haverá no total?",
            "answer": "Atualmente há 15 árvores. Eles plantarão mais 5 hoje, então 15 + 5 = 20 árvores. Depois plantarão mais 3 amanhã, então 20 + 3 = 23 árvores. A resposta é 23"
        },
        {
            "question": "Um fazendeiro tem 12 ovelhas e 8 cabras. Ele compra mais 5 ovelhas e mais 2 cabras. Quantos animais ele tem agora?",
            "answer": "Inicialmente, o fazendeiro tem 12 ovelhas + 8 cabras = 20 animais. Ele compra mais 5 ovelhas, então as ovelhas se tornam 12 + 5 = 17. Ele compra mais 2 cabras, então as cabras se tornam 8 + 2 = 10. Número total de animais = 17 + 10 = 27. A resposta é 27"
        }
    ],
    'it': [  # 意大利语
        {
            "question": "Ci sono 15 alberi nel giardino. I lavoratori pianteranno 5 alberi in più oggi e 3 in più domani. Quanti alberi ci saranno in totale?",
            "answer": "Attualmente ci sono 15 alberi. Ne pianteranno 5 in più oggi, quindi 15 + 5 = 20 alberi. Poi ne pianteranno 3 in più domani, quindi 20 + 3 = 23 alberi. La risposta è 23"
        },
        {
            "question": "Un contadino ha 12 pecore e 8 capre. Compra altre 5 pecore e altre 2 capre. Quanti animali ha ora?",
            "answer": "Inizialmente, il contadino ha 12 pecore + 8 capre = 20 animali. Compra altre 5 pecore, quindi le pecore diventano 12 + 5 = 17. Compra altre 2 capre, quindi le capre diventano 8 + 2 = 10. Numero totale di animali = 17 + 10 = 27. La risposta è 27"
        }
    ],
    'ar': [  # 阿拉伯语
        {
            "question": "هناك 15 شجرة في الحديقة. سيزرع العمال 5 أشجار إضافية اليوم و 3 أشجار إضافية غدًا. كم عدد الأشجار التي ستكون في المجموع؟",
            "answer": "يوجد حالياً 15 شجرة. سيزرعون 5 أشجار إضافية اليوم، لذا 15 + 5 = 20 شجرة. ثم سيزرعون 3 أشجار إضافية غدًا، لذا 20 + 3 = 23 شجرة. الجواب هو 23"
        },
        {
            "question": "لدى مزارع 12 خروفًا و 8 ماعز. يشتري 5 خرفان إضافية و 2 معزاة إضافيتان. كم عدد الحيوانات التي لديه الآن؟",
            "answer": "في البداية، لدى المزارع 12 خروفًا + 8 ماعز = 20 حيوانًا. يشتري 5 خرفان إضافية، لذا أصبح عدد الخرفان 12 + 5 = 17. يشتري 2 معزاة إضافيتان، لذا أصبح عدد الماعز 8 + 2 = 10. إجمالي عدد الحيوانات = 17 + 10 = 27. الجواب هو 27"
        }
    ],
    'hi': [  # 印地语
        {
            "question": "बगीचे में 15 पेड़ हैं। कामगार आज 5 और पेड़ लगाएंगे और कल 3 और पेड़ लगाएंगे। कुल कितने पेड़ होंगे?",
            "answer": "वर्तमान में 15 पेड़ हैं। वे आज 5 और लगाएंगे, इसलिए 15 + 5 = 20 पेड़। फिर वे कल 3 और लगाएंगे, इसलिए 20 + 3 = 23 पेड़। उत्तर 23 है"
        },
        {
            "question": "एक किसान के पास 12 भेड़ें और 8 बकरियां हैं। वह 5 और भेड़ें और 2 और बकरियां खरीदता है। अब उसके पास कितने जानवर हैं?",
            "answer": "प्रारंभ में, किसान के पास 12 भेड़ें + 8 बकरियां = 20 जानवर हैं। वह 5 और भेड़ें खरीदता है, इसलिए भेड़ें 12 + 5 = 17 हो जाती हैं। वह 2 और बकरियां खरीदता है, इसलिए बकरियां 8 + 2 = 10 हो जाती हैं। कुल जानवर = 17 + 10 = 27। उत्तर 27 है"
        }
    ],
    'bn': [  # 孟加拉语
        {
            "question": "বাগানে 15টি গাছ আছে। শ্রমিকেরা আজ আরও 5টি গাছ রোপণ করবে এবং আগামীকাল আরও 3টি গাছ রোপণ করবে। মোট কতগুলি গাছ থাকবে?",
            "answer": "বর্তমানে 15টি গাছ আছে। তারা আজ আরও 5টি রোপণ করবে, তাই 15 + 5 = 20টি গাছ। তারপর তারা আগামীকাল আরও 3টি রোপণ করবে, তাই 20 + 3 = 23টি গাছ। উত্তর হল 23"
        },
        {
            "question": "এক কৃষকের 12টি ভেড়া ও 8টি ছাগল আছে। সে আরও 5টি ভেড়া ও 2টি ছাগল কিনছে। তার এখন কতগুলি প্রাণী আছে?",
            "answer": "প্রাথমিকভাবে, কৃষকের 12টি ভেড়া + 8টি ছাগল = 20টি প্রাণী আছে। সে আরও 5টি ভেড়া কিনছে, তাই ভেড়া হয়ে যায় 12 + 5 = 17। সে আরও 2টি ছাগল কিনছে, তাই ছাগল হয়ে যায় 8 + 2 = 10। মোট প্রাণী = 17 + 10 = 27। উত্তর হল 27"
        }
    ],
    'te': [  # 泰卢固语
        {
            "question": "తోటలో 15 చెట్లు ఉన్నాయి. కార్మికులు ఈరోజు మరో 5 చెట్లు, రేపు మరో 3 చెట్లు నాటుతారు. మొత్తం ఎన్ని చెట్లు ఉంటాయి?",
            "answer": "ప్రస్తుతం 15 చెట్లు ఉన్నాయి. వారు ఈరోజు మరో 5 చెట్లు నాటుతారు, కాబట్టి 15 + 5 = 20 చెట్లు. తర్వాత వారు రేపు మరో 3 చెట్లు నాటుతారు, కాబట్టి 20 + 3 = 23 చెట్లు. సమాధానం 23"
        },
        {
            "question": "ఒక రైతుకు 12 గొర్రెలు మరియు 8 మేకలు ఉన్నాయి. అతను మరో 5 గొర్రెలు మరియు మరో 2 మేకలు కొంటాడు. అతనికి ఇప్పుడు ఎన్ని జంతువులు ఉన్నాయి?",
            "answer": "ప్రారంభంలో, రైతుకు 12 గొర్రెలు + 8 మేకలు = 20 జంతువులు ఉన్నాయి. అతను మరో 5 గొర్రెలు కొంటాడు, కాబట్టి గొర్రెలు 12 + 5 = 17 అవుతాయి. అతను మరో 2 మేకలు కొంటాడు, కాబట్టి మేకలు 8 + 2 = 10 అవుతాయి. మొత్తం జంతువులు = 17 + 10 = 27. సమాధానం 27"
        }
    ],
    'th': [  # 泰语
        {
            "question": "ในสวนมีต้นไม้ 15 ต้น คนงานจะปลูกต้นไม้อีก 5 ต้นวันนี้ และอีก 3 ต้นในวันพรุ่งนี้ จะมีต้นไม้ทั้งหมดกี่ต้น?",
            "answer": "ปัจจุบันมีต้นไม้ 15 ต้น วันนี้จะปลูกเพิ่มอีก 5 ต้น ดังนั้น 15 + 5 = 20 ต้น พรุ่งนี้จะปลูกเพิ่มอีก 3 ต้น ดังนั้น 20 + 3 = 23 ต้น คำตอบคือ 23"
        },
        {
            "question": "ชาวนามีแกะ 12 ตัวและแพะ 8 ตัว เขาซื้อแกะเพิ่มอีก 5 ตัวและแพะเพิ่มอีก 2 ตัว ตอนนี้เขามีสัตว์ทั้งหมดกี่ตัว?",
            "answer": "เริ่มแรกชาวนามีแกะ 12 ตัว + แพะ 8 ตัว = 20 ตัว เขาซื้อแกะเพิ่มอีก 5 ตัว ดังนั้นแกะเป็น 12 + 5 = 17 ตัว เขาซื้อแพะเพิ่มอีก 2 ตัว ดังนั้นแพะเป็น 8 + 2 = 10 ตัว จำนวนสัตว์ทั้งหมด = 17 + 10 = 27 ตัว คำตอบคือ 27"
        }
    ],
    'sw': [  # 斯瓦希里语
        {
            "question": "Kuna miti 15 bustanini. Wafanyikazi watapanda miti 5 zaidi leo na miti 3 zaidi kesho. Itakuwa na miti ngapi kwa jumla?",
            "answer": "Kwa sasa kuna miti 15. Watapanda 5 zaidi leo, kwa hiyo 15 + 5 = 20 miti. Kisha watapanda 3 zaidi kesho, kwa hiyo 20 + 3 = 23 miti. Jibu ni 23"
        },
        {
            "question": "Mkulima ana kondoo 12 na mbuzi 8. Ananunua kondoo 5 zaidi na mbuzi 2 zaidi. Ana wanyama wangapi sasa?",
            "answer": "Awali, mkulima ana kondoo 12 + mbuzi 8 = wanyama 20. Ananunua kondoo 5 zaidi, kwa hiyo kondoo huwa 12 + 5 = 17. Ananunua mbuzi 2 zaidi, kwa hiyo mbuzi huwa 8 + 2 = 10. Jumla ya wanyama = 17 + 10 = 27. Jibu ni 27"
        }
    ]
}


# 为缺失的语言提供英语回退
def get_few_shot_examples(lang):
    """获取指定语言的few-shot示例，如果不存在则返回英语示例"""
    if lang in MULTILINGUAL_FEW_SHOT_EXAMPLES:
        return MULTILINGUAL_FEW_SHOT_EXAMPLES[lang]
    else:
        print(f"warning: 语言 {lang} 没有找到few-shot示例，使用英语示例作为回退")
        return MULTILINGUAL_FEW_SHOT_EXAMPLES['en']


# 创建数据集配置
mgsm_datasets = []
mgsm_datasets_map =  dict()
print(f"---------------------------{__file__}: ALL_LANGUAGES: {ALL_LANGUAGES}")


for lang in ALL_LANGUAGES:
    # 获取当前语言的few-shot示例
    few_shot_examples = get_few_shot_examples(lang)

    # 创建prompt模板
    rounds = []

    # 添加few-shot示例
    for example in few_shot_examples:
        rounds.append(dict(role='HUMAN', prompt=f"Question: {example['question']}"))
        rounds.append(dict(role='BOT', prompt=example['answer']))

    # 添加当前问题
    rounds.append(dict(role='HUMAN', prompt="Question: {question}"))

    prompt_template = dict(round=rounds)

    # 创建数据集配置
    mgsm_reader_cfg = dict(
        input_columns=['question'],
        output_column='answer',
    )

    mgsm_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=prompt_template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            # type=GenInferencer,
            type=DebugGenInferencer,
            batch_size=64,
            max_out_len=128,
        ),
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