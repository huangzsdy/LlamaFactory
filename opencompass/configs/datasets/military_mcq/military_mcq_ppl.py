"""Military MCQ 数据集评测配置"""

import os
import pandas as pd
from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import LOAD_DATASET


def load_military_mcq_dataset(path: str):
    """加载 military_mcq 数据集"""
    df = pd.read_excel(path)
    
    # 处理数据
    records = []
    for _, row in df.iterrows():
        question = row['测试问题']
        options_raw = row['测试选项']
        answer = row['测试答案']
        
        # 拆分选项 (A./B./C./D. 可能用 | 或换行分隔)
        options_str = str(options_raw).replace('|', '\n')
        options = {}
        for line in options_str.split('\n'):
            line = line.strip()
            if line.startswith('A.'):
                options['A'] = line[2:].strip()
            elif line.startswith('B.'):
                options['B'] = line[2:].strip()
            elif line.startswith('C.'):
                options['C'] = line[2:].strip()
            elif line.startswith('D.'):
                options['D'] = line[2:].strip()
        
        # 构建记录
        record = {
            'question': question,
            'A': options.get('A', ''),
            'B': options.get('B', ''),
            'C': options.get('C', ''),
            'D': options.get('D', ''),
            'gold': answer.strip(),  # 答案字母
        }
        records.append(record)
    
    return Dataset.from_list(records)


@LOAD_DATASET.register_module()
class MilitaryMCQDataset(BaseDataset):
    """Military MCQ 数据集"""
    
    @staticmethod
    def load(path, **kwargs):
        return load_military_mcq_dataset(path)


# 定义评测配置
military_mcq_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='gold',
)

military_mcq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'),
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

military_mcq_eval_cfg = dict(
    evaluator=dict(type='AccEvaluator'),
)

# 数据集配置
military_mcq_datasets = dict(
    abbr='military_mcq',
    type=MilitaryMCQDataset,
    path='./datasets/military_mcq.xlsx',
    reader_cfg=military_mcq_reader_cfg,
    infer_cfg=military_mcq_infer_cfg,
    eval_cfg=military_mcq_eval_cfg,
)
