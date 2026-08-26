"""评测 Qwen2.5-7B Instruct 模型的配置"""
from mmengine.config import read_base

with read_base():
    # 选择评测数据集（基础核心数据集）
    from opencompass.configs.datasets.collections.base_core import datasets
    # 使用 Qwen2.5-7B 模型配置
    from opencompass.configs.models.qwen2_5.hf_qwen2_5_7b_instruct import models

# 使用 HuggingFace 上的 Qwen2.5-7B-Instruct 模型
models[0]['path'] = 'Qwen/Qwen2.5-7B-Instruct'
models[0]['abbr'] = 'qwen2.5-7b-instruct-hf'

# 批量大小，根据显存调整
models[0]['batch_size'] = 4

# 评测输出目录
work_dir = 'outputs/qwen2.5-7b-instruct'
