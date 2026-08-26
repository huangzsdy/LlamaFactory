#!/usr/bin/env python3
"""
简化的评测脚本 - 避免导入所有opencompass模块
只导入必要的模块来运行自定义数据集评测
"""

import os
import sys
import glob

# 设置工作目录
os.chdir('/mnt/c/Users/ThinkPad/my_own_files/study/llm/LlamaFactory-main/LlamaFactory-main/opencompass')

# 添加opencompass到路径
sys.path.insert(0, '/mnt/c/Users/ThinkPad/my_own_files/study/llm/LlamaFactory-main/LlamaFactory-main/opencompass')

# 只导入需要的模块，避免导入bigcodebench等有问题的模块
from mmengine.config import Config
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

def main():
    # 数据集目录
    datasets_dir = './datasets'
    
    # 获取所有数据文件
    jsonl_files = sorted(glob.glob(os.path.join(datasets_dir, '*.jsonl')))
    xlsx_files = sorted(glob.glob(os.path.join(datasets_dir, '*.xlsx')))
    data_files = jsonl_files + xlsx_files
    
    print(f"找到 {len(data_files)} 个数据文件:")
    for f in data_files:
        print(f"  - {os.path.basename(f)}")
    
    if not data_files:
        print("未找到任何数据文件!")
        return
    
    # 创建简单的配置
    # 使用命令行方式运行
    model_path = '/mnt/c/Users/ThinkPad/Downloads/copy_models/Qwen/Qwen2.5-7B-Instruct'
    
    # 为每个数据文件创建配置并运行
    for data_file in data_files:
        filename = os.path.basename(data_file)
        dataset_name = os.path.splitext(filename)[0]
        
        print(f"\n{'='*60}")
        print(f"正在评测: {filename}")
        print(f"{'='*60}")
        
        # 创建临时配置文件
        config_content = f"""
# 数据集配置
datasets = [
    dict(
        abbr='{dataset_name[:20]}',
        path='{data_file}',
        reader_cfg=dict(
            input_columns=['question'],
            output_column='answer',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='{{question}}',
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    ),
]

# 模型配置
models = [
    dict(
        type='HuggingFace',
        path='{model_path}',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_seq_len=4096,
        max_out_len=512,
        batch_size=4,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

# 输出目录
work_dir = './outputs/{dataset_name[:20]}'
"""
        
        # 写入临时配置文件
        config_file = f'./temp_config_{dataset_name[:20]}.py'
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # 运行评测
        os.system(f'cd /mnt/c/Users/ThinkPad/my_own_files/study/llm/LlamaFactory-main/LlamaFactory-main/opencompass && python3 run.py {config_file} 2>&1')
        
        # 删除临时配置文件
        # os.remove(config_file)
    
    print("\n" + "="*60)
    print("所有评测完成!")
    print("="*60)

if __name__ == '__main__':
    main()
