#!/usr/bin/env python3
"""测试数据集扫描功能"""
import os
import glob

datasets_dir = './datasets'

jsonl_files = sorted(glob.glob(os.path.join(datasets_dir, '*.jsonl')))
xlsx_files = sorted(glob.glob(os.path.join(datasets_dir, '*.xlsx')))

data_files = jsonl_files + xlsx_files

print(f'找到 {len(data_files)} 个数据文件:')
for f in data_files:
    print(f'  - {os.path.basename(f)}')

datasets = []
for data_file in data_files:
    filename = os.path.basename(data_file)
    dataset_name = os.path.splitext(filename)[0]
    abbr_name = dataset_name[:20]
    print(f'生成数据集配置: {abbr_name}')
    datasets.append({'abbr': abbr_name, 'path': data_file})

print(f'\n生成了 {len(datasets)} 个数据集配置')
