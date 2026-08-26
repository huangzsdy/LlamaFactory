#!/usr/bin/env python3
"""将 xlsx 文件转换为 jsonl 格式"""
import pandas as pd
import json
import os

# 转换 abuse_qa.xlsx
print("转换 abuse_qa.xlsx...")
df = pd.read_excel('./datasets/abuse_qa.xlsx')
print("列名:", df.columns.tolist())
print("前3行:")
print(df.head(3))

# 保存为 jsonl
output_file = './datasets/abuse_qa.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        # 将每行转换为字典
        item = {}
        for col in df.columns:
            item[col] = row[col]
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"已保存到: {output_file}")

# 转换 military_mcq.xlsx
print("\n转换 military_mcq.xlsx...")
df = pd.read_excel('./datasets/military_mcq.xlsx')
print("列名:", df.columns.tolist())
print("前3行:")
print(df.head(3))

# 保存为 jsonl
output_file = './datasets/military_mcq.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        # 将每行转换为字典
        item = {}
        for col in df.columns:
            item[col] = row[col]
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"已保存到: {output_file}")
print("\n转换完成!")
