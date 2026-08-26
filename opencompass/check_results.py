#!/usr/bin/env python3
import json

# 读取已有的结果
results = {}

# 知识理解
with open('./outputs/知识理解_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    correct = sum(1 for x in data if x.get('correct', False))
    results['知识理解'] = {'correct': correct, 'total': len(data)}

# 逻辑推理
with open('./outputs/逻辑推理_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    correct = sum(1 for x in data if x.get('correct', False))
    results['逻辑推理'] = {'correct': correct, 'total': len(data)}

# 长程依赖
with open('./outputs/长程依赖_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    correct = sum(1 for x in data if x.get('correct', False))
    results['长程依赖'] = {'correct': correct, 'total': len(data)}

print('=' * 50)
print('已有评测结果汇总 (使用 Qwen2.5-7B-Instruct)')
print('=' * 50)
for name, res in results.items():
    acc = res['correct'] / res['total'] * 100
    print(f'{name}: {res["correct"]}/{res["total"]} = {acc:.1f}%')

print()
total_correct = sum(r['correct'] for r in results.values())
total_all = sum(r['total'] for r in results.values())
print(f'总体准确率: {total_correct}/{total_all} = {total_correct/total_all*100:.1f}%')
