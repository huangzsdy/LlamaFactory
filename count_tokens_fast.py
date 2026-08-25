#!/usr/bin/env python3
"""
快速统计 Token 数量
使用多进程加速
"""

import json
import os
import sys
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer

def count_tokens_in_line(args):
    """统计单行的token数量"""
    line, model_name = args
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    try:
        data = json.loads(line.strip())
        text = data.get('text', '')
        tokens = tokenizer.encode(text)
        return len(tokens)
    except:
        return 0

def count_tokens_fast(file_path, model_name='Qwen/Qwen2.5-7B-Instruct', num_workers=None):
    """快速统计token数量"""
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    # 读取所有行
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总共 {len(lines)} 行，使用 {num_workers} 个进程...")
    
    # 并行处理
    args_list = [(line, model_name) for line in lines]
    
    with Pool(num_workers) as pool:
        token_counts = pool.map(count_tokens_in_line, args_list)
    
    total_tokens = sum(token_counts)
    return total_tokens

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', required=True, help='要统计的jsonl文件')
    parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-7B-Instruct', help='模型名称')
    parser.add_argument('--workers', '-w', type=int, default=None, help='进程数')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"文件不存在: {args.file}")
        sys.exit(1)
    
    print(f"统计文件: {args.file}")
    print(f"使用模型: {args.model}")
    
    total = count_tokens_fast(args.file, args.model, args.workers)
    
    print(f"\n结果:")
    print(f"  总 Token 数: {total:,}")
    print(f"  总 Token 数: {total / 1e9:.2f} B")
    print(f"  总 Token 数: {total / 1e6:.2f} M")
