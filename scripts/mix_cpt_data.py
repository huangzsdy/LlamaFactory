#!/usr/bin/env python3
"""
CPT Data Mixing Script
将领域数据与通用语料数据按比例混合，用于CPT训练。

Usage:
    python scripts/mix_cpt_data.py \
        --domain-data data/cpt_dataset/your_domain_data.jsonl \
        --corpus-data data/cpt_dataset/your_corpus_data.jsonl \
        --output data/cpt_dataset/cpt_mixed.jsonl \
        --domain-ratio 0.6 \
        --corpus-ratio 0.4

Args:
    --domain-data: 你的领域数据文件路径 (JSONL格式)
    --corpus-data: 通用语料数据文件路径 (JSONL格式)
    --output: 混合后的输出文件路径
    --domain-ratio: 领域数据比例 (默认: 0.6)
    --corpus-ratio: 通用语料数据比例 (默认: 0.4)
    --max-samples: 最大样本数，可选，用于限制总数据量
    --seed: 随机种子 (默认: 42)
    
    # 字段提取参数:
    --domain-field: 从domain-data中提取的字段名 (默认: text)
    --corpus-field: 从corpus-data中提取的字段名 (默认: text)
    
    # 批量生成多批次数据:
    --fields text,synthesized_QA,synthesized_Wikipedia-style_rephrasing
"""

import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm


def load_jsonl(file_path: str):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {file_path}"):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {file_path}")
                continue
    return data


def extract_field(data: list, field_name: str):
    """从数据中提取指定字段，生成text格式的样本"""
    result = []
    for item in data:
        value = item.get(field_name, "")
        if value and isinstance(value, str):
            result.append({"text": value})
    return result


def save_jsonl(data: list, output_path: str):
    """保存为JSONL文件"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="Writing output"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples to {output_path}")


def mix_data(
    domain_data: list,
    fineweb_data: list,
    domain_ratio: float,
    fineweb_ratio: float,
    max_samples: int = None,
    seed: int = 42
):
    """混合两个数据集
    
    修复说明：
    - 当没有设置 max_samples 时，以领域数据为基准
    - 根据用户设置的比例，计算需要采样的通用语料数量
    - 确保最终混合比例符合用户设置的比例
    """
    random.seed(seed)
    
    # 计算目标数量
    if max_samples:
        target_domain = int(max_samples * domain_ratio)
        target_fineweb = int(max_samples * fineweb_ratio)
    else:
        # 修复：以领域数据为基准，根据比例计算需要的通用语料数量
        # 例如：领域数据 1000 条，设置 domain:corpus = 0.6:0.4
        # 则通用语料应该采样 1000 * (0.4/0.6) = 667 条
        # 这样最终混合后比例就是 60%:40%
        target_domain = len(domain_data)  # 使用全部领域数据
        target_fineweb = int(len(domain_data) * (fineweb_ratio / domain_ratio)) if domain_ratio > 0 else 0
    
    print(f"\nData mixing summary:")
    print(f"  Domain data: {len(domain_data)} -> sampling {min(target_domain, len(domain_data))}")
    print(f"  Fineweb data: {len(fineweb_data)} -> sampling {min(target_fineweb, len(fineweb_data))}")
    
    # 采样
    sampled_domain = random.sample(
        domain_data, 
        min(target_domain, len(domain_data))
    )
    sampled_fineweb = random.sample(
        fineweb_data, 
        min(target_fineweb, len(fineweb_data))
    )
    
    # 混合
    mixed_data = sampled_domain + sampled_fineweb
    random.shuffle(mixed_data)
    
    print(f"  Final mixed: {len(mixed_data)} samples")
    print(f"  Final ratio: {len(sampled_domain)/len(mixed_data)*100:.1f}% domain + {len(sampled_fineweb)/len(mixed_data)*100:.1f}% fineweb")
    
    return mixed_data


def main():
    parser = argparse.ArgumentParser(
        description="Mix domain data with Fineweb for CPT training"
    )
    parser.add_argument(
        "--domain-data",
        type=str,
        required=True,
        help="Path to your domain data JSONL file"
    )
    parser.add_argument(
        "--corpus-data",
        type=str,
        required=True,
        help="Path to general corpus data JSONL file (e.g., Fineweb, Wikipedia)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output mixed data JSONL file"
    )
    parser.add_argument(
        "--domain-ratio",
        type=float,
        default=0.6,
        help="Ratio for domain data (default: 0.6)"
    )
    parser.add_argument(
        "--corpus-ratio",
        type=float,
        default=0.4,
        help="Ratio for corpus data (default: 0.4)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total samples (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    # 新增参数：从domain-data中提取的字段名
    parser.add_argument(
        "--domain-field",
        type=str,
        default="text",
        help="Field name to extract from domain data (default: text)"
    )
    # 新增参数：从corpus-data中提取的字段名
    parser.add_argument(
        "--corpus-field",
        type=str,
        default="text",
        help="Field name to extract from corpus data (default: text)"
    )
    # 新增参数：批量生成多批次数据
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Comma-separated field names to extract from same domain-data file. "
             "If provided, will generate multiple output files. "
             "Example: --fields text,synthesized_QA,synthesized_Wikipedia-style_rephrasing"
    )
    
    args = parser.parse_args()
    
    # 验证比例
    total_ratio = args.domain_ratio + args.corpus_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio}, should be 1.0")
        args.domain_ratio /= total_ratio
        args.corpus_ratio /= total_ratio
        print(f"Normalized to: domain={args.domain_ratio}, corpus={args.corpus_ratio}")
    
    # 加载数据
    print("Loading data...")
    domain_data = load_jsonl(args.domain_data)
    corpus_data = load_jsonl(args.corpus_data)
    
    # 提取corpus字段
    extracted_corpus = extract_field(corpus_data, args.corpus_field)
    print(f"Extracted {len(extracted_corpus)} samples from corpus field '{args.corpus_field}'")
    
    # 判断是单字段还是批量处理
    if args.fields:
        # 批量处理：从同一文件提取多个字段
        field_list = [f.strip() for f in args.fields.split(',')]
        print(f"\nBatch mode: will generate {len(field_list)} output files")
        print(f"Fields: {field_list}")
        
        for field_name in field_list:
            print(f"\n{'='*50}")
            print(f"Processing field: {field_name}")
            
            # 提取指定字段
            extracted_domain = extract_field(domain_data, field_name)
            print(f"Extracted {len(extracted_domain)} samples from field '{field_name}'")
            
            if len(extracted_domain) == 0:
                print(f"Warning: No data extracted for field '{field_name}', skipping...")
                continue
            
            # 生成输出文件名
            output_filename = f"batch_{field_name}.jsonl"
            output_path = Path(args.output).parent / output_filename
            
            # 混合数据
            mixed_data = mix_data(
                domain_data=extracted_domain,
                fineweb_data=extracted_corpus,
                domain_ratio=args.domain_ratio,
                fineweb_ratio=args.corpus_ratio,
                max_samples=args.max_samples,
                seed=args.seed
            )
            
            # 保存结果
            save_jsonl(mixed_data, str(output_path))
        
        print(f"\n{'='*50}")
        print("All batch processing completed!")
    else:
        # 单字段处理
        extracted_domain = extract_field(domain_data, args.domain_field)
        print(f"Extracted {len(extracted_domain)} samples from domain field '{args.domain_field}'")
        
        # 混合数据
        mixed_data = mix_data(
            domain_data=extracted_domain,
            fineweb_data=extracted_corpus,
            domain_ratio=args.domain_ratio,
            fineweb_ratio=args.corpus_ratio,
            max_samples=args.max_samples,
            seed=args.seed
        )
        
        # 保存结果
        save_jsonl(mixed_data, args.output)
        print("\nDone!")


if __name__ == "__main__":
    main()
