#!/usr/bin/env python3
"""
综合数据预处理脚本：处理同一条记录中的 QA 和 Wiki 数据

输入：目录路径，目录下包含多个 jsonl 文件
      每条记录包含 synthesized_QA 和 synthesized_Wikipedia-style_rephrasing 字段

输出：转换后的 jsonl 文件，包含 QA 和 Wiki 两种格式的数据

用法：
    python convert_all_data.py \
        --input_dir /path/to/your/data \
        --output_dir data/mixed_train \
        --qa_weight 0.7 \
        --wiki_weight 0.3
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


def parse_qa_pairs(text: str) -> List[Dict[str, str]]:
    """解析包含多个 Question-Answer 对的长文本"""
    qa_pairs = []
    
    patterns = [
        r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=(?:\s*Question:)|$)',
        r'Q:\s*(.+?)\s*A:\s*(.+?)(?=(?:\s*Q:)|$)',
        r'Question:\s*\n*(.+?)\n*Answer:\s*\n*(.+?)(?=(?:\s*Question:)|$)',
        r'Q:\s*\n*(.+?)\n*A:\s*\n*(.+?)(?=(?:\s*Q:)|$)',
    ]
    
    text = text.strip()
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            for q, a in matches:
                q = q.strip()
                a = a.strip()
                if q and a:
                    qa_pairs.append({"question": q, "answer": a})
            if qa_pairs:
                break
    
    return qa_pairs


def convert_record(
    record: Dict[str, Any],
    qa_weight: float = 1.0,
    wiki_weight: float = 1.0,
    wiki_instruction: str = "请续写或扩展以下文本："
) -> List[Dict[str, Any]]:
    """
    将单条记录转换为 Alpaca 格式
    
    Args:
        record: 原始数据记录
        qa_weight: QA 数据权重
        wiki_weight: Wiki 数据权重
        wiki_instruction: Wiki 数据指令模板
    
    Returns:
        转换后的记录列表
    """
    results = []
    
    # 处理 QA 数据
    qa_text = record.get("synthesized_QA", "")
    if qa_text and qa_weight > 0:
        qa_pairs = parse_qa_pairs(qa_text)
        for qa in qa_pairs:
            results.append({
                "instruction": qa["question"],
                "input": "",
                "output": qa["answer"],
                "data_type": "qa"
            })
    
    # 处理 Wiki 数据
    wiki_text = (
        record.get("synthesized_Wikipedia-style_rephrasing") or
        record.get("synthesized_Wkipedia-style_rephrasing") or
        record.get("synthesized_Wikipedia_style_rephrasing") or
        record.get("synthesized_wikipedia_style_rephrasing")
    )
    if wiki_text and wiki_weight > 0:
        results.append({
            "instruction": wiki_instruction,
            "input": "",
            "output": wiki_text,
            "data_type": "wiki"
        })
    
    return results


def process_jsonl_file(
    file_path: Path,
    qa_weight: float,
    wiki_weight: float,
    wiki_instruction: str,
    sampling: bool = True
) -> List[Dict[str, Any]]:
    """处理单个 jsonl 文件"""
    all_results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 转换记录
            converted = convert_record(
                record,
                qa_weight=qa_weight,
                wiki_weight=wiki_weight,
                wiki_instruction=wiki_instruction
            )
            
            all_results.extend(converted)
    
    # 如果需要采样
    if sampling and qa_weight != wiki_weight:
        qa_results = [r for r in all_results if r.get("data_type") == "qa"]
        wiki_results = [r for r in all_results if r.get("data_type") == "wiki"]
        
        # 按权重比例采样
        if qa_weight > 0 and wiki_weight > 0:
            min_count = min(len(qa_results), len(wiki_results))
            target_qa = int(min_count * (qa_weight / (qa_weight + wiki_weight)))
            target_wiki = min_count - target_qa
            
            if target_qa < len(qa_results):
                qa_results = random.sample(qa_results, target_qa)
            if target_wiki < len(wiki_results):
                wiki_results = random.sample(wiki_results, target_wiki)
            
            all_results = qa_results + wiki_results
            random.shuffle(all_results)
    
    return all_results


def process_directory(
    input_dir: str,
    output_dir: str,
    qa_weight: float,
    wiki_weight: float,
    wiki_instruction: str,
    output_filename: str = "train.jsonl",
    seed: int = 42
) -> Dict[str, int]:
    """处理整个目录"""
    random.seed(seed)
    
    input_path = Path(input_dir)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    # 查找所有 jsonl 文件
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"警告: 在 {input_dir} 中没有找到 jsonl 文件")
        return stats
    
    print(f"找到 {len(jsonl_files)} 个 jsonl 文件")
    
    # 处理所有文件
    all_mixed_data = []
    
    for jsonl_file in jsonl_files:
        print(f"处理: {jsonl_file.name}")
        
        file_data = process_jsonl_file(
            jsonl_file,
            qa_weight=qa_weight,
            wiki_weight=wiki_weight,
            wiki_instruction=wiki_instruction,
            sampling=False  # 最后统一采样
        )
        
        print(f"  -> {len(file_data)} 条记录")
        stats[jsonl_file.name] = len(file_data)
        all_mixed_data.extend(file_data)
    
    # 统一采样混合
    if qa_weight != wiki_weight:
        qa_results = [r for r in all_mixed_data if r.get("data_type") == "qa"]
        wiki_results = [r for r in all_mixed_data if r.get("data_type") == "wiki"]
        
        if qa_weight > 0 and wiki_weight > 0:
            min_count = min(len(qa_results), len(wiki_results))
            target_qa = int(min_count * (qa_weight / (qa_weight + wiki_weight)))
            target_wiki = min_count - target_qa
            
            sampled_qa = random.sample(qa_results, min(target_qa, len(qa_results))) if qa_results else []
            sampled_wiki = random.sample(wiki_results, min(target_wiki, len(wiki_results))) if wiki_results else []
            
            all_mixed_data = sampled_qa + sampled_wiki
            random.shuffle(all_mixed_data)
    
    # 移除 data_type 字段
    for record in all_mixed_data:
        record.pop("data_type", None)
    
    # 保存输出
    output_file = output_path / output_filename
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_mixed_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n已保存到: {output_file}")
    
    return stats, len(all_mixed_data)


def main():
    parser = argparse.ArgumentParser(
        description="处理同一条记录中的 QA 和 Wiki 数据"
    )
    
    # 输入输出
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # 混合比例
    parser.add_argument("--qa_weight", type=float, default=0.7, help="QA 数据权重")
    parser.add_argument("--wiki_weight", type=float, default=0.3, help="Wiki 数据权重")
    
    # Wiki 数据选项
    parser.add_argument(
        "--wiki_instruction", 
        type=str, 
        default="请续写或扩展以下文本：",
        help="Wiki 数据指令模板"
    )
    
    # 输出选项
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="train.jsonl",
        help="输出文件名"
    )
    
    # 随机种子
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("综合数据预处理脚本")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"混合比例: QA={args.qa_weight}, Wiki={args.wiki_weight}")
    print(f"Wiki 指令: {args.wiki_instruction}")
    print("=" * 60)
    
    stats, total = process_directory(
        args.input_dir,
        args.output_dir,
        args.qa_weight,
        args.wiki_weight,
        args.wiki_instruction,
        args.output_file,
        args.seed
    )
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"总记录数: {total}")
    print("=" * 60)
    
    # 生成 dataset_info.json
    output_path = Path(args.output_dir)
    dataset_info = {
        "mixed_train": {
            "file_name": args.output_file
        }
    }
    
    info_file = output_path / "dataset_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n数据集配置已生成: {info_file}")
    print(f"在训练配置中使用: dataset: mixed_train")


if __name__ == "__main__":
    main()
