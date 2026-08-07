#!/usr/bin/env python3
"""
数据预处理脚本：将包含多个 Question-Answer 对的长文本转换为 Alpaca 格式

输入：目录路径，目录下包含多个 jsonl 文件，每个文件的 synthesized_QA 字段
      是一个长字符串，包含多个 "Question: XXX Answer: XXX" 对

输出：转换后的 jsonl 文件，格式为 Alpaca 格式 (instruction, input, output)

用法：
    python convert_qa_data.py --input_dir /path/to/your/data --output_dir /path/to/output
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any


def parse_qa_pairs(text: str) -> List[Dict[str, str]]:
    """
    解析包含多个 Question-Answer 对的长文本
    
    支持的格式：
    - Question: XXX Answer: XXX
    - Question: XXX\nAnswer: XXX
    - Q: XXX A: XXX
    - Q: XXX\nA: XXX
    """
    qa_pairs = []
    
    # 正则表达式匹配 Question/Answer 对
    # 匹配 "Question:" 或 "Q:" 后面的内容，直到 "Answer:" 或 "A:"
    patterns = [
        # Question: xxx Answer: xxx (可能换行)
        r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=(?:\s*Question:)|$)',
        r'Q:\s*(.+?)\s*A:\s*(.+?)(?=(?:\s*Q:)|$)',
        # 带换行的格式
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
                if q and a:  # 确保问题和答案都不为空
                    qa_pairs.append({
                        "question": q,
                        "answer": a
                    })
            if qa_pairs:
                break
    
    return qa_pairs


def convert_single_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将单条记录转换为 Alpaca 格式
    """
    results = []
    
    # 获取 synthesized_QA 字段
    qa_text = record.get("synthesized_QA", "")
    
    if not qa_text:
        return results
    
    # 解析 QA 对
    qa_pairs = parse_qa_pairs(qa_text)
    
    for qa in qa_pairs:
        # Alpaca 格式
        result = {
            "instruction": qa["question"],
            "input": "",
            "output": qa["answer"]
        }
        results.append(result)
    
    return results


def process_jsonl_file(input_path: Path, output_path: Path) -> int:
    """
    处理单个 jsonl 文件
    """
    count = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 转换记录
            converted_records = convert_single_record(record)
            
            for conv_record in converted_records:
                fout.write(json.dumps(conv_record, ensure_ascii=False) + '\n')
                count += 1
    
    return count


def process_directory(input_dir: str, output_dir: str) -> Dict[str, int]:
    """
    处理整个目录下的所有 jsonl 文件
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    # 查找所有 jsonl 文件
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"警告: 在 {input_dir} 中没有找到 jsonl 文件")
        return stats
    
    print(f"找到 {len(jsonl_files)} 个 jsonl 文件")
    
    for jsonl_file in jsonl_files:
        output_file = output_path / jsonl_file.name
        
        print(f"处理: {jsonl_file.name} -> {output_file.name}")
        count = process_jsonl_file(jsonl_file, output_file)
        
        stats[jsonl_file.name] = count
        print(f"  转换了 {count} 条记录")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="将包含多个 Question-Answer 对的文本转换为 Alpaca 格式"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录，包含 jsonl 文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，保存转换后的 jsonl 文件"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("数据预处理脚本")
    print("=" * 50)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 50)
    
    stats = process_directory(args.input_dir, args.output_dir)
    
    print("=" * 50)
    print("处理完成!")
    print(f"总处理文件数: {len(stats)}")
    total_records = sum(stats.values())
    print(f"总转换记录数: {total_records}")
    print("=" * 50)
    
    # 生成 dataset_info.json 配置
    dataset_info = {
        "your_dataset_name": {
            "file_name": args.output_dir
        }
    }
    
    info_file = Path(args.output_dir) / "dataset_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n已生成数据集配置: {info_file}")
    print("在训练配置中使用:")
    print(f"  dataset: your_dataset_name")


if __name__ == "__main__":
    main()
