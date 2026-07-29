#!/usr/bin/env python3
"""
数据预处理脚本：将 Wiki 百科风格改写文本转换为 Alpaca 格式

输入：目录路径，目录下包含多个 jsonl 文件，每个文件的 
      synthesized_Wkipedia-style_rephrasing 字段是 Wiki 风格改写后的文本

输出：转换后的 jsonl 文件，格式为 Alpaca 格式 (instruction, input, output)

用法：
    python convert_wiki_data.py --input_dir /path/to/your/wiki_data --output_dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


def convert_wiki_record(
    record: Dict[str, Any],
    instruction: str = "请续写或扩展以下文本：",
    input_field: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    将单条 Wiki 记录转换为 Alpaca 格式
    
    Args:
        record: 原始数据记录
        instruction: 指令模板
        input_field: 如果不为None，则将原始文本放在input字段
    
    Returns:
        Alpaca 格式的记录
    """
    # 获取 Wiki 文本字段（支持不同大小写）
    wiki_text = (
        record.get("synthesized_Wkipedia-style_rephrasing") or
        record.get("synthesized_Wikipedia-style_rephrasing") or
        record.get("synthesized_Wikipedia_style_rephrasing") or
        record.get("synthesized_wikipedia_style_rephrasing") or
        record.get("wiki_text") or
        record.get("text") or
        record.get("rephrased_text")
    )
    
    # 获取原始文本（如果存在）
    original_text = None
    if input_field:
        original_text = record.get(input_field)
    
    if not wiki_text:
        return None
    
    # 构建 Alpaca 格式
    if original_text and instruction:
        # 有原始文本和指令：让模型学习根据原文续写
        result = {
            "instruction": instruction,
            "input": original_text,
            "output": wiki_text
        }
    elif instruction:
        # 只有指令和改写文本：让模型学习文本扩展/改写
        result = {
            "instruction": instruction,
            "input": "",
            "output": wiki_text
        }
    else:
        # 纯文本续写（无指令）
        result = {
            "instruction": "",
            "input": "",
            "output": wiki_text
        }
    
    return result


def process_wiki_jsonl(
    input_path: Path, 
    output_path: Path,
    instruction: str = "请续写或扩展以下文本：",
    input_field: Optional[str] = None,
    max_samples: Optional[int] = None
) -> int:
    """
    处理单个 jsonl 文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        instruction: 指令模板
        input_field: 原始文本字段名（可选）
        max_samples: 最大样本数（用于采样）
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
            result = convert_wiki_record(record, instruction, input_field)
            
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                count += 1
                
                if max_samples and count >= max_samples:
                    break
    
    return count


def process_wiki_directory(
    input_dir: str, 
    output_dir: str,
    instruction: str = "请续写或扩展以下文本：",
    input_field: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict[str, int]:
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
        count = process_wiki_jsonl(
            jsonl_file, 
            output_file,
            instruction=instruction,
            input_field=input_field,
            max_samples=max_samples
        )
        
        stats[jsonl_file.name] = count
        print(f"  转换了 {count} 条记录")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="将 Wiki 百科风格改写文本转换为 Alpaca 格式"
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
    parser.add_argument(
        "--instruction",
        type=str,
        default="请续写或扩展以下文本：",
        help="指令模板"
    )
    parser.add_argument(
        "--input_field",
        type=str,
        default=None,
        help="原始文本字段名（可选）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个文件最大样本数（可选）"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Wiki 数据预处理脚本")
    print("=" * 50)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"指令模板: {args.instruction}")
    print(f"原始文本字段: {args.input_field or '无'}")
    print("=" * 50)
    
    stats = process_wiki_directory(
        args.input_dir, 
        args.output_dir,
        instruction=args.instruction,
        input_field=args.input_field,
        max_samples=args.max_samples
    )
    
    print("=" * 50)
    print("处理完成!")
    print(f"总处理文件数: {len(stats)}")
    total_records = sum(stats.values())
    print(f"总转换记录数: {total_records}")
    print("=" * 50)


if __name__ == "__main__":
    main()
