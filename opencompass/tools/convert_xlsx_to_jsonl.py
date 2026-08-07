#!/usr/bin/env python
"""
将 xlsx 文件转换为 opencompass 可用的 jsonl 格式
用于选择题数据集 (question, A, B, C, D, answer)
"""

import argparse
import json
import os

import pandas as pd


def convert_xlsx_to_jsonl(input_path: str, output_path: str = None):
    """将 xlsx 文件转换为 jsonl 格式

    Args:
        input_path: 输入的 xlsx 文件路径
        output_path: 输出的 jsonl 文件路径，如果为 None 则自动生成
    """
    # 读取 xlsx 文件
    df = pd.read_excel(input_path)

    # 验证必需的列
    required_columns = ['question', 'A', 'B', 'C', 'D', 'answer']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")

    # 转换为 list[dict] 格式
    dataset = df.to_dict(orient='records')

    # 如果没有指定输出路径，则自动生成
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}.jsonl"

    # 保存为 jsonl 格式
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            # 确保所有值都是字符串类型
            for key in item:
                if pd.isna(item[key]):
                    item[key] = ""
                else:
                    item[key] = str(item[key])
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"转换完成！")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print(f"  数据条数: {len(dataset)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='将 xlsx 文件转换为 opencompass 可用的 jsonl 格式'
    )
    parser.add_argument(
        'input',
        type=str,
        help='输入的 xlsx 文件路径'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出的 jsonl 文件路径 (默认: 自动生成)'
    )

    args = parser.parse_args()

    convert_xlsx_to_jsonl(args.input, args.output)


if __name__ == '__main__':
    main()
