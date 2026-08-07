#!/usr/bin/env python3
"""
CPT (Continued Pre-Training) Data Converter
将包含 text、synthesized_QA、synthesized_Wikipedia-style_rephrasing 三个字段的JSONL数据
转换为用于CPT（继续预训练）的自回归续写格式。

功能说明:
    1. 输入: 每行包含 text（原始文本）、synthesized_QA（问答对）、synthesized_Wikipedia-style_rephrasing（Wiki风格改写）
    2. 输出: 每行为 {"text": "..."} 格式的JSONL文件
    3. 四种数据类型及示例:
       - Type A (60%): 原始文本直接续写
         输入: text = "Attention is a mechanism used in neural networks."
         输出: {"text": "Attention is a mechanism used in neural networks.\n"}
       
       - Type B (20%): Wiki风格改写
         输入: synthesized_Wikipedia-style_rephrasing = "In the field of deep learning, the attention mechanism serves as a crucial component..."
         输出: {"text": "In the field of deep learning, the attention mechanism serves as a crucial component...\n"}
       
       - Type C (18%): 带上下文的问答 (90% QA对)
         输入: text = "Attention is a mechanism used in neural networks.", QA = "Question: What is attention? Answer: Attention is a mechanism..."
         输出: {"text": "Refer to the following article to answer the question:\nAttention is a mechanism used in neural networks.\n\nQuestion: What is attention?\nAnswer: Attention is a mechanism...\n"}
       
       - Type D (2%): 纯问答 (10% QA对)
         输入: QA = "Question: Where is attention used? Answer: It is used in deep learning models."
         输出: {"text": "Question: Where is attention used?\nAnswer: It is used in deep learning models.\n"}
    4. 曝光上限: 每条原始文本在A+C类型中最多出现3次
    5. QA分配: 90%用于Type C，10%用于Type D

Args:
    --input, -i:      输入JSONL文件路径 (必需)
    --output, -o:     输出JSONL文件路径 (必需)
    --ratio-a:        Type A 原始文本比例 (默认: 0.60)
    --ratio-b:        Type B Wiki改写比例 (默认: 0.20)
    --ratio-c:        Type C 问答-上下文比例 (默认: 0.18)
    --ratio-d:        Type D 纯问答比例 (默认: 0.02)
    --max-exposure:   每条原始文本最大曝光次数 (默认: 3)
    --seed, -s:       随机种子 (默认: 42)

Usage:
    python scripts/convert_cpt_data.py --input input.jsonl --output output.jsonl

Example:
    python scripts/convert_cpt_data.py -i data.jsonl -o cpt_data.jsonl --max-exposure 3 --seed 42
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def parse_qa_pairs(qa_string: str) -> list[tuple[str, str]]:
    """
    Parse QA pairs from the synthesized_QA string.
    
    Args:
        qa_string: String containing "Question: ... Answer: ..." pairs
        
    Returns:
        List of (question, answer) tuples
    """
    pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\nQuestion:|$)"
    matches = re.findall(pattern, qa_string, re.DOTALL)
    
    qa_pairs = []
    for q, a in matches:
        q = q.strip()
        a = a.strip()
        if q and a:
            qa_pairs.append((q, a))
    
    return qa_pairs


def build_type_a(text: str) -> dict:
    """Original Text (Direct Continuation)"""
    return {"text": f"{text}\n"}


def build_type_b(wiki_rephrasing: str) -> dict:
    """Wiki Rephrasing (As a standard document)"""
    return {"text": f"{wiki_rephrasing}\n"}


def build_type_c(text: str, question: str, answer: str) -> dict:
    """QA-with-Context (Core knowledge injection)"""
    return {
        "text": (
            "Refer to the following article to answer the question:\n"
            f"{text}\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
        )
    }


def build_type_d(question: str, answer: str) -> dict:
    """Pure-QA (Minimal alignment, no context)"""
    return {"text": f"Question: {question}\nAnswer: {answer}\n"}


def count_text_occurrences(samples: list[dict], original_texts: list[str]) -> Counter:
    """
    Count how many times each original text appears in Type A and Type C samples.
    """
    text_counter = Counter()
    for sample in samples:
        text_content = sample["text"]
        for orig_text in original_texts:
            # Check if original text appears in this sample
            if orig_text in text_content:
                text_counter[orig_text] += 1
    return text_counter


def convert_cpt_data(
    input_path: str,
    output_path: str,
    target_ratios: dict = None,
    max_exposure: int = 3,
    seed: int = 42
):
    """
    Convert JSONL data to CPT format with specified mixing ratios.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        target_ratios: Target ratios for each type (A, B, C, D)
        max_exposure: Maximum times each text can appear per epoch
        seed: Random seed for reproducibility
    """
    if target_ratios is None:
        target_ratios = {"A": 0.60, "B": 0.20, "C": 0.18, "D": 0.02}
    
    random.seed(seed)
    
    # Read all input data
    print(f"Reading input from: {input_path}")
    input_data = []
    original_texts_set = set()
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading input data"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                input_data.append(data)
                original_texts_set.add(data["text"])
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")
                continue
    
    print(f"Loaded {len(input_data)} records")
    
    # Collect all samples by type
    samples_a = []  # Original Text
    samples_b = []  # Wiki Rephrasing
    samples_c = []  # QA-with-Context
    samples_d = []  # Pure-QA
    
    original_text_to_entries = {}  # Track original text for exposure capping
    
    print("Processing data and generating samples...")
    
    for entry in tqdm(input_data, desc="Generating samples"):
        text = entry.get("text") or ""
        qa_string = entry.get("synthesized_QA") or ""
        wiki_rephrasing = entry.get("synthesized_Wikipedia-style_rephrasing") or ""
        
        if not text:
            continue
        
        # Type A: Original Text
        samples_a.append(build_type_a(text))
        
        # Track for exposure capping
        if text not in original_text_to_entries:
            original_text_to_entries[text] = {"A": 0, "C": 0}
        original_text_to_entries[text]["A"] += 1
        
        # Type B: Wiki Rephrasing
        if wiki_rephrasing:
            samples_b.append(build_type_b(wiki_rephrasing))
        
        # Parse QA pairs
        qa_pairs = parse_qa_pairs(qa_string)
        if qa_pairs:
            random.shuffle(qa_pairs)
            
            # 90% for Type C, 10% for Type D
            split_idx = max(1, int(len(qa_pairs) * 0.9))
            type_c_pairs = qa_pairs[:split_idx]
            type_d_pairs = qa_pairs[split_idx:]
            
            # Type C: QA-with-Context
            for q, a in type_c_pairs:
                samples_c.append(build_type_c(text, q, a))
                original_text_to_entries[text]["C"] += 1
            
            # Type D: Pure-QA
            for q, a in type_d_pairs:
                samples_d.append(build_type_d(q, a))
    
    print(f"\nGenerated samples before exposure capping:")
    print(f"  Type A: {len(samples_a)}")
    print(f"  Type B: {len(samples_b)}")
    print(f"  Type C: {len(samples_c)}")
    print(f"  Type D: {len(samples_d)}")
    
    # Apply exposure capping for Type A and Type C
    # We need to ensure each original text appears at most max_exposure times
    # in the combined A + C samples
    
    print(f"\nApplying exposure capping (max {max_exposure} occurrences per text)...")
    
    # First, collect all A and C samples with their source text
    a_c_samples = []
    for sample in samples_a:
        source_text = sample["text"].strip().rstrip("\n")
        a_c_samples.append((source_text, sample, "A"))
    
    for sample in samples_c:
        # Extract the original text from the context
        # Format: "Refer to the following article to answer the question:\n{text}\n\n..."
        text_content = sample["text"]
        if "Refer to the following article to answer the question:\n" in text_content:
            try:
                # Extract text between the header and "\n\n"
                start = text_content.index("Refer to the following article to answer the question:\n") + len("Refer to the following article to answer the question:\n")
                end = text_content.index("\n\n", start)
                source_text = text_content[start:end]
                a_c_samples.append((source_text, sample, "C"))
            except ValueError:
                # If parsing fails, skip this sample
                continue
    
    # Count occurrences per text
    text_occurrence_count = Counter()
    for source_text, _, _ in a_c_samples:
        text_occurrence_count[source_text] += 1
    
    # Filter samples based on exposure cap
    text_used_count = Counter()
    filtered_a_c_samples = []
    
    for source_text, sample, sample_type in a_c_samples:
        if text_used_count[source_text] < max_exposure:
            filtered_a_c_samples.append(sample)
            text_used_count[source_text] += 1
    
    # Re-split into A and C after filtering
    final_samples_a = []
    final_samples_c = []
    
    for sample in filtered_a_c_samples:
        if sample["text"].endswith("\n") and not "Refer to the following article" in sample["text"]:
            # Type A ends with just newline
            final_samples_a.append(sample)
        else:
            final_samples_c.append(sample)
    
    print(f"After exposure capping:")
    print(f"  Type A: {len(final_samples_a)}")
    print(f"  Type C: {len(final_samples_c)}")
    
    # Calculate current ratios
    total_samples = len(final_samples_a) + len(samples_b) + len(final_samples_c) + len(samples_d)
    
    print(f"\nCurrent ratios before balancing:")
    print(f"  Type A: {len(final_samples_a)/total_samples*100:.2f}%")
    print(f"  Type B: {len(samples_b)/total_samples*100:.2f}%")
    print(f"  Type C: {len(final_samples_c)/total_samples*100:.2f}%")
    print(f"  Type D: {len(samples_d)/total_samples*100:.2f}%")
    
    # Calculate target counts based on ratios
    target_a = int(total_samples * target_ratios["A"])
    target_b = int(total_samples * target_ratios["B"])
    target_c = int(total_samples * target_ratios["C"])
    target_d = int(total_samples * target_ratios["D"])
    
    print(f"\nTarget counts:")
    print(f"  Type A: {target_a}")
    print(f"  Type B: {target_b}")
    print(f"  Type C: {target_c}")
    print(f"  Type D: {target_d}")
    
    # Subsample to match target ratios
    final_list = []
    
    # Type A
    if len(final_samples_a) > target_a:
        final_list.extend(random.sample(final_samples_a, target_a))
    else:
        final_list.extend(final_samples_a)
    
    # Type B
    if len(samples_b) > target_b:
        final_list.extend(random.sample(samples_b, target_b))
    else:
        final_list.extend(samples_b)
    
    # Type C
    if len(final_samples_c) > target_c:
        final_list.extend(random.sample(final_samples_c, target_c))
    else:
        final_list.extend(final_samples_c)
    
    # Type D
    if len(samples_d) > target_d:
        final_list.extend(random.sample(samples_d, target_d))
    else:
        final_list.extend(samples_d)
    
    # Global shuffle for uniform distribution
    random.shuffle(final_list)
    
    print(f"\nFinal sample count: {len(final_list)}")
    
    # Write output
    print(f"Writing output to: {output_path}")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(final_list, desc="Writing output"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL data to CPT (Continued Pre-Training) format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--ratio-a",
        type=float,
        default=0.60,
        help="Ratio for Original Text (default: 0.60)"
    )
    parser.add_argument(
        "--ratio-b",
        type=float,
        default=0.20,
        help="Ratio for Wiki Rephrasing (default: 0.20)"
    )
    parser.add_argument(
        "--ratio-c",
        type=float,
        default=0.18,
        help="Ratio for QA-with-Context (default: 0.18)"
    )
    parser.add_argument(
        "--ratio-d",
        type=float,
        default=0.02,
        help="Ratio for Pure-QA (default: 0.02)"
    )
    parser.add_argument(
        "--max-exposure",
        type=int,
        default=3,
        help="Maximum exposure count per text (default: 3)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios sum to 1.0
    total_ratio = args.ratio_a + args.ratio_b + args.ratio_c + args.ratio_d
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        args.ratio_a /= total_ratio
        args.ratio_b /= total_ratio
        args.ratio_c /= total_ratio
        args.ratio_d /= total_ratio
    
    target_ratios = {
        "A": args.ratio_a,
        "B": args.ratio_b,
        "C": args.ratio_c,
        "D": args.ratio_d
    }
    
    convert_cpt_data(
        input_path=args.input,
        output_path=args.output,
        target_ratios=target_ratios,
        max_exposure=args.max_exposure,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
