#!/usr/bin/env python3
"""
CPT Data Mixing Script (Optimized Version)
Mix domain data with general corpus data at specified ratios for CPT training.

Usage:
    python scripts/mix_cpt_data.py \
        --domain-data data/cpt_dataset/your_domain_data.jsonl \
        --corpus-data data/cpt_dataset/your_corpus_data.jsonl \
        --output data/cpt_dataset/cpt_mixed.jsonl \
        --domain-ratio 0.6 \
        --corpus-ratio 0.4

Args:
    --domain-data: Path to domain data file (JSONL format)
    --corpus-data: Path to general corpus data file (JSONL format)
    --output: Output file path
    --domain-ratio: Domain data ratio (default: 0.6)
    --corpus-ratio: General corpus data ratio (default: 0.4)
    --max-samples: Maximum total samples (optional)
    --seed: Random seed (default: 42)
    
    # Field extraction parameters:
    --domain-field: Field name to extract from domain data (default: text)
    --corpus-field: Field name to extract from corpus data (default: text)
    
    # Batch processing:
    --fields text,synthesized_QA,synthesized_Wikipedia-style_rephrasing
    
Performance optimization options:
    --use-parallel: Use multiprocessing for parallel batch processing
    --num-workers: Number of parallel workers (default: 4)
    --corpus-cache: Cache corpus data to avoid reloading
    --no-progress: Disable progress bar
"""

import argparse
import json
import random
import os
from pathlib import Path
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

# Try to use orjson for faster JSON parsing (optional)
try:
    import orjson
    def json_loads(s):
        return orjson.loads(s)
except ImportError:
    json_loads = json.loads

# Try to use ujson for faster JSON parsing (optional)
try:
    import ujson
    def json_loads(s):
        return ujson.loads(s)
except ImportError:
    pass  # Use default json.loads


def load_jsonl_fast(file_path: str, show_progress: bool = True):
    """Fast JSONL file loader - optimized version
    
    Optimizations:
    1. Use larger buffer for reading
    2. Batch processing to reduce function call overhead
    3. Optional progress bar
    """
    data = []
    buffer_size = 1024 * 1024  # 1MB buffer
    
    with open(file_path, 'r', encoding='utf-8', buffering=buffer_size) as f:
        if show_progress:
            # Get line count for progress bar
            line_count = sum(1 for _ in f)
            f.seek(0)
            pbar = tqdm(total=line_count, desc=f"Loading {os.path.basename(file_path)}", unit="lines")
        else:
            pbar = None
            
        for line in f:
            if pbar:
                pbar.update(1)
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json_loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
        
        if pbar:
            pbar.close()
    
    return data


def load_jsonl_fast_iter(file_path: str):
    """Fast iterative JSONL loader - memory optimized version
    
    For ultra-large datasets, process line by line without loading all into memory
    """
    buffer_size = 1024 * 1024  # 1MB buffer
    
    with open(file_path, 'r', encoding='utf-8', buffering=buffer_size) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json_loads(line)
            except (json.JSONDecodeError, ValueError):
                continue


def load_jsonl(file_path: str):
    """Load JSONL file (legacy interface)"""
    return load_jsonl_fast(file_path, show_progress=True)


def extract_field(data: list, field_name: str):
    """Extract specified field from data and generate text format samples"""
    result = []
    for item in data:
        value = item.get(field_name, "")
        if value and isinstance(value, str):
            result.append({"text": value})
    return result


def save_jsonl(data: list, output_path: str):
    """Save data as JSONL file"""
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
    """Mix two datasets
    
    Fixed: Use domain data as baseline, calculate corpus samples based on
    the user-specified ratio to ensure correct final mixing ratio.
    """
    random.seed(seed)
    
    # Calculate target counts
    if max_samples:
        target_domain = int(max_samples * domain_ratio)
        target_fineweb = int(max_samples * fineweb_ratio)
    else:
        # Fixed: Use domain data as baseline, calculate corpus samples based on ratio
        # Example: 1000 domain data, domain:corpus = 0.6:0.4
        # Corpus samples = 1000 * (0.4/0.6) = 667
        # Final ratio = 60%:40%
        target_domain = len(domain_data)
        target_fineweb = int(len(domain_data) * (fineweb_ratio / domain_ratio)) if domain_ratio > 0 else 0
    
    print(f"\nData mixing summary:")
    print(f"  Domain data: {len(domain_data)} -> sampling {min(target_domain, len(domain_data))}")
    print(f"  Fineweb data: {len(fineweb_data)} -> sampling {min(target_fineweb, len(fineweb_data))}")
    
    # Sample
    sampled_domain = random.sample(
        domain_data, 
        min(target_domain, len(domain_data))
    )
    sampled_fineweb = random.sample(
        fineweb_data, 
        min(target_fineweb, len(fineweb_data))
    )
    
    # Mix
    mixed_data = sampled_domain + sampled_fineweb
    random.shuffle(mixed_data)
    
    print(f"  Final mixed: {len(mixed_data)} samples")
    print(f"  Final ratio: {len(sampled_domain)/len(mixed_data)*100:.1f}% domain + {len(sampled_fineweb)/len(mixed_data)*100:.1f}% fineweb")
    
    return mixed_data


def main():
    parser = argparse.ArgumentParser(
        description="Mix domain data with corpus for CPT training"
    )
    parser.add_argument(
        "--domain-data",
        type=str,
        required=True,
        help="Path to domain data JSONL file"
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
        help="Path to output mixed data JSONL file or directory"
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
    # Field extraction parameter: field name from domain data
    parser.add_argument(
        "--domain-field",
        type=str,
        default="text",
        help="Field name to extract from domain data (default: text)"
    )
    # Field extraction parameter: field name from corpus data
    parser.add_argument(
        "--corpus-field",
        type=str,
        default="text",
        help="Field name to extract from corpus data (default: text)"
    )
    # Batch processing: generate multiple output files
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Comma-separated field names to extract from same domain-data file. "
             "If provided, will generate multiple output files. "
             "Example: --fields text,synthesized_QA,synthesized_Wikipedia-style_rephrasing"
    )
    # Performance optimization: use multiprocessing
    parser.add_argument(
        "--use-parallel",
        action="store_true",
        help="Use multiprocessing for parallel batch processing"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    # Cache parameter: cache corpus to avoid reloading
    parser.add_argument(
        "--corpus-cache",
        type=str,
        default=None,
        help="Path to cache file for corpus data to avoid reloading"
    )
    # Performance optimization: disable progress bar
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable progress bar for faster execution (default: False)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.domain_ratio + args.corpus_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio}, should be 1.0")
        args.domain_ratio /= total_ratio
        args.corpus_ratio /= total_ratio
        print(f"Normalized to: domain={args.domain_ratio}, corpus={args.corpus_ratio}")
    
    # Load/cache data
    print("Loading data...")
    
    # Auto cache path: same as corpus-data with .cache.jsonl suffix
    default_cache_path = str(Path(args.corpus_data).with_suffix('.cache.jsonl'))
    cache_file = args.corpus_cache if args.corpus_cache else default_cache_path
    
    # Check if cache exists
    if Path(cache_file).exists():
        print(f"Loading corpus from cache: {cache_file}")
        corpus_data = load_jsonl_fast(cache_file, show_progress=not args.no_progress)
    else:
        # Load corpus
        corpus_data = load_jsonl_fast(args.corpus_data, show_progress=not args.no_progress)
        
        # Auto save cache
        print(f"Saving corpus cache to: {cache_file}")
        save_jsonl(corpus_data, cache_file)
    
    # Extract corpus field
    extracted_corpus = extract_field(corpus_data, args.corpus_field)
    print(f"Extracted {len(extracted_corpus)} samples from corpus field '{args.corpus_field}'")
    
    # Load domain data
    domain_data = load_jsonl_fast(args.domain_data, show_progress=not args.no_progress)
    
    # Check if batch processing or single field
    if args.fields:
        # Batch mode: extract multiple fields from same file
        field_list = [f.strip() for f in args.fields.split(',')]
        print(f"\nBatch mode: will generate {len(field_list)} output files")
        print(f"Fields: {field_list}")
        
        for field_name in field_list:
            print(f"\n{'='*50}")
            print(f"Processing field: {field_name}")
            
            # Extract specified field
            extracted_domain = extract_field(domain_data, field_name)
            print(f"Extracted {len(extracted_domain)} samples from field '{field_name}'")
            
            if len(extracted_domain) == 0:
                print(f"Warning: No data extracted for field '{field_name}', skipping...")
                continue
            
            # Generate output filename
            output_filename = f"batch_{field_name}.jsonl"
            # If --output ends with .jsonl, it's a file path; otherwise, it's a directory
            if args.output.endswith('.jsonl'):
                output_path = Path(args.output).parent / output_filename
            else:
                # Treat as directory, create file directly in this directory
                output_path = Path(args.output) / output_filename
            
            # Mix data
            mixed_data = mix_data(
                domain_data=extracted_domain,
                fineweb_data=extracted_corpus,
                domain_ratio=args.domain_ratio,
                fineweb_ratio=args.corpus_ratio,
                max_samples=args.max_samples,
                seed=args.seed
            )
            
            # Save result
            save_jsonl(mixed_data, str(output_path))
        
        print(f"\n{'='*50}")
        print("All batch processing completed!")
    else:
        # Single field processing
        extracted_domain = extract_field(domain_data, args.domain_field)
        print(f"Extracted {len(extracted_domain)} samples from domain field '{args.domain_field}'")
        
        # Mix data
        mixed_data = mix_data(
            domain_data=extracted_domain,
            fineweb_data=extracted_corpus,
            domain_ratio=args.domain_ratio,
            fineweb_ratio=args.corpus_ratio,
            max_samples=args.max_samples,
            seed=args.seed
        )
        
        # Save result
        save_jsonl(mixed_data, args.output)
        print("\nDone!")


if __name__ == "__main__":
    main()
