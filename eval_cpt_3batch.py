#!/usr/bin/env python3
"""
CPT 3-Batch Evaluation Script
Evaluate models before and after annealing using standalone_eval_fast.py

Usage:
    python eval_cpt_3batch.py \
        --base_model /path/to/base_model \
        --output_dir saves/qwen2.5-7b/full/cpt_3batch \
        --eval_data_dir opencompass \
        --eval_results_dir eval_results

This script will evaluate:
1. Base model (before CPT)
2. CPT models (after main training, before annealing)
3. Annealed models (after annealing)
"""

import os
import sys
import glob
import json
import shutil
import argparse
from pathlib import Path

# Import the evaluation function from standalone_eval_fast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opencompass.standalone_eval_fast import load_data, load_model, evaluate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPT 3-Batch Evaluation Script"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/home/hzs/260304/models/Qwen/Qwen2.5-7B-Instruct",
        help="Path to base model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saves/qwen2.5-7b/full/cpt_3batch",
        help="Output directory for training checkpoints"
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default="opencompass",
        help="Directory containing evaluation datasets"
    )
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--main_steps",
        type=int,
        default=150,
        help="Number of main training steps"
    )
    parser.add_argument(
        "--anneal_steps",
        type=int,
        default=100,
        help="Number of annealing steps"
    )
    return parser.parse_args()


def find_eval_datasets(eval_data_dir):
    """Find all evaluation datasets (xlsx and jsonl files)"""
    datasets = []
    
    # Find xlsx files
    for pattern in ['*.xlsx', '*.xls']:
        datasets.extend(glob.glob(os.path.join(eval_data_dir, pattern)))
    
    # Find jsonl files
    for pattern in ['*.jsonl']:
        datasets.extend(glob.glob(os.path.join(eval_data_dir, pattern)))
    
    return datasets


def run_evaluation(model_path, work_name, eval_datasets, eval_results_dir):
    """Run evaluation for a specific model"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {work_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    # Import here to avoid issues
    from opencompass.standalone_eval_fast import main as eval_main
    
    # Save original MODEL_PATH
    import opencompass.standalone_eval_fast as eval_module
    original_model_path = eval_module.MODEL_PATH
    
    try:
        # Set model path
        eval_module.MODEL_PATH = model_path
        
        # Create output directory
        work_dir = os.path.join(eval_results_dir, work_name)
        os.makedirs(work_dir, exist_ok=True)
        
        # Run evaluation for each dataset
        for dataset_path in eval_datasets:
            dataset_name = os.path.basename(dataset_path)
            print(f"\nEvaluating dataset: {dataset_name}")
            
            # Copy dataset to expected location
            temp_dir = "./datasets"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, dataset_name)
            shutil.copy(dataset_path, temp_path)
            
            try:
                # Run evaluation
                eval_module.main()
            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        print(f"\nEvaluation completed for {work_name}")
        
    finally:
        # Restore original model path
        eval_module.MODEL_PATH = original_model_path


def main():
    args = parse_args()
    
    print("="*60)
    print("CPT 3-Batch Evaluation Script")
    print("="*60)
    
    # Find evaluation datasets
    eval_datasets = find_eval_datasets(args.eval_data_dir)
    
    if not eval_datasets:
        print(f"Error: No evaluation datasets found in {args.eval_data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(eval_datasets)} evaluation datasets:")
    for ds in eval_datasets:
        print(f"  - {ds}")
    
    # Create results directory
    os.makedirs(args.eval_results_dir, exist_ok=True)
    
    output_dir = args.output_dir
    
    # ==========================================================================
    # Step 1: Evaluate base model (before CPT)
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 1: Evaluate Base Model (Before CPT)")
    print("="*60)
    
    run_evaluation(
        args.base_model,
        "base_model",
        eval_datasets,
        args.eval_results_dir
    )
    
    # ==========================================================================
    # Step 2: Evaluate CPT models (before annealing)
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 2: Evaluate CPT Models (Before Annealing)")
    print("="*60)
    
    checkpoints_before_anneal = [
        ("batch1_text", os.path.join(output_dir, f"batch1_text/checkpoint-{args.main_steps}")),
        ("batch2_qa", os.path.join(output_dir, f"batch2_qa/checkpoint-{args.main_steps}")),
        ("batch3_wiki", os.path.join(output_dir, f"batch3_wiki/checkpoint-{args.main_steps}")),
    ]
    
    for batch_name, checkpoint_path in checkpoints_before_anneal:
        if os.path.exists(checkpoint_path):
            run_evaluation(
                checkpoint_path,
                f"cpt_{batch_name}_before_anneal",
                eval_datasets,
                args.eval_results_dir
            )
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    # ==========================================================================
    # Step 3: Evaluate annealed models
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 3: Evaluate Annealed Models")
    print("="*60)
    
    checkpoints_after_anneal = [
        ("batch1_text", os.path.join(output_dir, f"batch1_text_anneal/checkpoint-{args.anneal_steps}")),
        ("batch2_qa", os.path.join(output_dir, f"batch2_qa_anneal/checkpoint-{args.anneal_steps}")),
        ("batch3_wiki", os.path.join(output_dir, f"batch3_wiki_anneal/checkpoint-{args.anneal_steps}")),
    ]
    
    for batch_name, checkpoint_path in checkpoints_after_anneal:
        if os.path.exists(checkpoint_path):
            run_evaluation(
                checkpoint_path,
                f"cpt_{batch_name}_after_anneal",
                eval_datasets,
                args.eval_results_dir
            )
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    # ==========================================================================
    # Completion
    # ==========================================================================
    print("\n" + "="*60)
    print("All Evaluations Completed!")
    print("="*60)
    print(f"Results saved to: {args.eval_results_dir}")


if __name__ == "__main__":
    main()
