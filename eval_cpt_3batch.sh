#!/bin/bash
# =============================================================================
# CPT 3-Batch Evaluation Script
# Evaluate models before and after annealing using standalone_eval_fast.py
# =============================================================================

# =============================================================================
# Configuration
# =============================================================================

# Model paths
BASE_MODEL_PATH="/home/hzs/260304/models/Qwen/Qwen2.5-7B-Instruct"  # Base model path

# Output directories (same as training)
OUTPUT_DIR="saves/qwen2.5-7b/full/cpt_3batch"

# Checkpoint paths - update these after training completes
BATCH1_TEXT_CKPT="$OUTPUT_DIR/batch1_text/checkpoint-150"
BATCH2_QA_CKPT="$OUTPUT_DIR/batch2_qa/checkpoint-150"
BATCH3_WIKI_CKPT="$OUTPUT_DIR/batch3_wiki/checkpoint-150"

BATCH1_TEXT_ANNEAL_CKPT="$OUTPUT_DIR/batch1_text_anneal/checkpoint-100"
BATCH2_QA_ANNEAL_CKPT="$OUTPUT_DIR/batch2_qa_anneal/checkpoint-100"
BATCH3_WIKI_ANNEAL_CKPT="$OUTPUT_DIR/batch3_wiki_anneal/checkpoint-100"

# Evaluation datasets directory
EVAL_DATA_DIR="opencompass"

# Evaluation results output directory
EVAL_RESULTS_DIR="eval_results"

# =============================================================================
# Functions
# =============================================================================

# Find all evaluation datasets (xlsx and jsonl files)
find_eval_datasets() {
    local datasets=()
    
    # Find xlsx files
    for f in $EVAL_DATA_DIR/*.xlsx; do
        if [ -f "$f" ]; then
            datasets+=("$f")
        fi
    done
    
    # Find jsonl files
    for f in $EVAL_DATA_DIR/*.jsonl; do
        if [ -f "$f" ]; then
            datasets+=("$f")
        fi
    done
    
    echo "${datasets[@]}"
}

# Run evaluation
run_eval() {
    local model_path=$1
    local work_name=$2
    local datasets=$3
    
    echo "============================================"
    echo "Evaluating: $work_name"
    echo "Model: $model_path"
    echo "============================================"
    
    # Create output directory
    mkdir -p "$EVAL_RESULTS_DIR/$work_name"
    
    # Run evaluation for each dataset
    for dataset in $datasets; do
        dataset_name=$(basename "$dataset")
        echo ""
        echo "Evaluating dataset: $dataset_name"
        
        python opencompass/standalone_eval_fast.py \
            --model_path "$model_path" \
            --eval_file "$dataset" \
            --work_name "$work_name" \
            --output_dir "$EVAL_RESULTS_DIR/$work_name"
        
        echo "Completed: $dataset_name"
    done
    
    echo ""
    echo "Evaluation completed for $work_name"
}

# =============================================================================
# Main Flow
# =============================================================================

echo "============================================"
echo "CPT 3-Batch Evaluation Script"
echo "============================================"

# Find all evaluation datasets
EVAL_DATASETS=$(find_eval_datasets)

if [ -z "$EVAL_DATASETS" ]; then
    echo "Error: No evaluation datasets found in $EVAL_DATA_DIR"
    exit 1
fi

echo "Found evaluation datasets:"
for ds in $EVAL_DATASETS; do
    echo "  - $ds"
done
echo ""

# Create results directory
mkdir -p "$EVAL_RESULTS_DIR"

# -----------------------------------------------------------------------------
# Step 1: Evaluate base model (before CPT)
# -----------------------------------------------------------------------------
echo "============================================"
echo "Step 1: Evaluate Base Model (Before CPT)"
echo "============================================"

run_eval "$BASE_MODEL_PATH" "base_model" "$EVAL_DATASETS"

# -----------------------------------------------------------------------------
# Step 2: Evaluate CPT models (after main training, before annealing)
# -----------------------------------------------------------------------------
echo "============================================"
echo "Step 2: Evaluate CPT Models (Before Annealing)"
echo "============================================"

if [ -d "$BATCH1_TEXT_CKPT" ]; then
    run_eval "$BATCH1_TEXT_CKPT" "cpt_batch1_text_before_anneal" "$EVAL_DATASETS"
else
    echo "Warning: Checkpoint not found: $BATCH1_TEXT_CKPT"
fi

if [ -d "$BATCH2_QA_CKPT" ]; then
    run_eval "$BATCH2_QA_CKPT" "cpt_batch2_qa_before_anneal" "$EVAL_DATASETS"
else
    echo "Warning: Checkpoint not found: $BATCH2_QA_CKPT"
fi

if [ -d "$BATCH3_WIKI_CKPT" ]; then
    run_eval "$BATCH3_WIKI_CKPT" "cpt_batch3_wiki_before_anneal" "$EVAL_DATASETS"
else
    echo "Warning: Checkpoint not found: $BATCH3_WIKI_CKPT"
fi

# -----------------------------------------------------------------------------
# Step 3: Evaluate annealed models (after annealing)
# -----------------------------------------------------------------------------
echo "============================================"
echo "Step 3: Evaluate Annealed Models"
echo "============================================"

if [ -d "$BATCH1_TEXT_ANNEAL_CKPT" ]; then
    run_eval "$BATCH1_TEXT_ANNEAL_CKPT" "cpt_batch1_text_after_anneal" "$EVAL_DATASETS"
else
    echo "Warning: Checkpoint not found: $BATCH1_TEXT_ANNEAL_CKPT"
fi

if [ -d "$BATCH2_QA_ANNEAL_CKPT" ]; then
    run_eval "$BATCH2_QA_ANNEAL_CKPT" "cpt_batch2_qa_after_anneal" "$EVAL_DATASETS"
else
    echo "Warning: Checkpoint not found: $BATCH2_QA_ANNEAL_CKPT"
fi

if [ -d "$BATCH3_WIKI_ANNEAL_CKPT" ]; then
    run_eval "$BATCH3_WIKI_ANNEAL_CKPT" "cpt_batch3_wiki_after_anneal" "$EVAL_DATASETS"
else
    echo "Warning: Checkpoint not found: $BATCH3_WIKI_ANNEAL_CKPT"
fi

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------
echo "============================================"
echo "All Evaluations Completed!"
echo "============================================"
echo "Results saved to: $EVAL_RESULTS_DIR"
echo ""
echo "Summary:"
echo "  - Base model: $BASE_MODEL_PATH"
echo "  - Batch 1 (text) checkpoints: $BATCH1_TEXT_CKPT, $BATCH1_TEXT_ANNEAL_CKPT"
echo "  - Batch 2 (QA) checkpoints: $BATCH2_QA_CKPT, $BATCH2_QA_ANNEAL_CKPT"
echo "  - Batch 3 (Wiki) checkpoints: $BATCH3_WIKI_CKPT, $BATCH3_WIKI_ANNEAL_CKPT"
