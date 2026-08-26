#!/bin/bash
# =============================================================================
# CPT 3-Batch Training + Annealing Script
# =============================================================================
# 
# Description:
#   This script performs continued pre-training (CPT) with 3 batches in parallel:
#   - Batch 1: Original text
#   - Batch 2: QA pairs
#   - Batch 3: Wikipedia-style rephrasing
#   
#   Each batch runs on separate GPU, followed by annealing stage if checkpoint exists.
#
# Features:
#   - Parallel training on 3 GPUs
#   - Separate log files for each batch
#   - Auto-checkpoint detection for annealing
#   - Nohup background execution
#
# Usage:
#   1. Modify DOMAIN_DATA and CORPUS_DATA paths below
#   2. Run: bash train_cpt_3batch.sh
#
# =============================================================================

# =============================================================================
# Configuration
# =============================================================================

# Data paths
DOMAIN_DATA="/home/user/data/your_data.jsonl"       # Domain data JSONL (contains text, synthesized_QA, synthesized_Wikipedia-style_rephrasing fields)
CORPUS_DATA="/home/user/data/your_corpus.jsonl"     # General corpus data (e.g., Fineweb, Wikipedia)

# GPU configuration (3 batches use 3 GPUs)
GPU_IDS=(3 4 6)

# Output directories
OUTPUT_DIR="saves/qwen2.5-7b/full/cpt_3batch"
MAIN_DATA_DIR="data/cpt_dataset/main"
ANNEAL_DATA_DIR="data/cpt_dataset/anneal"

# Training parameters
LEARNING_RATE="5e-6"
ANNEAL_LEARNING_RATE="5e-7"
MAIN_STEPS=150
ANNEAL_STEPS=100

# =============================================================================
# Functions
# =============================================================================

# Run main training for single batch
run_main_train() {
    local batch_name=$1       # Batch name
    local dataset_name=$2      # Dataset name
    local output_subdir=$3     # Output subdirectory
    local gpu_id=$4           # GPU ID
    local log_file="$OUTPUT_DIR/logs/${batch_name}_main.log"
    
    mkdir -p "$OUTPUT_DIR/logs"
    
    echo "Starting batch $batch_name main training, GPU: $gpu_id, log: $log_file"
    
    # Run with nohup in background
    FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=$gpu_id nohup llamafactory-cli train \
        examples/train_full/qwen2.5_7b_full_cpt_3batch.yaml \
        dataset=$dataset_name \
        output_dir="$OUTPUT_DIR/$output_subdir" \
        > "$log_file" 2>&1 &
    
    echo "Main training started, PID: $!, log: $log_file"
}

# Check and run annealing
check_and_run_anneal() {
    local batch_name=$1           # Batch name
    local main_output_dir=$2      # Main training output directory
    local anneal_dataset=$3        # Annealing dataset name
    local anneal_output_dir=$4    # Annealing output directory
    local gpu_id=$5               # GPU ID
    local log_file="$OUTPUT_DIR/logs/${batch_name}_anneal.log"
    
    # Check if main training checkpoint exists
    local checkpoint_path="$main_output_dir/checkpoint-$MAIN_STEPS"
    
    if [ -d "$checkpoint_path" ]; then
        echo "Main training completed: $checkpoint_path, starting annealing"
        
        mkdir -p "$OUTPUT_DIR/logs"
        
        # Run annealing with nohup
        FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=$gpu_id nohup llamafactory-cli train \
            examples/train_full/qwen2.5_7b_full_cpt_3batch_anneal.yaml \
            dataset=$anneal_dataset \
            output_dir="$anneal_output_dir" \
            resume_from_checkpoint="$checkpoint_path" \
            learning_rate=$ANNEAL_LEARNING_RATE \
            max_steps=$ANNEAL_STEPS \
            > "$log_file" 2>&1 &
        
        echo "Annealing started, PID: $!, log: $log_file"
    else
        echo "Main training checkpoint not found: $checkpoint_path, skipping annealing"
    fi
}

# =============================================================================
# Main Flow
# =============================================================================

echo "============================================"
echo "CPT 3-Batch Training + Annealing Script"
echo "============================================"
echo "Domain data: $DOMAIN_DATA"
echo "Corpus data: $CORPUS_DATA"
echo "Output dir:  $OUTPUT_DIR"
echo "GPU IDs:     ${GPU_IDS[@]}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Generate main training data (60% domain + 40% corpus)
# -----------------------------------------------------------------------------
echo "============================================"
echo "Step 1: Generate main training data"
echo "============================================"
python scripts/mix_cpt_data.py \
    --domain-data "$DOMAIN_DATA" \
    --corpus-data "$CORPUS_DATA" \
    --output "$MAIN_DATA_DIR/" \
    --domain-ratio 0.6 --corpus-ratio 0.4 \
    --domain-field text \
    --corpus-field text \
    --fields text,synthesized_QA,synthesized_Wikipedia-style_rephrasing

echo "Main training data generated"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Generate annealing data (70% domain + 30% corpus)
# -----------------------------------------------------------------------------
echo "============================================"
echo "Step 2: Generate annealing data"
echo "============================================"
python scripts/mix_cpt_data.py \
    --domain-data "$DOMAIN_DATA" \
    --corpus-data "$CORPUS_DATA" \
    --output "$ANNEAL_DATA_DIR/" \
    --domain-ratio 0.7 --corpus-ratio 0.3 \
    --domain-field text \
    --corpus-field text \
    --fields text,synthesized_QA,synthesized_Wikipedia-style_rephrasing

echo "Annealing data generated"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Start 3 main training jobs in parallel
# -----------------------------------------------------------------------------
echo "============================================"
echo "Step 3: Start 3 main training jobs (parallel)"
echo "============================================"

# Batch 1: Original text (GPU 3)
run_main_train "batch1_text" "batch_text" "batch1_text" "${GPU_IDS[0]}"

# Batch 2: QA pairs (GPU 4)
run_main_train "batch2_qa" "batch_synthesized_QA" "batch2_qa" "${GPU_IDS[1]}"

# Batch 3: Wikipedia rephrasing (GPU 6)
run_main_train "batch3_wiki" "batch_synthesized_Wikipedia-style_rephrasing" "batch3_wiki" "${GPU_IDS[2]}"

echo ""
echo "3 main training jobs started in parallel"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Wait for main training and check for annealing
# -----------------------------------------------------------------------------
echo "Waiting for main training to complete..."

# Poll to check if all main training completed
while true; do
    sleep 30
    
    all_done=true
    
    # Check batch 1
    if [ ! -d "$OUTPUT_DIR/batch1_text/checkpoint-$MAIN_STEPS" ]; then
        all_done=false
    fi
    
    # Check batch 2
    if [ ! -d "$OUTPUT_DIR/batch2_qa/checkpoint-$MAIN_STEPS" ]; then
        all_done=false
    fi
    
    # Check batch 3
    if [ ! -d "$OUTPUT_DIR/batch3_wiki/checkpoint-$MAIN_STEPS" ]; then
        all_done=false
    fi
    
    if [ "$all_done" = true ]; then
        echo "All main training completed, checking annealing..."
        break
    fi
    
    echo "Main training in progress, waiting..."
done

# -----------------------------------------------------------------------------
# Step 5: Check and start annealing training
# -----------------------------------------------------------------------------
echo "============================================"
echo "Step 5: Check and start annealing"
echo "============================================"

# Annealing batch 1
check_and_run_anneal "batch1_text" "$OUTPUT_DIR/batch1_text" "anneal_text" "$OUTPUT_DIR/batch1_text_anneal" "${GPU_IDS[0]}"

# Annealing batch 2
check_and_run_anneal "batch2_qa" "$OUTPUT_DIR/batch2_qa" "anneal_synthesized_QA" "$OUTPUT_DIR/batch2_qa_anneal" "${GPU_IDS[1]}"

# Annealing batch 3
check_and_run_anneal "batch3_wiki" "$OUTPUT_DIR/batch3_wiki" "anneal_synthesized_Wikipedia-style_rephrasing" "$OUTPUT_DIR/batch3_wiki_anneal" "${GPU_IDS[2]}"

echo ""
echo "All annealing jobs started (if checkpoints exist)"
echo ""

# =============================================================================
# Completion
# =============================================================================
echo "============================================"
echo "Main flow completed!"
echo "============================================"
echo "Models saved to: $OUTPUT_DIR"
echo ""
echo "Log files:"
echo "  Batch 1 main:  $OUTPUT_DIR/logs/batch1_text_main.log"
echo "  Batch 2 main:  $OUTPUT_DIR/logs/batch2_qa_main.log"
echo "  Batch 3 main:  $OUTPUT_DIR/logs/batch3_wiki_main.log"
echo "  Batch 1 anneal: $OUTPUT_DIR/logs/batch1_text_anneal.log"
echo "  Batch 2 anneal: $OUTPUT_DIR/logs/batch2_qa_anneal.log"
echo "  Batch 3 anneal: $OUTPUT_DIR/logs/batch3_wiki_anneal.log"
