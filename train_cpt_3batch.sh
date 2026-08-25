#!/bin/bash
# =============================================================================
# CPT 3-Batch Training + Annealing Script
# CPT三批训练 + 退火完整流程脚本
# =============================================================================
# 
# Description / 描述:
#   This script performs continued pre-training (CPT) with 3 batches in parallel:
#   - Batch 1: Original text / 原文
#   - Batch 2: QA pairs / QA问答
#   - Batch 3: Wikipedia-style rephrasing / Wiki改写
#   
#   Each batch runs on separate GPU, followed by annealing stage if checkpoint exists.
#   每个批次在不同GPU上运行，完成后检测是否存在checkpoint，存在则进行退火训练。
#
# Features / 功能:
#   - Parallel training on 3 GPUs / 三GPU并行训练
#   - Separate log files for each batch / 每个批次独立日志
#   - Auto-checkpoint detection for annealing / 自动检测checkpoint启动退火
#   - No-hup background execution / nohup后台运行
#
# Usage / 使用方法:
#   1. Modify DOMAIN_DATA and CORPUS_DATA paths below
#   2. Run: bash train_cpt_3batch.sh
#
# =============================================================================

# =============================================================================
# Configuration / 配置参数
# =============================================================================

# Data paths / 数据路径
DOMAIN_DATA="/home/user/data/your_data.jsonl"       # Domain data JSONL (contains text, synthesized_QA, synthesized_Wikipedia-style_rephrasing fields)
CORPUS_DATA="/home/user/data/your_corpus.jsonl"     # General corpus data (e.g., Fineweb, Wikipedia)

# GPU configuration / GPU配置 (3 batches use 3 GPUs)
GPU_IDS=(3 4 6)

# Output directories / 输出目录
OUTPUT_DIR="saves/qwen2.5-7b/full/cpt_3batch"
MAIN_DATA_DIR="data/cpt_dataset/main"
ANNEAL_DATA_DIR="data/cpt_dataset/anneal"

# Training parameters / 训练参数
LEARNING_RATE="5e-6"
ANNEAL_LEARNING_RATE="5e-7"
MAIN_STEPS=150
ANNEAL_STEPS=100

# =============================================================================
# Functions / 函数定义
# =============================================================================

# Run main training for single batch / 运行单批次主训练
run_main_train() {
    local batch_name=$1       # Batch name / 批次名称
    local dataset_name=$2     # Dataset name / 数据集名称
    local output_subdir=$3    # Output subdirectory / 输出子目录
    local gpu_id=$4           # GPU ID / GPU编号
    local log_file="$OUTPUT_DIR/logs/${batch_name}_main.log"
    
    mkdir -p "$OUTPUT_DIR/logs"
    
    echo "Starting batch $batch_name main training, GPU: $gpu_id, log: $log_file"
    
    # Run with nohup in background / nohup后台运行
    CUDA_VISIBLE_DEVICES=$gpu_id nohup llamafactory-cli train \
        examples/train_full/qwen2.5_7b_full_cpt_3batch.yaml \
        --dataset $dataset_name \
        --output_dir "$OUTPUT_DIR/$output_subdir" \
        > "$log_file" 2>&1 &
    
    echo "Main training started, PID: $!, log: $log_file"
}

# Check and run annealing / 检查并运行退火训练
check_and_run_anneal() {
    local batch_name=$1           # Batch name / 批次名称
    local main_output_dir=$2      # Main training output directory / 主训练输出目录
    local anneal_dataset=$3        # Annealing dataset name / 退火数据集名称
    local anneal_output_dir=$4    # Annealing output directory / 退火输出目录
    local gpu_id=$5               # GPU ID / GPU编号
    local log_file="$OUTPUT_DIR/logs/${batch_name}_anneal.log"
    
    # Check if main training checkpoint exists / 检查主训练checkpoint是否存在
    local checkpoint_path="$main_output_dir/checkpoint-$MAIN_STEPS"
    
    if [ -d "$checkpoint_path" ]; then
        echo "Main training completed: $checkpoint_path, starting annealing"
        
        mkdir -p "$OUTPUT_DIR/logs"
        
        # Run annealing with nohup / nohup运行退火
        CUDA_VISIBLE_DEVICES=$gpu_id nohup llamafactory-cli train \
            examples/train_full/qwen2.5_7b_full_cpt_3batch_anneal.yaml \
            --dataset $anneal_dataset \
            --output_dir "$anneal_output_dir" \
            --resume_from_checkpoint "$checkpoint_path" \
            --learning_rate $ANNEAL_LEARNING_RATE \
            --max_steps $ANNEAL_STEPS \
            > "$log_file" 2>&1 &
        
        echo "Annealing started, PID: $!, log: $log_file"
    else
        echo "Main training checkpoint not found: $checkpoint_path, skipping annealing"
    fi
}

# =============================================================================
# Main Flow / 主流程
# =============================================================================

echo "============================================"
echo "CPT 3-Batch Training + Annealing Script"
echo "CPT三批训练 + 退火完整流程"
echo "============================================"
echo "Domain data: $DOMAIN_DATA"
echo "Corpus data: $CORPUS_DATA"
echo "Output dir:  $OUTPUT_DIR"
echo "GPU IDs:     ${GPU_IDS[@]}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Generate main training data (60% domain + 40% corpus)
# 步骤1: 生成主训练数据 (60%领域 + 40%语料)
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
# 步骤2: 生成退火数据 (70%领域 + 30%语料)
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
# 步骤3: 并行启动三个主训练任务
# -----------------------------------------------------------------------------
echo "============================================"
echo "Step 3: Start 3 main training jobs (parallel)"
echo "============================================"

# Batch 1: Original text / 批次1: 原文 (GPU 3)
run_main_train "batch1_text" "batch_text" "batch1_text" "${GPU_IDS[0]}"

# Batch 2: QA pairs / 批次2: QA (GPU 4)
run_main_train "batch2_qa" "batch_synthesized_QA" "batch2_qa" "${GPU_IDS[1]}"

# Batch 3: Wikipedia rephrasing / 批次3: Wiki改写 (GPU 6)
run_main_train "batch3_wiki" "batch_synthesized_Wikipedia-style_rephrasing" "batch3_wiki" "${GPU_IDS[2]}"

echo ""
echo "3 main training jobs started in parallel"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Wait for main training and check for annealing
# 步骤4: 等待主训练完成并检测退火
# -----------------------------------------------------------------------------
echo "Waiting for main training to complete..."

# Poll to check if all main training completed / 轮询检查所有主训练是否完成
while true; do
    sleep 30
    
    all_done=true
    
    # Check batch 1 / 检查批次1
    if [ ! -d "$OUTPUT_DIR/batch1_text/checkpoint-$MAIN_STEPS" ]; then
        all_done=false
    fi
    
    # Check batch 2 / 检查批次2
    if [ ! -d "$OUTPUT_DIR/batch2_qa/checkpoint-$MAIN_STEPS" ]; then
        all_done=false
    fi
    
    # Check batch 3 / 检查批次3
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
# 步骤5: 检查并启动退火训练
# -----------------------------------------------------------------------------
echo "============================================"
echo "Step 5: Check and start annealing"
echo "============================================"

# Annealing batch 1 / 退火批次1
check_and_run_anneal "batch1_text" "$OUTPUT_DIR/batch1_text" "anneal_text" "$OUTPUT_DIR/batch1_text_anneal" "${GPU_IDS[0]}"

# Annealing batch 2 / 退火批次2
check_and_run_anneal "batch2_qa" "$OUTPUT_DIR/batch2_qa" "anneal_synthesized_QA" "$OUTPUT_DIR/batch2_qa_anneal" "${GPU_IDS[1]}"

# Annealing batch 3 / 退火批次3
check_and_run_anneal "batch3_wiki" "$OUTPUT_DIR/batch3_wiki" "anneal_synthesized_Wikipedia-style_rephrasing" "$OUTPUT_DIR/batch3_wiki_anneal" "${GPU_IDS[2]}"

echo ""
echo "All annealing jobs started (if checkpoints exist)"
echo ""

# =============================================================================
# Completion / 完成
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
