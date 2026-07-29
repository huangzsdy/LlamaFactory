#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

# train job
GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6002
WORLD_SIZE=1
RANK=0
TOKENIZER_MODEL=/user/pretrain/120k_v2.model

DATA_PATH="1.0 /user/traindoc/text/by_minicpm3_tokenizer.74k/github_repo"
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $WORLD_SIZE 
    --node_rank $RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --vocab-size 73448
    --make-vocab-size-divisible-by 1
    --num-layers 24
    --hidden-size 1024 
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 2
    --kv-channels 64
    --ffn-hidden-size 4096
    --seq-length 4096
    --max-position-embeddings 4096
    --attention-dropout 0
    --hidden-dropout 0
    --swiglu
    --position-embedding-type rope
    --disable-bias-linear
    --use-flash-attn
    --normalization RMSNorm
    --use-mup
    --mup-emb-scale 12
    --mup-depth-scale 1.4
    --init-method-std 0.1
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
    --bf16
    --lr 1e-2
    --min-lr 0.000625
    --train-iters 5000
    --lr-warmup-iters 100
    --lr-decay-style WSD
    --lr-wsd-decay-style exponential
    --lr-wsd-decay-iters 600
    --lr-decay-iters 5000
    --use-mcore-models
    --no-load-optim
)

MODEL_PARALLEL_ARGS=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-distributed-optimizer
    --tensor-model-parallel-size 1
    --use-dist-ckpt
    --dist-ckpt-format torch_dist
    --async-save
    --finetune
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-model $TOKENIZER_MODEL
    --tokenizer-type Llama2Tokenizer
    --split 1000,0,0
    --use-modelbest-sdk
    --dataloader-type external
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-task-loss-interval 10
    --save-interval 1
    --eval-interval 100000000 
    # --save temp/ 
    # --load $CHECKPOINT_PATH
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

set -ex
torchrun ${DISTRIBUTED_ARGS[@]} tools/dist_ckpt_to_hf_minicpm_1b.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    $@