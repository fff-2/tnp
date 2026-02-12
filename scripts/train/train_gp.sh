#!/bin/bash

# Training Configuration for GP Regression
NUM_STEPS=100000
BATCH_SIZE=16
LEARNING_RATE=5e-4
SEED=0
MODEL="tnpd"  # Options: np, anp, cnp, canp, bnp, banp, tnpa, tnpd, tnpnd
WANDB_PROJECT="tnp-gp"
WANDB_ENTITY=""

# Data Configuration
MAX_NUM_POINTS=50

# Training Parameters
NUM_SAMPLES=4
NUM_BS=10
PRINT_FREQ=200
EVAL_FREQ=5000
SAVE_FREQ=1000

echo "Running GP Regression training with Model: $MODEL"

python3 regression/gp.py \
    --mode train \
    --model "$MODEL" \
    --num_steps "$NUM_STEPS" \
    --train_batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --train_seed "$SEED" \
    --max_num_points "$MAX_NUM_POINTS" \
    --train_num_samples "$NUM_SAMPLES" \
    --train_num_bs "$NUM_BS" \
    --print_freq "$PRINT_FREQ" \
    --save_freq "$SAVE_FREQ" \
    --eval_freq "$EVAL_FREQ" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    "$@"
