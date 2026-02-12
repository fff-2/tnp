#!/bin/bash

# Training Configuration for 1D GP Surrogate (Bayesian Optimization)
NUM_EPOCHS=100000
BATCH_SIZE=16
LEARNING_RATE=5e-4
SEED=0
MODEL="tnpa"  # Options: np, anp, cnp, canp, bnp, banp, tnpa, tnpd, tnpnd
WANDB_PROJECT="tnp-bo-1d"
WANDB_ENTITY=""

# Data Configuration for 1D BO Surrogate Training
MAX_NUM_POINTS=50

# Training Parameters
NUM_SAMPLES=4
NUM_BS=10
PRINT_FREQ=200
EVAL_FREQ=5000
SAVE_FREQ=1000

echo "Running 1D GP Surrogate Training with Model: $MODEL"

python3 bayesian_optimization/1d_gp.py \
    --mode train \
    --model "$MODEL" \
    --num_epochs "$NUM_EPOCHS" \
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
