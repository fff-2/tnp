#!/bin/bash

# Training Configuration for High-Dim GP Surrogate (Bayesian Optimization)
NUM_STEPS=100000
BATCH_SIZE=16
LEARNING_RATE=5e-4
SEED=0
MODEL="tnpa"  # Options: np, anp, cnp, canp, bnp, banp, tnpa, tnpd, tnpnd
WANDB_PROJECT="tnp-bo-highdim"
WANDB_ENTITY=""
DIMENSION=3

# Data Configuration
MAX_NUM_POINTS=256
MIN_NUM_POINTS=64

# Training Parameters
NUM_BOOTSTRAP=10
PRINT_FREQ=200
EVAL_FREQ=5000
SAVE_FREQ=1000

echo "Running High-Dim GP Surrogate Training with Model: $MODEL Dimension: $DIMENSION"

python3 bayesian_optimization/highdim_gp.py \
    --mode train \
    --model "$MODEL" \
    --dimension "$DIMENSION" \
    --num_steps "$NUM_STEPS" \
    --train_batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --train_seed "$SEED" \
    --max_num_points "$MAX_NUM_POINTS" \
    --min_num_points "$MIN_NUM_POINTS" \
    --train_num_bootstrap "$NUM_BOOTSTRAP" \
    --print_freq "$PRINT_FREQ" \
    --save_freq "$SAVE_FREQ" \
    --eval_freq "$EVAL_FREQ" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    "$@"
