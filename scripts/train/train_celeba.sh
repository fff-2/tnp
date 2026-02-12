#!/bin/bash

# Training Configuration for CelebA
EPOCHS=200
BATCH_SIZE=100
LEARNING_RATE=5e-4
SEED=0
MODEL="tnpa"  # Options: np, anp, cnp, canp, bnp, banp, tnpa, tnpd, tnpnd
WANDB_PROJECT="tnp-celeba"
WANDB_ENTITY=""  # Leave empty if not applicable

# Data Configuration
MAX_NUM_POINTS=200

# Training Parameters
NUM_SAMPLES=4
NUM_BS=10
SAVE_FREQ=10
EVAL_FREQ=10

echo "Running CelebA training with Model: $MODEL"

python3 regression/celeba.py \
    --mode train \
    --model "$MODEL" \
    --num_epochs "$EPOCHS" \
    --train_batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --train_seed "$SEED" \
    --max_num_points "$MAX_NUM_POINTS" \
    --train_num_samples "$NUM_SAMPLES" \
    --train_num_bs "$NUM_BS" \
    --save_freq "$SAVE_FREQ" \
    --eval_freq "$EVAL_FREQ" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    "$@"
