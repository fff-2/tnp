#!/bin/bash

# Training Configuration for Contextual Bandits (CMAB)
# Note: CMAB runner handles both train/eval in one go typically, but we use mode args if supported.
# The main.py delegates to cmab_runner which parses args.

NUM_ITER=10000
DIMENSION=50
NUM_DATA=100
LEARNING_RATE=3e-4
SEED=1
MODEL="tnpa"
WANDB_PROJECT="tnp-cmab"
WANDB_ENTITY=""

# Task Specifics
TASK="wheeler" # Options: wheeler, etc.

echo "Running CMAB with Model: $MODEL on Task: $TASK"

python3 contextual_bandits/main.py \
    --task "$TASK" \
    --model "$MODEL" \
    --dim "$DIMENSION" \
    --num-data "$NUM_DATA" \
    --lr "$LEARNING_RATE" \
    --num-iter "$NUM_ITER" \
    --seed "$SEED" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    "$@"
