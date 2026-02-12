#!/bin/bash

# Evaluation Configuration for Contextual Bandits (CMAB)
# This script is used to evaluate a trained CMAB model.

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
EXPID="default"
RESUME_PATH=""

echo "Running CMAB Evaluation with Model: $MODEL on Task: $TASK ExpId: $EXPID"

python3 contextual_bandits/main.py \
    --cmab_mode eval \
    --task "$TASK" \
    --model "$MODEL" \
    --dim "$DIMENSION" \
    --num-data "$NUM_DATA" \
    --lr "$LEARNING_RATE" \
    --num-iter "$NUM_ITER" \
    --seed "$SEED" \
    --expid "$EXPID" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    "$@"
