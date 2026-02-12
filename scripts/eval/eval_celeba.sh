#!/bin/bash

# Evaluation Configuration for CelebA
MODEL="tnpa"  # Options: np, anp, cnp, canp, bnp, banp, tnpa, tnpd, tnpnd
# EXPID is not set here — defaults to the latest timestamped run if not passed via "$@"

# Eval Parameters
EVAL_NUM_BS=50
EVAL_BATCH_SIZE=16
EVAL_NUM_SAMPLES=50
EVAL_LOGFILE="eval_celeba.log"

echo "Running CelebA evaluation (eval_all_metrics) with Model: $MODEL"

python3 regression/celeba.py \
    --mode eval_all_metrics \
    --model "$MODEL" \
    --eval_num_bs "$EVAL_NUM_BS" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_num_samples "$EVAL_NUM_SAMPLES" \
    --eval_logfile "$EVAL_LOGFILE" \
    "$@"

echo "Running CelebA plot with Model: $MODEL"

python3 regression/celeba.py \
    --mode plot \
    --model "$MODEL" \
    "$@"
