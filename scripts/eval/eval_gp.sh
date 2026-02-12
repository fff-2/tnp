#!/bin/bash

# Evaluation Configuration for GP Regression
MODEL="tnpd"
# EXPID is not set here — defaults to the latest timestamped run if not passed via "$@"

# Eval Parameters
EVAL_NUM_BATCHES=3000
EVAL_BATCH_SIZE=16
EVAL_NUM_SAMPLES=50

echo "Running GP Regression evaluation (eval_all_metrics) with Model: $MODEL"

python3 regression/gp.py \
    --mode eval_all_metrics \
    --model "$MODEL" \
    --eval_num_batches "$EVAL_NUM_BATCHES" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_num_samples "$EVAL_NUM_SAMPLES" \
    "$@"

echo "Running GP Regression plot with Model: $MODEL"

python3 regression/gp.py \
    --mode plot \
    --model "$MODEL" \
    "$@"
