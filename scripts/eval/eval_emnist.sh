#!/bin/bash

# Evaluation Configuration for EMNIST
MODEL="tnpa"
# EXPID is not set here — defaults to the latest timestamped run if not passed via "$@"

# Eval Parameters
EVAL_NUM_BS=50
EVAL_BATCH_SIZE=16
EVAL_NUM_SAMPLES=50
EVAL_LOGFILE="eval_emnist.log"

echo "Running EMNIST evaluation (eval_all_metrics) with Model: $MODEL"

python3 regression/emnist.py \
    --mode eval_all_metrics \
    --model "$MODEL" \
    --eval_num_bs "$EVAL_NUM_BS" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_num_samples "$EVAL_NUM_SAMPLES" \
    --eval_logfile "$EVAL_LOGFILE" \
    "$@"

echo "Running EMNIST plot with Model: $MODEL"

python3 regression/emnist.py \
    --mode plot \
    --model "$MODEL" \
    "$@"
