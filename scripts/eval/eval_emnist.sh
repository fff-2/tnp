#!/bin/bash

# Evaluation Configuration for EMNIST
MODEL="tnpa"
EXPID="default"
RESUME_PATH=""

# Eval Parameters
EVAL_NUM_BS=50
EVAL_BATCH_SIZE=16
EVAL_NUM_SAMPLES=50
EVAL_LOGFILE="eval_emnist.log"

echo "Running EMNIST evaluation with Model: $MODEL ExpId: $EXPID"

python3 regression/emnist.py \
    --mode eval \
    --model "$MODEL" \
    --expid "$EXPID" \
    --eval_num_bs "$EVAL_NUM_BS" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_num_samples "$EVAL_NUM_SAMPLES" \
    --eval_logfile "$EVAL_LOGFILE" \
    --resume "$RESUME_PATH" \
    "$@"
