#!/bin/bash

# Evaluation Configuration for GP Regression
MODEL="tnpd"
EXPID="default"
RESUME_PATH=""

# Eval Parameters
EVAL_NUM_BATCHES=3000
EVAL_BATCH_SIZE=16
EVAL_NUM_SAMPLES=50
EVAL_LOGFILE="eval_gp.log"

echo "Running GP Regression evaluation with Model: $MODEL ExpId: $EXPID"

python3 regression/gp.py \
    --mode eval \
    --model "$MODEL" \
    --expid "$EXPID" \
    --eval_num_batches "$EVAL_NUM_BATCHES" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_num_samples "$EVAL_NUM_SAMPLES" \
    --eval_logfile "$EVAL_LOGFILE" \
    --resume "$RESUME_PATH" \
    "$@"
