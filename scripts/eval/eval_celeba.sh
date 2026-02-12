#!/bin/bash

# Evaluation Configuration for CelebA
MODEL="tnpa"  # Options: np, anp, cnp, canp, bnp, banp, tnpa, tnpd, tnpnd
EXPID="default"
RESUME_PATH="" # Path to checkpoint if needed, or rely on internal logic finding it via expid

# Eval Parameters
EVAL_NUM_BS=50
EVAL_BATCH_SIZE=16
EVAL_NUM_SAMPLES=50
EVAL_LOGFILE="eval_celeba.log"

echo "Running CelebA evaluation with Model: $MODEL ExpId: $EXPID"

python3 regression/celeba.py \
    --mode eval \
    --model "$MODEL" \
    --expid "$EXPID" \
    --eval_num_bs "$EVAL_NUM_BS" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_num_samples "$EVAL_NUM_SAMPLES" \
    --eval_logfile "$EVAL_LOGFILE" \
    --resume "$RESUME_PATH" \
    "$@"
