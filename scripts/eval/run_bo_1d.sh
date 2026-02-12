#!/bin/bash

# Evaluation Configuration for 1D BO Loops
# This runs the actual Bayesian Optimization loop using a trained surrogate or oracle.
BO_MODE="oracle" # Options: oracle, models
ACQ_FUNC="ucb"   # Options: ucb, ei
NUM_ITER=100
SEED=1

# Model Configuration (if BO_MODE=models)
MODEL="tnpa"
# EXPID is not set here — defaults to the latest timestamped run if not passed via "$@"

echo "Running 1D BO Loop Mode: $BO_MODE Acq: $ACQ_FUNC"

python3 bayesian_optimization/1d_bo.py \
    --mode bo \
    --bo_mode "$BO_MODE" \
    --acquisition "$ACQ_FUNC" \
    --num_task 100 \
    --num_iter "$NUM_ITER" \
    --seed "$SEED" \
    --model "$MODEL" \
    "$@"
