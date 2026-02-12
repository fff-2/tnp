#!/bin/bash

# Evaluation Configuration for High-Dim BO Loops
OBJECTIVE="ackley" # Options: ackley, cosine, rastrigin, etc.
DIMENSION=2
ACQ_FUNC="ucb"   # Options: ucb, ei
NUM_ITER=100
SEED=1

# Model Configuration (if applicable, highdim_bo usually runs 'bo' mode which can use GP or other models)
MODEL="tnpa" # Options: gp, or TNP models
# EXPID is not set here — defaults to the latest timestamped run if not passed via "$@"

echo "Running High-Dim BO Loop Objective: $OBJECTIVE Dim: $DIMENSION Acq: $ACQ_FUNC"

python3 bayesian_optimization/highdim_bo.py \
    --mode bo \
    --objective "$OBJECTIVE" \
    --dimension "$DIMENSION" \
    --acquisition "$ACQ_FUNC" \
    --num_iter "$NUM_ITER" \
    --seed "$SEED" \
    --model "$MODEL" \
    "$@"
