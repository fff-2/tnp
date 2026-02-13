# Contextual Bandits (Wheel Bandit)

## Training

Train TNPs on randomly sampled wheel bandit data:

```bash
./scripts/train/train_cmab.sh
```

If training for the first time, wheel data will be generated and saved in `datasets/`. Model weights and logs will be saved in `results/{data}/train-{reward}-R/{model}/{expid}/`.

**Key training arguments:**

| Argument            | Description                               | Default  |
| ------------------- | ----------------------------------------- | -------- |
| `--model`           | Model architecture                        | `tnpa`   |
| `--expid`           | Experiment ID (auto-generated if not set) | `None`   |
| `--cmab_train_seed` | Random seed                               | `0`      |
| `--num_epochs`      | Total training epochs                     | `100000` |
| `--resume`          | Resume training from checkpoint           | `None`   |

W&B logging is active during training.

## Evaluation

After training, run the contextual bandit evaluation loop:

```bash
./scripts/eval/eval_cmab.sh
```

If `--expid` is not specified, the latest experiment is used automatically.

The evaluation runs the TNP model (and a uniform baseline) across multiple seeds, then generates comparison plots.

| Argument                 | Description                                   | Default |
| ------------------------ | --------------------------------------------- | ------- |
| `--cmab_eval_method`     | Action selection: `mean`, `ucb`, or `perturb` | `mean`  |
| `--cmab_eval_seed_start` | Start seed for eval runs                      | `1`     |
| `--cmab_eval_seed_end`   | End seed for eval runs                        | `5`     |

Results (cumulative regret plots, logs) are saved in `results/{data}/eval-{reward}-R/` and `results/{data}/plot-{reward}-R/`.
