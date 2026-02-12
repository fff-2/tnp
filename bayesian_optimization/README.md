# Bayesian Optimization

## 1-Dimensional BO

### Training

The 1D BO surrogate is the **same model** as the regression GP. Train it using the regression training script:

```bash
./scripts/train/train_gp.sh
```

There is no separate training script for 1D BO — the checkpoint saved at `results/gp/{model}/{expid}/ckpt.tar` is directly reused by the BO loop.

### Evaluation (BO Loop)

Run the actual Bayesian Optimization loop using a trained regression GP surrogate:

```bash
./scripts/eval/run_bo_1d.sh
```

If `--expid` is not specified, the latest **regression GP** experiment is used automatically.

| Argument | Description | Default |
|---|---|---|
| `--bo_mode` | `oracle` (true GP) or `models` (learned surrogate) | `oracle` |
| `--acquisition` | Acquisition function: `ucb` or `ei` | `ucb` |
| `--model` | Which trained model to use | `tnpa` |
| `--expid` | Experiment ID (auto-detected from regression GP if not set) | `None` |

```bash
# Run BO with trained TNP surrogate (auto-detects latest regression GP expid)
./scripts/eval/run_bo_1d.sh --bo_mode models --model tnpa

# Run BO with a specific regression GP experiment
./scripts/eval/run_bo_1d.sh --bo_mode models --model tnpa --expid 20260212-2030

# Run BO with oracle GP (baseline, no trained model needed)
./scripts/eval/run_bo_1d.sh --bo_mode oracle
```

### Evaluating the Surrogate (OOD)

To evaluate the regression GP surrogate on different kernels, use the regression eval script:

```bash
# In-distribution (RBF)
./scripts/eval/eval_gp.sh --eval_kernel rbf

# Out-of-distribution (Matérn)
./scripts/eval/eval_gp.sh --eval_kernel matern

# Out-of-distribution (Periodic)
./scripts/eval/eval_gp.sh --eval_kernel periodic
```

#### `--t_noise` (Student-t Noise)

```bash
./scripts/eval/eval_gp.sh --t_noise 0.1
```

---

## Multi-Dimensional BO

### Training

```bash
./scripts/train/train_bo_gp_highdim.sh
```

### Evaluation

Run the BO loop and choose the objective function:

```bash
./scripts/eval/run_bo_highdim.sh
```

| Argument | Description | Default |
|---|---|---|
| `--objective` | Objective function | `ackley` |
| `--dimension` | Input dimension | `2` |
| `--acquisition` | Acquisition function: `ucb` or `ei` | `ucb` |
| `--model` | Model to use (`gp` or TNP variants) | `tnpa` |

Available objectives: `ackley`, `cosine`, `rastrigin`, `dropwave`, `goldsteinprice`, `michalewicz`, `hartmann`.

```bash
# 2D Ackley with UCB
./scripts/eval/run_bo_highdim.sh --objective ackley --dimension 2

# 3D Hartmann with EI
./scripts/eval/run_bo_highdim.sh --objective hartmann --dimension 3 --acquisition ei
```
