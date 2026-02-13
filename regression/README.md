# Regression Tasks

## 1D GP Regression

### Training

```bash
./scripts/train/train_gp.sh
```

The config of hyperparameters of each model is saved in `configs/gp`. If training for the first time, evaluation data will be generated and saved in `evalsets/gp`. Model weights and logs are saved in `results/gp/{model}/{expid}`.

**Key training arguments:**

| Argument       | Description                                                 | Default  |
| -------------- | ----------------------------------------------------------- | -------- |
| `--model`      | Model architecture (see main README for options)            | `tnpd`   |
| `--expid`      | Experiment ID. Auto-generated as `YYYYMMDD-HHMM` if not set | `None`   |
| `--train_seed` | Random seed for training reproducibility                    | `0`      |
| `--num_steps`  | Total training steps                                        | `100000` |
| `--lr`         | Learning rate                                               | `5e-4`   |
| `--resume`     | Resume from checkpoint — pass the `expid` to load from      | `None`   |

#### Resuming Training

```bash
./scripts/train/train_gp.sh --expid 20260212-2030 --resume 20260212-2030
```

This loads model, optimizer, and scheduler states from `results/gp/tnpd/20260212-2030/ckpt.tar` and continues training from the last saved step.

#### Seed Arguments

- `--train_seed` controls randomness during training (data sampling, model initialization). Default `0` provides deterministic runs.
- `--eval_seed` controls evaluation set generation. Same seed = same eval data.

To report variance across runs, train with different seeds:

```bash
./scripts/train/train_gp.sh --train_seed 0 --expid seed0
./scripts/train/train_gp.sh --train_seed 1 --expid seed1
./scripts/train/train_gp.sh --train_seed 2 --expid seed2
```

### Evaluation

```bash
./scripts/eval/eval_gp.sh
```

The eval script runs **two modes sequentially**:

1. `eval_all_metrics` — computes accuracy (MAE, RMSE, R², etc.), calibration, sharpness, and log-likelihood
2. `plot` — saves prediction plots to `results/gp/{model}/{expid}/`

If `--expid` is not specified, the script automatically finds the **latest experiment** in the results directory.

```bash
# Evaluate a specific experiment
./scripts/eval/eval_gp.sh --expid 20260212-2030

# Evaluate the latest experiment (auto-detected)
./scripts/eval/eval_gp.sh
```

#### `--eval_kernel` (Out-of-Distribution Evaluation)

By default, models trained on RBF kernel data are evaluated on the same kernel. To test **out-of-distribution** generalization, use `--eval_kernel`:

```bash
# Evaluate on Matérn kernel (OOD)
./scripts/eval/eval_gp.sh --eval_kernel matern --expid 20260212-2030

# Evaluate on Periodic kernel (OOD)
./scripts/eval/eval_gp.sh --eval_kernel periodic --expid 20260212-2030
```

| `--eval_kernel` | Description                                                 |
| --------------- | ----------------------------------------------------------- |
| `rbf`           | RBF (Squared Exponential) kernel — default, in-distribution |
| `matern`        | Matérn 5/2 kernel — out-of-distribution                     |
| `periodic`      | Periodic kernel — out-of-distribution                       |

#### `--t_noise` (Student-t Noise)

Add heavy-tailed Student-t noise to evaluation data for robustness testing:

```bash
./scripts/eval/eval_gp.sh --t_noise 0.1 --expid 20260212-2030
```

---

## CelebA Image Completion

### Prepare Data

Download [img_align_celeba.zip](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg) and unzip. Download [list_eval_partitions.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pdjI3dmwtNm5jRkE) and [identity_CelebA.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs). Place downloaded files in `datasets/celeba` folder.
Run

```bash
cd regression
python -m data.celeba
```

### Training

```bash
./scripts/train/train_celeba.sh
```

### Evaluation

```bash
./scripts/eval/eval_celeba.sh
```

The eval script runs `eval_all_metrics` followed by `plot` to save figures. If `--expid` is not specified, the latest experiment is used automatically. If evaluating for the first time, evaluation data will be generated and saved in `evalsets/celeba`.

---

## EMNIST Image Completion

### Training

```bash
./scripts/train/train_emnist.sh
```

If training for the first time, EMNIST training data will be automatically downloaded and saved in `datasets/emnist`.

### Evaluation

```bash
./scripts/eval/eval_emnist.sh
```

The eval script runs `eval_all_metrics` followed by `plot` to save figures. If `--expid` is not specified, the latest experiment is used automatically. If evaluating for the first time, evaluation data will be generated and saved in `evalsets/emnist`.
