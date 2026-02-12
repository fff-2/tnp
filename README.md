# Transformer Neural Processes: Uncertainty-Aware Meta Learning Via Sequence Modeling

This is the implementation of the paper [Transformer Neural Processes: Uncertainty-Aware Meta Learning Via Sequence Modeling](https://arxiv.org/abs/2207.04179) in Pytorch.

<img width="100%" src="./tnp.png">

## Install

First, clone the repository:

```
git clone https://github.com/tung-nd/TNP-pytorch.git
```

Then install the dependencies:

```
conda create -n tnp python==3.10
conda activate tnp
pip install -r requirements.txt
wandb login
```

## Usage

We provide modular shell scripts for training and evaluation in the `scripts/` directory.

For detailed instructions on each task, please refer to the specific READMEs:
- [Regression Tasks](regression/README.md)
- [Contextual Bandits](contextual_bandits/README.md)
- [Bayesian Optimization](bayesian_optimization/README.md)

### Training

To train models for different tasks, use the corresponding script in `scripts/train/`:

```bash
# CelebA
./scripts/train/train_celeba.sh

# EMNIST
./scripts/train/train_emnist.sh

# GP Regression
./scripts/train/train_gp.sh

# Contextual Bandits
./scripts/train/train_cmab.sh

# 1D Bayesian Optimization (GP surrogate training)
./scripts/train/train_bo_gp_1d.sh

# High-Dim Bayesian Optimization (GP surrogate training)
./scripts/train/train_bo_gp_highdim.sh
```

### Evaluation

Similarly, use the evaluation scripts:

```bash
# CelebA Evaluation
./scripts/eval/eval_celeba.sh

# EMNIST Evaluation
./scripts/eval/eval_emnist.sh

# GP Regression Evaluation
./scripts/eval/eval_gp.sh

# Contextual Bandits Evaluation
./scripts/eval/eval_cmab.sh

# 1D BO Loop
./scripts/eval/run_bo_1d.sh

# High-Dim BO Loop
./scripts/eval/run_bo_highdim.sh
```

You can customize the hyperparameters by editing the variable definitions at the top of each script.

## Acknowledgement

The implementation of the baselines is borrowed from the official code base of [Transformer Neural Processes: Uncertainty-Aware Meta Learning Via Sequence Modeling](https://github.com/tung-nd/TNP-pytorch).