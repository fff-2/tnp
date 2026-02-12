### 1-dimensional BO
---
### Training
Training is exactly the same to meta regression.
```
../scripts/train/train_bo_gp_1d.sh
```

### Evaluation
Run BO using a trained model.
```
../scripts/eval/run_bo_1d.sh
```

## Multi-dimensional BO
---
### Training
First, generate the training dataset (if needed, handled by script), and then train.
```
../scripts/train/train_bo_gp_highdim.sh
```

### Evaluation

Run `run_bo_highdim.sh`.   
Please choose objective function (e.g. `ackley`) by editing the script or passing arguments.

```
../scripts/eval/run_bo_highdim.sh --objective=ackley
```
