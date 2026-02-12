### Training
First, we have to train TNPs on randomly sampled wheel data. Training is similar to meta regression.
```
../scripts/train/train_cmab.sh
```
If training for the first time, wheel data will be generated and saved in `datasets`. Model weights and logs will be saved in `results/train-all-R`.

### Evaluate
After training, we can run contextual bandit to evaluate the trained model.
```
../scripts/eval/eval_cmab.sh
```
Model weights according to `{expid}` will be loaded and evaluated. If running contextual bandit for the first time, evaluation data wil be generated and saved in `evalsets`. The results will be saved in `results/eval-all-R`.