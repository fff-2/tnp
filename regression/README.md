## 1D Regression

---
### Training
```
../scripts/train/train_gp.sh
```
The config of hyperparameters of each model is saved in `configs/gp`. If training for the first time, evaluation data will be generated and saved in `evalsets/gp`. Model weights and logs are saved in `results/gp/{model}/{expid}`.

### Evaluation
```
../scripts/eval/eval_gp.sh
```
Note that you have to specify `{expid}` correctly in the script or pass it as an argument. The model will load weights from `results/gp/{model}/{expid}` to evaluate.

## CelebA Image Completion
---

### Prepare data
Download [img_align_celeba.zip](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg) and unzip. Download [list_eval_partitions.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pdjI3dmwtNm5jRkE) and [identity_CelebA.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs). Place downloaded files in `datasets/celeba` folder. Run `python data/celeba.py` to preprocess the data.

### Training
```
../scripts/train/train_celeba.sh
```

### Evaluation
```
../scripts/eval/eval_celeba.sh
```
If evaluating for the first time, evaluation data will be generated and saved in `evalsets/celeba`.

## EMNIST Image Completion
---

### Training
```
../scripts/train/train_emnist.sh
```
If training for the first time, EMNIST training data will automatically downloaded and saved in `datasets/emnist`.

### Evaluation
```
../scripts/eval/eval_emnist.sh
```
If evaluating for the first time, evaluation data will be generated and saved in `evalsets/emnist`.