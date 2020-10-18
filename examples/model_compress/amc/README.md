# AMCPruner Example
This example shows us how to use AMCPruner example.

## Step 1: train a model for pruning
Run following command to train a mobilenetv2 model:
```bash
python3 amc_train.py --model_type mobilenetv2 --n_epochs 50
```
Once finished, saved checkpoint file can be found at:
```
logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth.tar
```

## Pruning with AMCPruner
Run following command to prune the trained model:
```bash
python amc_search.py --model_type mobilenetv2 --ckpt logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth.tar
```
Once finished, pruned model and mask can be found at:
```
logs/mobilenetv2_cifar10_r0.5_search-run2
```

## Finetune pruned model
Run `amc_train.py` again with `--ckpt` and `--mask` to finetune the pruned model:
```
python amc_train.py --model_type mobilenetv2 --ckpt logs/mobilenetv2_cifar10_r0.5_search-run2/best_model.pth --mask logs/mobilenetv2_cifar10_r0.5_search-run2/best_mask.pth --n_epoch 100
```
