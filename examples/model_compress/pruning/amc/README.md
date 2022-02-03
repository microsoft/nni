# AMCPruner Example
This example shows us how to use AMCPruner example.

## Step 1: train a model for pruning
Run following command to train a mobilenetv2 model:
```bash
python3 amc_train.py --model_type mobilenetv2 --n_epoch 50
```
Once finished, saved checkpoint file can be found at:
```
logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth
```

## Pruning with AMCPruner
Run following command to prune the trained model:
```bash
python3 amc_search.py --model_type mobilenetv2 --ckpt logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth
```
Once finished, pruned model and mask can be found at:
```
logs/mobilenetv2_cifar10_r0.5_search-run2
```

## Finetune pruned model
Run `amc_train.py` again with `--ckpt` and `--mask` to speedup and finetune the pruned model:
```bash
python3 amc_train.py --model_type mobilenetv2 --ckpt logs/mobilenetv2_cifar10_r0.5_search-run2/best_model.pth --mask logs/mobilenetv2_cifar10_r0.5_search-run2/best_mask.pth --n_epoch 100
```

# RM+AMCPruner Example([RMNet: Equivalently Removing Residual Connection from Networks](https://arxiv.org/abs/2111.00687))
This example shows us how to use RM Operation and AMCPruner example.


## Step 1: train a MobileNetV2
Run following command to train a mobilenetv2 model:
```bash
python3 amc_train.py --model_type mobilenetv2
```
Once finished, saved checkpoint file can be found at:
```
logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth
```

## Step 2: Convert MobileNetV2 to MobileNetV1 and Finetune it
Run following command to convert MobileNetV2 to MobileNetV1 and finetune the MobileNetV1:
```bash
python3 amc_train.py --model_type rmnetv2 --ckpt_path logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth
```
Once finished, saved checkpoint file can be found at:
```
logs/rmnetv2_cifar10_train-run2/ckpt.best.pth
```

## Pruning the MobileNetV1 with AMCPruner
Run following command to prune the trained model:
```bash
python3 amc_search.py --model_type rmnetv1 --ckpt_path logs/rmnetv2_cifar10_train-run2/ckpt.best.pth
```
Once finished, pruned model and mask can be found at:
```
logs/rmnetv1_cifar10_r0.5_search-run3
```

## Finetune pruned MobileNetV1
Run `amc_train.py` again with `--ckpt` and `--mask` to speedup and finetune the pruned model:
```bash
python3 amc_train.py --model_type rmnetv1 --ckpt_path logs/rmnetv1_cifar10_r0.5_search-run3/best_model.pth --mask_path logs/rmnetv1_cifar10_r0.5_search-run3/best_mask.pth
```
Once finished, saved checkpoint file can be found at:
```
logs/rmnetv1_cifar10_train-run4
```
## Results
In the following tables, RM+AMC first convert a pruned MobileNetV2 to MobileNetV1 and prune the MobileNetV1.
Checkpoints are available at [Baidu Cloud(提取码:1jw2)](https://pan.baidu.com/s/1tCq7JWRKr3BuwgBlyF7ZPg)

Comparing on Tesla V100
|Dataset| Method | Speed(Imgs/Sec) | Acc(%)|
|----------| ----------------- | ----------------- | ---------- |
||Baseline|3752|71.79|
||AMC(0.7)|4949|70.84|
|ImageNet|RM+AMC(0.75)|5120|**73.21**|
||RM+AMC(0.7)|5238|72.63|
||RM+AMC(0.6)|5675|71.88|
||RM+AMC(0.5)|**6250**|71.01|

Comparing on Multiple Hardwares.
|Method|Acc(%)|A100(fps)|V100(fps)|P40(fps)|3090(fps)|2080Ti(fps)|1060(fps)|T4(fps)|CPU(fps)|
|---|---|---|---|---|----|-----|-----|----|---------|
|Baseline|71.79|6253|3759|1131|4136|2650|667|550|43|
|AMC|70.84|8234|4982|1520|5421|3474|891|732|54|
|**RM+AMC**|**71.88**|**9671**|**5675**|**1860**|**6346**|**4333**|**990**|**863**|**77**|
