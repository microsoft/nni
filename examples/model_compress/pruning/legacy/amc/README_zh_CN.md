# AMCPruner 示例
此示例将说明如何使用 AMCPruner。

## 步骤一：训练模型
运行以下命令来训练 mobilenetv2 模型：
```bash
python3 amc_train.py --model_type mobilenetv2 --n_epoch 50
```
训练完成之后，检查点文件被保存在这里：
```
logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth
```

## 使用 AMCPruner 剪枝
运行以下命令对模型进行剪枝：
```bash
python3 amc_search.py --model_type mobilenetv2 --ckpt logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth
```
完成之后，剪枝后的模型和掩码文件被保存在：
```
logs/mobilenetv2_cifar10_r0.5_search-run2
```

## 微调剪枝后的模型
加上 `--ckpt` 和 `--mask` 参数，再次运行 `amc_train.py` 命令去加速和微调剪枝后的模型。
```bash
python3 amc_train.py --model_type mobilenetv2 --ckpt logs/mobilenetv2_cifar10_r0.5_search-run2/best_model.pth --mask logs/mobilenetv2_cifar10_r0.5_search-run2/best_mask.pth --n_epoch 100
```
