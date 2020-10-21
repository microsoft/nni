# EfficientNet

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

Use Grid search to find the best combination of alpha, beta and gamma for EfficientNet-B1, as discussed in Section 3.3 in paper. Search space, tuner, configuration examples are provided here.

## Instructions

[Example code](https://github.com/microsoft/nni/tree/v1.9/examples/trials/efficientnet)

1. Set your working directory here in the example code directory.
2. Run `git clone https://github.com/ultmaster/EfficientNet-PyTorch` to clone this modified version of [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch). The modifications were done to adhere to the original [Tensorflow version](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) as close as possible (including EMA, label smoothing and etc.); also added are the part which gets parameters from tuner and reports intermediate/final results. Clone it into `EfficientNet-PyTorch`; the files like `main.py`, `train_imagenet.sh` will appear inside, as specified in the configuration files.
3. Run `nnictl create --config config_local.yml` (use `config_pai.yml` for OpenPAI) to find the best EfficientNet-B1. Adjust the training service (PAI/local/remote), batch size in the config files according to the environment.

For training on ImageNet, read `EfficientNet-PyTorch/train_imagenet.sh`. Download ImageNet beforehand and extract it adhering to [PyTorch format](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet) and then replace `/mnt/data/imagenet` in with the location of the ImageNet storage. This file should also be a good example to follow for mounting ImageNet into the container on OpenPAI.

## Results

The follow image is a screenshot, demonstrating the relationship between acc@1 and alpha, beta, gamma.

![](../../img/efficientnet_search_result.png)
