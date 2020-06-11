# EfficientNet

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

如论文中 3.3 所述，使用遍历搜索来找到 EfficientNet-B1 的 alpha, beta 和 gamma 的最好组合。 搜索空间，Tuner，配置示例如下。

## 说明

[示例代码](https://github.com/microsoft/nni/tree/master/examples/trials/efficientnet)

1. 将示例代码目录设为当前工作目录。
2. 运行 `git clone https://github.com/ultmaster/EfficientNet-PyTorch` 来克隆修改过的 [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)。 修改尽可能接近原始的 [TensorFlow 版本](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) （包括 EMA，标记平滑度等等。）；另外添加了代码从 Tuner 获取参数并回调中间和最终结果。 将其 clone 至 `EfficientNet-PyTorch`；`main.py`，`train_imagenet.sh` 等文件会在配置文件中指定的路径。
3. 运行 `nnictl create --config config_local.yml` (OpenPAI 可使用 `config_pai.yml`) 来找到最好的 EfficientNet-B1。 根据环境来调整训练平台（OpenPAI、本机、远程），batch size。

在 ImageNet 上的训练，可阅读 `EfficientNet-PyTorch/train_imagenet.sh`。 下载 ImageNet，并参考 [PyTorch 格式](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet) 来解压，然后将 `/mnt/data/imagenet` 替换为 ImageNet 的路径。 此文件也是如何将 ImageNet 挂载到 OpenPAI 容器的示例。

## 结果

下图展示了 acc@1 和 alpha、beta、gamma 之间的关系。

![](../../img/efficientnet_search_result.png)
