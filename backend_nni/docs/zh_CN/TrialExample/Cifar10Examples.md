# CIFAR-10 示例

## 概述

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 分类是机器学习中常用的基准问题。 CIFAR-10 数据集是图像的集合。 它也是机器学习领域最常用的数据集之一，包含 60000 万张 32x32 的图像，共有 10 个分类。 因此以 CIFAR-10 分类为例来介绍 NNI 的用法。

### **目标**

总所周知，模型 optimizer (优化器）的选择直接影响了最终指标的性能。 本教程的目标是**调优出性能更好的优化器**，从而为图像识别训练出一个相对较小的卷积网络（CNN）。

本例中，选择了以下常见的深度学习优化器：

> "SGD", "Adadelta", "Adagrad", "Adam", "Adamax"

### **实验**

#### 准备

此示例需要安装 PyTorch。 PyTorch 安装包需要选择所基于的 Python 和 CUDA 版本。

这是环境 python==3.5 且 cuda == 8.0 的示例，然后用下列命令来安装 [ PyTorch](https://pytorch.org/)：

```bash
python3 -m pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
python3 -m pip install torchvision
```

#### NNI 与 CIFAR-10

**搜索空间**

正如本文目标，要为 CIFAR-10 找到最好的`优化器`。 使用不同的优化器时，还要相应的调整`学习率`和`网络架构`。 因此，选择这三个参数作为超参，并创建如下的搜索空间。

```json
{
    "lr":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    "optimizer":{"_type":"choice", "_value":["SGD", "Adadelta", "Adagrad", "Adam", "Adamax"]},
    "model":{"_type":"choice", "_value":["vgg", "resnet18", "googlenet", "densenet121", "mobilenet", "dpn92", "senet18"]}
}
```

*实现代码：[search_space.json](https://github.com/Microsoft/nni/blob/master/examples/trials/cifar10_pytorch/search_space.json)*

**Trial（尝试）**

这是超参集合的训练代码，关注以下几点：

* 使用 `nni.get_next_parameter()` 来获取下一组训练的超参组合。
* 使用 `nni.report_intermediate_result(acc)` 在每个 epoch 结束时返回中间结果。
* 使用 `nni.report_final_result(acc)` 在每个 Trial 结束时返回最终结果。

*实现代码：[main.py](https://github.com/Microsoft/nni/blob/master/examples/trials/cifar10_pytorch/main.py)*

还可直接修改现有的代码来支持 Nni，参考：[如何实现 Trial](Trials.md)。

**配置**

这是在本机运行 Experiment 的示例（多GPU）：

代码：[examples/trials/cifar10_pytorch/config.yml](https://github.com/Microsoft/nni/blob/master/examples/trials/cifar10_pytorch/config.yml)

这是在 OpenPAI 上运行 Experiment 的示例：

代码：[examples/trials/cifar10_pytorch/config_pai.yml](https://github.com/Microsoft/nni/blob/master/examples/trials/cifar10_pytorch/config_pai.yml)

*完整示例：[examples/trials/cifar10_pytorch/](https://github.com/Microsoft/nni/tree/master/examples/trials/cifar10_pytorch)*

#### 运行 Experiment

以上即为 Experiment 的代码介绍，**从命令行运行 config.yml 文件来开始 Experiment**。

```bash
nnictl create --config nni/examples/trials/cifar10_pytorch/config.yml
```