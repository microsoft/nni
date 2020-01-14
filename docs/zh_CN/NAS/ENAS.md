# ENAS

## 介绍

论文 [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268) 通过在子模型之间共享参数来加速 NAS 过程。 在 ENAS 中，Contoller 学习在大的计算图中搜索最有子图的方式来发现神经网络。 Controller 通过梯度策略训练，从而选择出能在验证集上有最大期望奖励的子图。 同时对与所选子图对应的模型进行训练，以最小化规范交叉熵损失。

Implementation on NNI is based on the [official implementation in Tensorflow](https://github.com/melodyguan/enas), including a general-purpose Reinforcement-learning controller and a trainer that trains target network and this controller alternatively. Following paper, we have also implemented macro and micro search space on CIFAR10 to demonstrate how to use these trainers. Since code to train from scratch on NNI is not ready yet, reproduction results are currently unavailable.

## Examples

### CIFAR10 Macro/Micro Search Space

[示例代码](https://github.com/microsoft/nni/tree/master/examples/nas/enas)

```bash
＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
git clone https://github.com/Microsoft/nni.git

# 搜索最好的网络架构
cd examples/nas/enas

# 在 Macro 搜索空间中搜索
python3 search.py --search-for macro

# 在 Micro 搜索空间中搜索
python3 search.py --search-for micro

# 查看更多选项
python3 search.py -h
```

## 参考

### PyTorch

```eval_rst
..  autoclass:: nni.nas.pytorch.enas.EnasTrainer
    :members:

    .. automethod:: __init__

..  autoclass:: nni.nas.pytorch.enas.EnasMutator
    :members:

    .. automethod:: __init__
```
