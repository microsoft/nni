# NNI 中的 ENAS

## 介绍

论文 [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268) 通过在子模型之间共享参数来加速 NAS 过程。 在 ENAS 中，Contoller 学习在大的计算图中搜索最有子图的方式来发现神经网络。 Controller 通过梯度策略训练，从而选择出能在验证集上有最大期望奖励的子图。 同时对与所选子图对应的模型进行训练，以最小化规范交叉熵损失。

NNI 的实现基于 [Tensorflow 的官方实现](https://github.com/melodyguan/enas)，包括了 CIFAR10 上的 Macro/Micro 搜索空间。 NNI 中从头训练的代码还未完成，当前还没有重现结果。
