Lottery Ticket 假设
===

## 介绍

[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) 是主要衡量和分析的论文，它提供了非常有意思的见解。 为了在 NNI 上支持此算法，主要实现了找到*获奖彩票*的训练方法。

本文中，作者使用叫做*迭代*修剪的方法：
> 1. 随机初始化一个神经网络 f(x;theta_0) (其中 theta_0 为 D_{theta}).
> 2. 将网络训练 j 次，得出参数 theta_j。
> 3. 在 theta_j 修剪参数的 p%，创建掩码 m。
> 4. 将其余参数重置为 theta_0 的值，创建获胜彩票 f(x;m*theta_0)。
> 5. 重复步骤 2、3 和 4。

如果配置的最终稀疏度为 P (e.g., 0.8) 并且有 n 次修建迭代，每次迭代修剪前一轮中剩余权重的 1-(1-P)^(1/n)。

## 重现结果

在重现时，在 MNIST 使用了与论文相同的配置。 [此处](https://github.com/microsoft/nni/tree/master/examples/model_compress/lottery_torch_mnist_fc.py)为实现代码。 在次实验中，修剪了10次，在每次修剪后，训练了 50 个 epoch。

![](../../img/lottery_ticket_mnist_fc.png)

上图展示了全连接网络的结果。 `round0-sparsity-0.0` 是没有剪枝的性能。 与论文一致，修剪约 80% 也能获得与不修剪时相似的性能，收敛速度也会更快。 如果修剪过多（例如，大于 94%），则精度会降低，收敛速度会稍慢。 与本文稍有不同，论文中数据的趋势比较明显。
