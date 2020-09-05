# Metis Tuner

## Metis Tuner

[Metis](https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/) 相对于别的调优算法，有几个优势。 大多数调参工具仅仅预测最优配置，而 Metis 具有两个输出，最优配置的预测， 以及下一次 Trial 的建议。 不再需要随机猜测！

大多数工具假设训练集没有噪声数据，但 Metis 会知道是否需要对某个超参重新采样。

大多数工具都有着重于在已有结果上继续发展的问题，而 Metis 的搜索策略可以在探索，发展和重新采样（可选）中进行平衡。

Metis 属于基于序列的贝叶斯优化 (SMBO) 算法的类别，它也基于贝叶斯优化框架。 为了对超参-性能空间建模，Metis 同时使用了高斯过程（Gaussian Process）和高斯混合模型（GMM）。 由于每次 Trial 都可能有很高的时间成本，Metis 大量使用了已有模型来进行推理计算。 在每次迭代中，Metis 执行两个任务：

* 在高斯过程空间中找到全局最优点。 这一点表示了最佳配置。

* 它会标识出下一个超参的候选项。 这是通过对隐含信息的探索、挖掘和重采样来实现的。

此 Tuner 搜索空间仅接受 `quniform`，`uniform`，`randint` 和数值的 `choice` 类型。

更多详情，参考[论文](https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/)。