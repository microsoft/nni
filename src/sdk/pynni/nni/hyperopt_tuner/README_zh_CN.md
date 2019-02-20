# TPE, Random Search, Anneal Tuners

## TPE

Tree-structured Parzen Estimator (TPE) 是一种 sequential model-based optimization（SMBO，即基于序列模型优化）的方法。 SMBO 方法根据历史指标数据来按顺序构造模型，来估算超参的性能，随后基于此模型来选择新的超参。 TPE 方法对 P(x|y) 和 P(y) 建模，其中 x 表示超参，y 表示相关的评估指标。 P(x|y) 通过变换超参的生成过程来建模，用非参数密度（non-parametric densities）代替配置的先验分布。 细节可参考 [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)。 ​

## Random Search（随机搜索）

[Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) 中介绍了随机搜索惊人的简单和效果。 建议当不清楚超参的先验分布时，采用随机搜索作为基准。

## Anneal（退火算法）

这种简单的退火算法从先前的采样开始，会越来越靠近发现的最佳点取样。 此算法是随机搜索的简单变体，利用了响应面的平滑性。 退火率不是自适应的。