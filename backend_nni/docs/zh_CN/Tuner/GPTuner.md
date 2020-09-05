# NNI 中的 GP Tuner

## GP Tuner

贝叶斯优化会构建一个能最好的描述优化目标的后验分布函数（使用高斯过程）。 随着观测值的增加，后验分布会得到改善，会在参数空间中确定哪些范围值得进一步探索，哪一些不值得。

GP Tuner 被设计为通过最大化或最小化步数来找到最接近最优结果的参数组合。 GP Tuner 使用了代理优化问题（找到采集函数的最大值）。虽然这仍然是个难题，但成本更低（从计算的角度来看），并且适合于作为通用工具。 因此，贝叶斯优化适合于采样函数的成本非常高时来使用。

注意，搜索空间接受的类型包括 `randint`, `uniform`, `quniform`, `loguniform`, `qloguniform`，以及数值的 `choice`。

优化方法在 [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) 的第三章有详细描述。