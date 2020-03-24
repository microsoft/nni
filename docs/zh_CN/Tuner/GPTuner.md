# NNI 中的 GP Tuner

## GP Tuner

Bayesian optimization works by constructing a posterior distribution of functions (a Gaussian Process) that best describes the function you want to optimize. 随着观测值的增加，后验分布会得到改善，会在参数空间中确定哪些范围值得进一步探索，哪一些不值得。

GP Tuner 被设计为通过最大化或最小化步数来找到最接近最优结果的参数组合。 To do so, this method uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) to solve, and it's amenable to common tools. Therefore, Bayesian Optimization is suggested for situations where sampling the function to be optimized is very expensive.

Note that the only acceptable types within the search space are `randint`, `uniform`, `quniform`, `loguniform`, `qloguniform`, and numerical `choice`.

优化方法在 [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) 的第三章有详细描述。