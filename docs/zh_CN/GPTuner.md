# NNI 中的 GP Tuner

## GP Tuner

Bayesian optimization works by constructing a posterior distribution of functions (Gaussian Process here) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not.

GP Tuner is designed to minimize/maximize the number of steps required to find a combination of parameters that are close to the optimal combination. To do so, this method uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) and common tools can be employed. 因此，贝叶斯优化适合于采样函数的成本非常高时来使用。

优化方法在 [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) 的第三章有详细描述。