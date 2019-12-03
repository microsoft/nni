# NNI 中的 DARTS

## 介绍

论文 [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) 通过可微分的方式来解决架构搜索中的伸缩性挑战。 此方法基于架构的连续放松的表示，从而允许在架构搜索时能使用梯度下降。

为了实现，作者在小批量中交替优化网络权重和架构权重。 还进一步探讨了使用二阶优化（unroll）来替代一阶，来提高性能的可能性。

NNI 的实现基于[官方实现](https://github.com/quark0/darts)以及一个[第三方实现](https://github.com/khanrc/pt.darts)。 So far, first and second order optimization and training from scratch on CIFAR10 have been implemented.

## Reproduce Results

To reproduce the results in the paper, we do experiments with first and second order optimization. Due to the time limit, we retrain *only the best architecture* derived from the search phase and we repeat the experiment *only once*. Our results is currently on par with the results reported in paper. We will add more results later when ready.

|                        | In paper      | Reproduction |
| ---------------------- | ------------- | ------------ |
| First order (CIFAR10)  | 3.00 +/- 0.14 | 2.78         |
| Second order (CIFAR10) | 2.76 +/- 0.09 | 2.89         |
