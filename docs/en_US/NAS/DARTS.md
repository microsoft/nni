# DARTS on NNI

## Introduction

The paper [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) addresses the scalability challenge of architecture search by formulating the task in a differentiable manner. Their method is based on the continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent

To implement, authors optimize the network weights and architecture weights alternatively in mini-batches. They further explore the possibility that uses second order optimization (unroll) instead of first order, to improve the performance.

Implementation on NNI is based on the [official implementation](https://github.com/quark0/darts) and a [popular 3rd-party repo](https://github.com/khanrc/pt.darts). So far, first and second order optimization and training from scratch on CIFAR10 have been implemented.

## Reproduce Results

To reproduce the results in the paper, we do experiments with first and second order optimization. Due to the time limit, we retrain *only the best architecture* derived from the search phase and we repeat the experiment *only once*. Our results is currently on par with the results reported in paper. We will add more results later when ready.

|                        | In paper      | Reproduction |
| ---------------------- | ------------- | ------------ |
| First order (CIFAR10)  | 3.00 +/- 0.14 | 2.78         |
| Second order (CIFAR10) | 2.76 +/- 0.09 | 2.89         |
