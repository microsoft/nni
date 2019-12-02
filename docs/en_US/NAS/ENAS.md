# ENAS on NNI

## Introduction

The paper [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268) uses parameter sharing between child models to accelerate the NAS process. In ENAS, a controller learns to discover neural network architectures by searching for an optimal subgraph within a large computational graph. The controller is trained with policy gradient to select a subgraph that maximizes the expected reward on the validation set. Meanwhile the model corresponding to the selected subgraph is trained to minimize a canonical cross entropy loss.

Implementation on NNI is based on the [official implementation in Tensorflow](https://github.com/melodyguan/enas), macro and micro search space on CIFAR10 included. Since code to train from scratch on NNI is not ready yet, reproduction results are currently unavailable.
