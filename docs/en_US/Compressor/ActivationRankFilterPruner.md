ActivationRankFilterPruner on NNI Compressor
===

## 1. Introduction

ActivationRankFilterPruner is a series of pruners which prune filters according to some importance criterion calculated from the filters' output activations.

|             Pruner             |       Importance criterion        |                       Reference paper                        |
| :----------------------------: | :-------------------------------: | :----------------------------------------------------------: |
| ActivationAPoZRankFilterPruner | APoZ(average percentage of zeros) | [Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250) |
| ActivationMeanRankFilterPruner | mean value of output activations  | [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) |

## 2. Pruners

### ActivationAPoZRankFilterPruner

Hengyuan Hu, Rui Peng, Yu-Wing Tai and Chi-Keung Tang,

"[Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250)", ICLR 2016.

ActivationAPoZRankFilterPruner prunes the filters with the smallest APoZ(average percentage of zeros) of output activations.

The APoZ is defined as:

![](../../img/apoz.png)

### ActivationMeanRankFilterPruner

Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila and Jan Kautz,

"[Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)", ICLR 2017.

ActivationMeanRankFilterPruner prunes the filters with the smallest mean value of output activations

## 3. Usage

PyTorch code

```
from nni.compression.torch import ActivationAPoZRankFilterPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'], 'op_names': ['conv1', 'conv2'] }]
pruner = ActivationAPoZRankFilterPruner(model, config_list,statistics_batch_num=1)
pruner.compress()
```

#### User configuration for ActivationAPoZRankFilterPruner

- **sparsity:** This is to specify the sparsity operations to be compressed to
- **op_types:** Only Conv2d is supported in ActivationAPoZRankFilterPruner

## 4. Experiment

TODO. 





