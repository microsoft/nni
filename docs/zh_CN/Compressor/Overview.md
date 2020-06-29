# 使用 NNI 进行模型压缩

```eval_rst
.. contents::
```

随着更多层和节点大型神经网络的使用，降低其存储和计算成本变得至关重要，尤其是对于某些实时应用程序。 模型压缩可用于解决此问题。

NNI provides a model compression toolkit to help user compress and speed up their model with state-of-the-art compression algorithms and strategies. There are several core features supported by NNI model compression:

* Support many popular pruning and quantization algorithms.
* Automate model pruning and quantization process with state-of-the-art strategies and NNI's auto tuning power.
* Speed up a compressed model to make it have lower inference latency and also make it become smaller.
* Provide friendly and easy-to-use compression utilities for users to dive into the compression process and results.
* Concise interface for users to customize their own compression algorithms.

*Note that the interface and APIs are unified for both PyTorch and TensorFlow, currently only PyTorch version has been supported, TensorFlow version will be supported in future.*


## Supported Algorithms

The algorithms include pruning algorithms and quantization algorithms.

### Pruning Algorithms

Pruning algorithms compress the original network by removing redundant weights or channels of layers, which can reduce model complexity and address the over-ﬁtting issue.

| 名称                                                                                                                           | 算法简介                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [Level Pruner](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#level-pruner)                                     | 根据权重的绝对值，来按比例修剪权重。                                                                                                                      |
| [AGP Pruner](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#agp-pruner)                                         | 自动的逐步剪枝（是否剪枝的判断：基于对模型剪枝的效果）[参考论文](https://arxiv.org/abs/1710.01878)                                                                     |
| [Lottery Ticket Pruner](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#lottery-ticket-hypothesis)               | "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" 提出的剪枝过程。 它会反复修剪模型。 [参考论文](https://arxiv.org/abs/1803.03635)  |
| [FPGM Pruner](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#fpgm-pruner)                                       | Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration [参考论文](https://arxiv.org/pdf/1811.00250.pdf)    |
| [L1Filter Pruner](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#l1filter-pruner)                               | 在卷积层中具有最小 L1 权重规范的剪枝过滤器（用于 Efficient Convnets 的剪枝过滤器） [参考论文](https://arxiv.org/abs/1608.08710)                                          |
| [L2Filter Pruner](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#l2filter-pruner)                               | 在卷积层中具有最小 L2 权重规范的剪枝过滤器                                                                                                                 |
| [ActivationAPoZRankFilterPruner](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#activationapozrankfilterpruner) | 基于指标 APoZ（平均百分比零）的剪枝过滤器，该指标测量（卷积）图层激活中零的百分比。 [参考论文](https://arxiv.org/abs/1607.03250)                                                   |
| [ActivationMeanRankFilterPruner](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#activationmeanrankfilterpruner) | 基于计算输出激活最小平均值指标的剪枝过滤器                                                                                                                   |
| [Slim Pruner](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#slim-pruner)                                       | 通过修剪 BN 层中的缩放因子来修剪卷积层中的通道 (Learning Efficient Convolutional Networks through Network Slimming) [参考论文](https://arxiv.org/abs/1708.06519) |
| [TaylorFO Pruner](https://nni.readthedocs.io/en/latest/Compressor/Pruner.html#taylorfoweightfilterpruner)                    | 基于一阶泰勒展开的权重 (Importance Estimation for Neural Network Pruning) [参考论文](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf) |


### Quantization Algorithms

Quantization algorithms compress the original network by reducing the number of bits required to represent weights or activations, which can reduce the computations and the inference time.

| 名称                                                                                                  | 算法简介                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Naive Quantizer](https://nni.readthedocs.io/en/latest/Compressor/Quantizer.html#naive-quantizer)   | 默认将权重量化为 8 位                                                                                                                                                               |
| [QAT Quantizer](https://nni.readthedocs.io/en/latest/Compressor/Quantizer.html#qat-quantizer)       | 为 Efficient Integer-Arithmetic-Only Inference 量化并训练神经网络。 [参考论文](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) |
| [DoReFa Quantizer](https://nni.readthedocs.io/en/latest/Compressor/Quantizer.html#dorefa-quantizer) | DoReFa-Net: 通过低位宽的梯度算法来训练低位宽的卷积神经网络。 [参考论文](https://arxiv.org/abs/1606.06160)                                                                                              |
| [BNN Quantizer](https://nni.readthedocs.io/en/latest/Compressor/Quantizer.html#bnn-quantizer)       | 二进制神经网络：使用权重和激活限制为 +1 或 -1 的深度神经网络。 [参考论文](https://arxiv.org/abs/1602.02830)                                                                                               |

## Automatic Model Compression

Given targeted compression ratio, it is pretty hard to obtain the best compressed ratio in a one shot manner. An automatic model compression algorithm usually need to explore the compression space by compressing different layers with different sparsities. NNI provides such algorithms to free users from specifying sparsity of each layer in a model. Moreover, users could leverage NNI's auto tuning power to automatically compress a model. Detailed document can be found [here](./AutoCompression.md).

## Model Speedup

The final goal of model compression is to reduce inference latency and model size. However, existing model compression algorithms mainly use simulation to check the performance (e.g., accuracy) of compressed model, for example, using masks for pruning algorithms, and storing quantized values still in float32 for quantization algorithms. Given the output masks and quantization bits produced by those algorithms, NNI can really speed up the model. The detailed tutorial of Model Speedup can be found [here](./ModelSpeedup.md).

## Compression Utilities

Compression utilities include some useful tools for users to understand and analyze the model they want to compress. For example, users could check sensitivity of each layer to pruning. Users could easily calculate the FLOPs and parameter size of a model. Please refer to [here](./CompressionUtils.md) for a complete list of compression utilities.

## Customize Your Own Compression Algorithms

NNI model compression leaves simple interface for users to customize a new compression algorithm. The design philosophy of the interface is making users focus on the compression logic while hiding framework specific implementation details from users. The detailed tutorial for customizing a new compression algorithm (pruning algorithm or quantization algorithm) can be found [here](./Framework.md).

## Reference and Feedback
* To [report a bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md) for this feature in GitHub;
* To [file a feature or improvement request](https://github.com/microsoft/nni/issues/new?template=enhancement.md) for this feature in GitHub;
* To know more about [Feature Engineering with NNI](../FeatureEngineering/Overview.md);
* To know more about [NAS with NNI](../NAS/Overview.md);
* To know more about [Hyperparameter Tuning with NNI](../Tuner/BuiltinTuner.md);
