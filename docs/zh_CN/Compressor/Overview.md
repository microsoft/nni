# 使用 NNI 进行模型压缩

```eval_rst
.. contents::
```

随着更多层和节点大型神经网络的使用，降低其存储和计算成本变得至关重要，尤其是对于某些实时应用程序。 模型压缩可用于解决此问题。

NNI 的模型压缩工具包，提供了最先进的模型压缩算法和策略，帮助压缩并加速模型。 NNI 模型压缩支持的主要功能有：

* 支持多种流行的剪枝和量化算法。
* 通过 NNI 强大的自动调优功能，可使用最先进的策略来自动化模型的剪枝和量化过程。
* 加速压缩的模型，使其在推理时有更低的延迟，同时文件也会变小。
* 提供优化且易用的压缩工具，帮助用户深入了解压缩过程和结果。
* 提供简洁的接口，帮助用户实现自己的压缩算法。

*注意，PyTorch 和 TensorFlow 有统一的 API 接口，当前仅支持 PyTorch 版本，未来会提供 TensorFlow 的支持。*


## 支持的算法

包括剪枝和量化算法。

### 剪枝算法

剪枝算法通过删除冗余权重或层通道来压缩原始网络，从而降低模型复杂性并解决过拟合问题。

| 名称                                                                                                                           | 算法简介                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [Level Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#level-pruner)                                     | 根据权重的绝对值，来按比例修剪权重。                                                                                                                      |
| [AGP Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#agp-pruner)                                         | 自动的逐步剪枝（是否剪枝的判断：基于对模型剪枝的效果）[参考论文](https://arxiv.org/abs/1710.01878)                                                                     |
| [Lottery Ticket Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#lottery-ticket-hypothesis)               | "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" 提出的剪枝过程。 它会反复修剪模型。 [参考论文](https://arxiv.org/abs/1803.03635)  |
| [FPGM Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#fpgm-pruner)                                       | Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration [参考论文](https://arxiv.org/pdf/1811.00250.pdf)    |
| [L1Filter Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#l1filter-pruner)                               | 在卷积层中具有最小 L1 权重规范的剪枝过滤器（用于 Efficient Convnets 的剪枝过滤器） [参考论文](https://arxiv.org/abs/1608.08710)                                          |
| [L2Filter Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#l2filter-pruner)                               | 在卷积层中具有最小 L2 权重规范的剪枝过滤器                                                                                                                 |
| [ActivationAPoZRankFilterPruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#activationapozrankfilterpruner) | 基于指标 APoZ（平均百分比零）的剪枝过滤器，该指标测量（卷积）图层激活中零的百分比。 [参考论文](https://arxiv.org/abs/1607.03250)                                                   |
| [ActivationMeanRankFilterPruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#activationmeanrankfilterpruner) | 基于计算输出激活最小平均值指标的剪枝过滤器                                                                                                                   |
| [Slim Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#slim-pruner)                                       | 通过修剪 BN 层中的缩放因子来修剪卷积层中的通道 (Learning Efficient Convolutional Networks through Network Slimming) [参考论文](https://arxiv.org/abs/1708.06519) |
| [TaylorFO Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#taylorfoweightfilterpruner)                    | 基于一阶泰勒展开的权重 (Importance Estimation for Neural Network Pruning) [参考论文](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf) |


### 量化算法

量化算法通过减少表示权重或激活所需的精度位数来压缩原始网络，这可以减少计算和推理时间。

| 名称                                                                                                  | 算法简介                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Naive Quantizer](https://nni.readthedocs.io/zh/latest/Compressor/Quantizer.html#naive-quantizer)   | 默认将权重量化为 8 位                                                                                                                                                               |
| [QAT Quantizer](https://nni.readthedocs.io/zh/latest/Compressor/Quantizer.html#qat-quantizer)       | 为 Efficient Integer-Arithmetic-Only Inference 量化并训练神经网络。 [参考论文](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) |
| [DoReFa Quantizer](https://nni.readthedocs.io/zh/latest/Compressor/Quantizer.html#dorefa-quantizer) | DoReFa-Net: 通过低位宽的梯度算法来训练低位宽的卷积神经网络。 [参考论文](https://arxiv.org/abs/1606.06160)                                                                                              |
| [BNN Quantizer](https://nni.readthedocs.io/zh/latest/Compressor/Quantizer.html#bnn-quantizer)       | 二进制神经网络：使用权重和激活限制为 +1 或 -1 的深度神经网络。 [参考论文](https://arxiv.org/abs/1602.02830)                                                                                               |

## 自动模型压缩

Given targeted compression ratio, it is pretty hard to obtain the best compressed ratio in a one shot manner. An automatic model compression algorithm usually need to explore the compression space by compressing different layers with different sparsities. NNI provides such algorithms to free users from specifying sparsity of each layer in a model. Moreover, users could leverage NNI's auto tuning power to automatically compress a model. Detailed document can be found [here](./AutoCompression.md).

## 模型加速

The final goal of model compression is to reduce inference latency and model size. However, existing model compression algorithms mainly use simulation to check the performance (e.g., accuracy) of compressed model, for example, using masks for pruning algorithms, and storing quantized values still in float32 for quantization algorithms. Given the output masks and quantization bits produced by those algorithms, NNI can really speed up the model. The detailed tutorial of Model Speedup can be found [here](./ModelSpeedup.md).

## 压缩工具

Compression utilities include some useful tools for users to understand and analyze the model they want to compress. For example, users could check sensitivity of each layer to pruning. Users could easily calculate the FLOPs and parameter size of a model. Please refer to [here](./CompressionUtils.md) for a complete list of compression utilities.

## Customize Your Own Compression Algorithms

NNI model compression leaves simple interface for users to customize a new compression algorithm. The design philosophy of the interface is making users focus on the compression logic while hiding framework specific implementation details from users. The detailed tutorial for customizing a new compression algorithm (pruning algorithm or quantization algorithm) can be found [here](./Framework.md).

## Reference and Feedback
* To [report a bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md) for this feature in GitHub;
* To [file a feature or improvement request](https://github.com/microsoft/nni/issues/new?template=enhancement.md) for this feature in GitHub;
* To know more about [Feature Engineering with NNI](../FeatureEngineering/Overview.md);
* To know more about [NAS with NNI](../NAS/Overview.md);
* To know more about [Hyperparameter Tuning with NNI](../Tuner/BuiltinTuner.md);
