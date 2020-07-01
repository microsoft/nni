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

| 名称                                                                                                                           | 算法简介                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [Level Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#level-pruner)                                     | 根据权重的绝对值，来按比例修剪权重。                                                                                                                            |
| [AGP Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#agp-pruner)                                         | 自动的逐步剪枝（是否剪枝的判断：基于对模型剪枝的效果）[参考论文](https://arxiv.org/abs/1710.01878)                                                                           |
| [Lottery Ticket Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#lottery-ticket-hypothesis)               | "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" 提出的剪枝过程。 它会反复修剪模型。 [参考论文](https://arxiv.org/abs/1803.03635)        |
| [FPGM Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#fpgm-pruner)                                       | Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration [参考论文](https://arxiv.org/pdf/1811.00250.pdf)          |
| [L1Filter Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#l1filter-pruner)                               | 在卷积层中具有最小 L1 权重规范的剪枝滤波器（用于 Efficient Convnets 的剪枝滤波器） [参考论文](https://arxiv.org/abs/1608.08710)                                                |
| [L2Filter Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#l2filter-pruner)                               | 在卷积层中具有最小 L2 权重规范的剪枝滤波器                                                                                                                       |
| [ActivationAPoZRankFilterPruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#activationapozrankfilterpruner) | 基于指标 APoZ（平均百分比零）的剪枝滤波器，该指标测量（卷积）图层激活中零的百分比。 [参考论文](https://arxiv.org/abs/1607.03250)                                                         |
| [ActivationMeanRankFilterPruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#activationmeanrankfilterpruner) | 基于计算输出激活最小平均值指标的剪枝滤波器                                                                                                                         |
| [Slim Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#slim-pruner)                                       | 通过修剪 BN 层中的缩放因子来修剪卷积层中的通道 (Learning Efficient Convolutional Networks through Network Slimming) [参考论文](https://arxiv.org/abs/1708.06519)       |
| [TaylorFO Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#taylorfoweightfilterpruner)                    | 基于一阶泰勒展开的权重对滤波器剪枝 (Importance Estimation for Neural Network Pruning) [参考论文](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf) |
| [ADMM Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#admm-pruner)                                       | 基于 ADMM 优化技术的剪枝 [参考论文](https://arxiv.org/abs/1804.03294)                                                                                      |
| [NetAdapt Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#netadapt-pruner)                               | 在满足计算资源预算的情况下，对预训练的网络迭代剪枝 [参考论文](https://arxiv.org/abs/1804.03230)                                                                            |
| [SimulatedAnnealing Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#simulatedannealing-pruner)           | 通过启发式的模拟退火算法进行自动剪枝 [参考论文](https://arxiv.org/abs/1907.03141)                                                                                   |
| [AutoCompress Pruner](https://nni.readthedocs.io/zh/latest/Compressor/Pruner.html#autocompress-pruner)                       | 通过迭代调用 SimulatedAnnealing Pruner 和 ADMM Pruner 进行自动剪枝 [参考论文](https://arxiv.org/abs/1907.03141)                                                |


### 量化算法

量化算法通过减少表示权重或激活所需的精度位数来压缩原始网络，这可以减少计算和推理时间。

| 名称                                                                                                  | 算法简介                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Naive Quantizer](https://nni.readthedocs.io/zh/latest/Compressor/Quantizer.html#naive-quantizer)   | 默认将权重量化为 8 位                                                                                                                                                               |
| [QAT Quantizer](https://nni.readthedocs.io/zh/latest/Compressor/Quantizer.html#qat-quantizer)       | 为 Efficient Integer-Arithmetic-Only Inference 量化并训练神经网络。 [参考论文](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) |
| [DoReFa Quantizer](https://nni.readthedocs.io/zh/latest/Compressor/Quantizer.html#dorefa-quantizer) | DoReFa-Net: 通过低位宽的梯度算法来训练低位宽的卷积神经网络。 [参考论文](https://arxiv.org/abs/1606.06160)                                                                                              |
| [BNN Quantizer](https://nni.readthedocs.io/zh/latest/Compressor/Quantizer.html#bnn-quantizer)       | 二进制神经网络：使用权重和激活限制为 +1 或 -1 的深度神经网络。 [参考论文](https://arxiv.org/abs/1602.02830)                                                                                               |

## 自动模型压缩

有时，给定的目标压缩率很难通过一次压缩就得到最好的结果。 自动模型压缩算法，通常需要通过对不同层采用不同的稀疏度来探索可压缩的空间。 NNI 提供了这样的算法，来帮助用户在模型中为每一层指定压缩度。 此外，还可利用 NNI 的自动调参功能来自动的压缩模型。 详细文档参考[这里](./AutoCompression.md)。

## 模型加速

模型压缩的目的是减少推理延迟和模型大小。 但现有的模型压缩算法主要通过模拟的方法来检查压缩模型性能（如精度）。例如，剪枝算法中使用掩码，而量化算法中量化值仍然是以 32 位浮点数来存储。 只要给出这些算法产生的掩码和量化位，NNI 可真正的加速模型。 模型加速的详细文档参考[这里](./ModelSpeedup.md)。

## 压缩工具

压缩工具包括了一些有用的工具，能帮助用户理解并分析要压缩的模型。 例如，可检查每层对剪枝的敏感度。 可很容易的计算模型的 FLOPs 和参数数量。 [点击这里](./CompressionUtils.md)，查看压缩工具的完整列表。

## 自定义压缩算法

NNI 模型压缩提供了简洁的接口，用于自定义新的压缩算法。 接口的设计理念是，将框架相关的实现细节包装起来，让用户能聚焦于压缩逻辑。 点击[这里](./Framework.md)，查看自定义新压缩算法（包括剪枝和量化算法）的详细教程。

## 参考和反馈
* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)；
* 了解更多关于 [NNI 中的特征工程](../FeatureEngineering/Overview.md)；
* 了解更多关于 [NNI 中的 NAS](../NAS/Overview.md)；
* 了解更多关于 [NNI 中的超参调优](../Tuner/BuiltinTuner.md)；
