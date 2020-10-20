# Model Compression with NNI

```eval_rst
.. contents::
```

As larger neural networks with more layers and nodes are considered, reducing their storage and computational cost becomes critical, especially for some real-time applications. Model compression can be used to address this problem.

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

| Name                                                                                                                          | Brief Introduction of Algorithm                                                                                                                                                                             |
| ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Level Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#level-pruner)                                     | Pruning the specified ratio on each weight based on absolute values of weights                                                                                                                              |
| [AGP Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#agp-pruner)                                         | Automated gradual pruning (To prune, or not to prune: exploring the efficacy of pruning for model compression) [Reference Paper](https://arxiv.org/abs/1710.01878)                                          |
| [Lottery Ticket Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#lottery-ticket-hypothesis)               | The pruning process used by "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks". It prunes a model iteratively. [Reference Paper](https://arxiv.org/abs/1803.03635)                  |
| [FPGM Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#fpgm-pruner)                                       | Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration [Reference Paper](https://arxiv.org/pdf/1811.00250.pdf)                                                             |
| [L1Filter Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#l1filter-pruner)                               | Pruning filters with the smallest L1 norm of weights in convolution layers (Pruning Filters for Efficient Convnets) [Reference Paper](https://arxiv.org/abs/1608.08710)                                     |
| [L2Filter Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#l2filter-pruner)                               | Pruning filters with the smallest L2 norm of weights in convolution layers                                                                                                                                  |
| [ActivationAPoZRankFilterPruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#activationapozrankfilterpruner) | Pruning filters based on the metric APoZ (average percentage of zeros) which measures the percentage of zeros in activations of (convolutional) layers. [Reference Paper](https://arxiv.org/abs/1607.03250) |
| [ActivationMeanRankFilterPruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#activationmeanrankfilterpruner) | Pruning filters based on the metric that calculates the smallest mean value of output activations                                                                                                           |
| [Slim Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#slim-pruner)                                       | Pruning channels in convolution layers by pruning scaling factors in BN layers(Learning Efficient Convolutional Networks through Network Slimming) [Reference Paper](https://arxiv.org/abs/1708.06519)      |
| [TaylorFO Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#taylorfoweightfilterpruner)                    | Pruning filters based on the first order taylor expansion on weights(Importance Estimation for Neural Network Pruning) [Reference Paper](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf)  |
| [ADMM Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#admm-pruner)                                       | Pruning based on ADMM optimization technique [Reference Paper](https://arxiv.org/abs/1804.03294)                                                                                                            |
| [NetAdapt Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#netadapt-pruner)                               | Automatically simplify a pretrained network to meet the resource budget by iterative pruning  [Reference Paper](https://arxiv.org/abs/1804.03230)                                                           |
| [SimulatedAnnealing Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#simulatedannealing-pruner)           | Automatic pruning with a guided heuristic search method, Simulated Annealing algorithm [Reference Paper](https://arxiv.org/abs/1907.03141)                                                                  |
| [AutoCompress Pruner](https://nni.readthedocs.io/en/latest/Compression/Pruner.html#autocompress-pruner)                       | Automatic pruning by iteratively call SimulatedAnnealing Pruner and ADMM Pruner [Reference Paper](https://arxiv.org/abs/1907.03141)                                                                         |

You can refer to this [benchmark](https://github.com/microsoft/nni/tree/master/docs/en_US/CommunitySharings/ModelCompressionComparison.md) for the performance of these pruners on some benchmark problems.

### Quantization Algorithms

Quantization algorithms compress the original network by reducing the number of bits required to represent weights or activations, which can reduce the computations and the inference time.

| Name                                                                                                 | Brief Introduction of Algorithm                                                                                                                                                                                            |
| ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Naive Quantizer](https://nni.readthedocs.io/en/latest/Compression/Quantizer.html#naive-quantizer)   | Quantize weights to default 8 bits                                                                                                                                                                                         |
| [QAT Quantizer](https://nni.readthedocs.io/en/latest/Compression/Quantizer.html#qat-quantizer)       | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. [Reference Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) |
| [DoReFa Quantizer](https://nni.readthedocs.io/en/latest/Compression/Quantizer.html#dorefa-quantizer) | DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. [Reference Paper](https://arxiv.org/abs/1606.06160)                                                                           |
| [BNN Quantizer](https://nni.readthedocs.io/en/latest/Compression/Quantizer.html#bnn-quantizer)       | Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. [Reference Paper](https://arxiv.org/abs/1602.02830)                                                         |

## Automatic Model Compression

Given targeted compression ratio, it is pretty hard to obtain the best compressed ratio in a one shot manner. An automatic model compression algorithm usually need to explore the compression space by compressing different layers with different sparsities. NNI provides such algorithms to free users from specifying sparsity of each layer in a model. Moreover, users could leverage NNI's auto tuning power to automatically compress a model. Detailed document can be found [here](./AutoPruningUsingTuners.md).

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
