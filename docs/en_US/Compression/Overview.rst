Model Compression with NNI
==========================

.. contents::

As larger neural networks with more layers and nodes are considered, reducing their storage and computational cost becomes critical, especially for some real-time applications. Model compression can be used to address this problem.

NNI provides a model compression toolkit to help user compress and speed up their model with state-of-the-art compression algorithms and strategies. There are several core features supported by NNI model compression:


* Support many popular pruning and quantization algorithms.
* Automate model pruning and quantization process with state-of-the-art strategies and NNI's auto tuning power.
* Speed up a compressed model to make it have lower inference latency and also make it become smaller.
* Provide friendly and easy-to-use compression utilities for users to dive into the compression process and results.
* Concise interface for users to customize their own compression algorithms.

*Note that the interface and APIs are unified for both PyTorch and TensorFlow, currently only PyTorch version has been supported, TensorFlow version will be supported in future.*

Supported Algorithms
--------------------

The algorithms include pruning algorithms and quantization algorithms.

Pruning Algorithms
^^^^^^^^^^^^^^^^^^

Pruning algorithms compress the original network by removing redundant weights or channels of layers, which can reduce model complexity and address the over-ﬁtting issue.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Introduction of Algorithm
   * - `Level Pruner </Compression/Pruner.html#level-pruner>`__
     - Pruning the specified ratio on each weight based on absolute values of weights
   * - `AGP Pruner </Compression/Pruner.html#agp-pruner>`__
     - Automated gradual pruning (To prune, or not to prune: exploring the efficacy of pruning for model compression) `Reference Paper <https://arxiv.org/abs/1710.01878>`__
   * - `Lottery Ticket Pruner </Compression/Pruner.html#lottery-ticket-hypothesis>`__
     - The pruning process used by "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks". It prunes a model iteratively. `Reference Paper <https://arxiv.org/abs/1803.03635>`__
   * - `FPGM Pruner </Compression/Pruner.html#fpgm-pruner>`__
     - Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration `Reference Paper <https://arxiv.org/pdf/1811.00250.pdf>`__
   * - `L1Filter Pruner </Compression/Pruner.html#l1filter-pruner>`__
     - Pruning filters with the smallest L1 norm of weights in convolution layers (Pruning Filters for Efficient Convnets) `Reference Paper <https://arxiv.org/abs/1608.08710>`__
   * - `L2Filter Pruner </Compression/Pruner.html#l2filter-pruner>`__
     - Pruning filters with the smallest L2 norm of weights in convolution layers
   * - `ActivationAPoZRankFilterPruner </Compression/Pruner.html#activationapozrankfilterpruner>`__
     - Pruning filters based on the metric APoZ (average percentage of zeros) which measures the percentage of zeros in activations of (convolutional) layers. `Reference Paper <https://arxiv.org/abs/1607.03250>`__
   * - `ActivationMeanRankFilterPruner </Compression/Pruner.html#activationmeanrankfilterpruner>`__
     - Pruning filters based on the metric that calculates the smallest mean value of output activations
   * - `Slim Pruner </Compression/Pruner.html#slim-pruner>`__
     - Pruning channels in convolution layers by pruning scaling factors in BN layers(Learning Efficient Convolutional Networks through Network Slimming) `Reference Paper <https://arxiv.org/abs/1708.06519>`__
   * - `TaylorFO Pruner </Compression/Pruner.html#taylorfoweightfilterpruner>`__
     - Pruning filters based on the first order taylor expansion on weights(Importance Estimation for Neural Network Pruning) `Reference Paper <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__
   * - `ADMM Pruner </Compression/Pruner.html#admm-pruner>`__
     - Pruning based on ADMM optimization technique `Reference Paper <https://arxiv.org/abs/1804.03294>`__
   * - `NetAdapt Pruner </Compression/Pruner.html#netadapt-pruner>`__
     - Automatically simplify a pretrained network to meet the resource budget by iterative pruning  `Reference Paper <https://arxiv.org/abs/1804.03230>`__
   * - `SimulatedAnnealing Pruner </Compression/Pruner.html#simulatedannealing-pruner>`__
     - Automatic pruning with a guided heuristic search method, Simulated Annealing algorithm `Reference Paper <https://arxiv.org/abs/1907.03141>`__
   * - `AutoCompress Pruner </Compression/Pruner.html#autocompress-pruner>`__
     - Automatic pruning by iteratively call SimulatedAnnealing Pruner and ADMM Pruner `Reference Paper <https://arxiv.org/abs/1907.03141>`__
   * - `AMC Pruner </Compression/Pruner.html#amc-pruner>`__
     - AMC: AutoML for Model Compression and Acceleration on Mobile Devices `Reference Paper <https://arxiv.org/pdf/1802.03494.pdf>`__


You can refer to this :githublink:`benchmark <docs/en_US/CommunitySharings/ModelCompressionComparison.rst>` for the performance of these pruners on some benchmark problems.

Quantization Algorithms
^^^^^^^^^^^^^^^^^^^^^^^

Quantization algorithms compress the original network by reducing the number of bits required to represent weights or activations, which can reduce the computations and the inference time.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Introduction of Algorithm
   * - `Naive Quantizer </Compression/Quantizer.html#naive-quantizer>`__
     - Quantize weights to default 8 bits
   * - `QAT Quantizer </Compression/Quantizer.html#qat-quantizer>`__
     - Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. `Reference Paper <http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf>`__
   * - `DoReFa Quantizer </Compression/Quantizer.html#dorefa-quantizer>`__
     - DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. `Reference Paper <https://arxiv.org/abs/1606.06160>`__
   * - `BNN Quantizer </Compression/Quantizer.html#bnn-quantizer>`__
     - Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. `Reference Paper <https://arxiv.org/abs/1602.02830>`__


Automatic Model Compression
---------------------------

Given targeted compression ratio, it is pretty hard to obtain the best compressed ratio in a one shot manner. An automatic model compression algorithm usually need to explore the compression space by compressing different layers with different sparsities. NNI provides such algorithms to free users from specifying sparsity of each layer in a model. Moreover, users could leverage NNI's auto tuning power to automatically compress a model. Detailed document can be found `here <./AutoPruningUsingTuners.rst>`__.

Model Speedup
-------------

The final goal of model compression is to reduce inference latency and model size. However, existing model compression algorithms mainly use simulation to check the performance (e.g., accuracy) of compressed model, for example, using masks for pruning algorithms, and storing quantized values still in float32 for quantization algorithms. Given the output masks and quantization bits produced by those algorithms, NNI can really speed up the model. The detailed tutorial of Model Speedup can be found `here <./ModelSpeedup.rst>`__.

Compression Utilities
---------------------

Compression utilities include some useful tools for users to understand and analyze the model they want to compress. For example, users could check sensitivity of each layer to pruning. Users could easily calculate the FLOPs and parameter size of a model. Please refer to `here <./CompressionUtils.rst>`__ for a complete list of compression utilities.

Customize Your Own Compression Algorithms
-----------------------------------------

NNI model compression leaves simple interface for users to customize a new compression algorithm. The design philosophy of the interface is making users focus on the compression logic while hiding framework specific implementation details from users. The detailed tutorial for customizing a new compression algorithm (pruning algorithm or quantization algorithm) can be found `here <./Framework.rst>`__.

Reference and Feedback
----------------------


* To `report a bug <https://github.com/microsoft/nni/issues/new?template=bug-report.rst>`__ for this feature in GitHub;
* To `file a feature or improvement request <https://github.com/microsoft/nni/issues/new?template=enhancement.rst>`__ for this feature in GitHub;
* To know more about `Feature Engineering with NNI <../FeatureEngineering/Overview.rst>`__\ ;
* To know more about `NAS with NNI <../NAS/Overview.rst>`__\ ;
* To know more about `Hyperparameter Tuning with NNI <../Tuner/BuiltinTuner.rst>`__\ ;
