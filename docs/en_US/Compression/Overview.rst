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

.. note::
  Since NNI compression algorithms are not meant to compress model while NNI speedup tool can truly compress model and reduce latency. To obtain a truly compact model, users should conduct `model speedup <./ModelSpeedup.rst>`__. The interface and APIs are unified for both PyTorch and TensorFlow, currently only PyTorch version has been supported, TensorFlow version will be supported in future.


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
   * - `Level Pruner <Pruner.rst#level-pruner>`__
     - Pruning the specified ratio on each weight based on absolute values of weights
   * - `AGP Pruner <../Compression/Pruner.rst#agp-pruner>`__
     - Automated gradual pruning (To prune, or not to prune: exploring the efficacy of pruning for model compression) `Reference Paper <https://arxiv.org/abs/1710.01878>`__
   * - `Lottery Ticket Pruner <../Compression/Pruner.rst#lottery-ticket-hypothesis>`__
     - The pruning process used by "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks". It prunes a model iteratively. `Reference Paper <https://arxiv.org/abs/1803.03635>`__
   * - `FPGM Pruner <../Compression/Pruner.rst#fpgm-pruner>`__
     - Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration `Reference Paper <https://arxiv.org/pdf/1811.00250.pdf>`__
   * - `L1Filter Pruner <../Compression/Pruner.rst#l1filter-pruner>`__
     - Pruning filters with the smallest L1 norm of weights in convolution layers (Pruning Filters for Efficient Convnets) `Reference Paper <https://arxiv.org/abs/1608.08710>`__
   * - `L2Filter Pruner <../Compression/Pruner.rst#l2filter-pruner>`__
     - Pruning filters with the smallest L2 norm of weights in convolution layers
   * - `ActivationAPoZRankFilterPruner <../Compression/Pruner.rst#activationapozrankfilter-pruner>`__
     - Pruning filters based on the metric APoZ (average percentage of zeros) which measures the percentage of zeros in activations of (convolutional) layers. `Reference Paper <https://arxiv.org/abs/1607.03250>`__
   * - `ActivationMeanRankFilterPruner <../Compression/Pruner.rst#activationmeanrankfilter-pruner>`__
     - Pruning filters based on the metric that calculates the smallest mean value of output activations
   * - `Slim Pruner <../Compression/Pruner.rst#slim-pruner>`__
     - Pruning channels in convolution layers by pruning scaling factors in BN layers(Learning Efficient Convolutional Networks through Network Slimming) `Reference Paper <https://arxiv.org/abs/1708.06519>`__
   * - `TaylorFO Pruner <../Compression/Pruner.rst#taylorfoweightfilter-pruner>`__
     - Pruning filters based on the first order taylor expansion on weights(Importance Estimation for Neural Network Pruning) `Reference Paper <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__
   * - `ADMM Pruner <../Compression/Pruner.rst#admm-pruner>`__
     - Pruning based on ADMM optimization technique `Reference Paper <https://arxiv.org/abs/1804.03294>`__
   * - `NetAdapt Pruner <../Compression/Pruner.rst#netadapt-pruner>`__
     - Automatically simplify a pretrained network to meet the resource budget by iterative pruning  `Reference Paper <https://arxiv.org/abs/1804.03230>`__
   * - `SimulatedAnnealing Pruner <../Compression/Pruner.rst#simulatedannealing-pruner>`__
     - Automatic pruning with a guided heuristic search method, Simulated Annealing algorithm `Reference Paper <https://arxiv.org/abs/1907.03141>`__
   * - `AutoCompress Pruner <../Compression/Pruner.rst#autocompress-pruner>`__
     - Automatic pruning by iteratively call SimulatedAnnealing Pruner and ADMM Pruner `Reference Paper <https://arxiv.org/abs/1907.03141>`__
   * - `AMC Pruner <../Compression/Pruner.rst#amc-pruner>`__
     - AMC: AutoML for Model Compression and Acceleration on Mobile Devices `Reference Paper <https://arxiv.org/pdf/1802.03494.pdf>`__


You can refer to this `benchmark <../CommunitySharings/ModelCompressionComparison.rst>`__ for the performance of these pruners on some benchmark problems.

Quantization Algorithms
^^^^^^^^^^^^^^^^^^^^^^^

Quantization algorithms compress the original network by reducing the number of bits required to represent weights or activations, which can reduce the computations and the inference time.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Introduction of Algorithm
   * - `Naive Quantizer <../Compression/Quantizer.rst#naive-quantizer>`__
     - Quantize weights to default 8 bits
   * - `QAT Quantizer <../Compression/Quantizer.rst#qat-quantizer>`__
     - Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. `Reference Paper <http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf>`__
   * - `DoReFa Quantizer <../Compression/Quantizer.rst#dorefa-quantizer>`__
     - DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. `Reference Paper <https://arxiv.org/abs/1606.06160>`__
   * - `BNN Quantizer <../Compression/Quantizer.rst#bnn-quantizer>`__
     - Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. `Reference Paper <https://arxiv.org/abs/1602.02830>`__


Model Speedup
-------------

The final goal of model compression is to reduce inference latency and model size. However, existing model compression algorithms mainly use simulation to check the performance (e.g., accuracy) of compressed model, for example, using masks for pruning algorithms, and storing quantized values still in float32 for quantization algorithms. Given the output masks and quantization bits produced by those algorithms, NNI can really speed up the model. The detailed tutorial of Masked Model Speedup can be found `here <./ModelSpeedup.rst>`__, The detailed tutorial of Mixed Precision Quantization Model Speedup can be found `here <./QuantizationSpeedup.rst>`__.

Compression Utilities
---------------------

Compression utilities include some useful tools for users to understand and analyze the model they want to compress. For example, users could check sensitivity of each layer to pruning. Users could easily calculate the FLOPs and parameter size of a model. Please refer to `here <./CompressionUtils.rst>`__ for a complete list of compression utilities.

Advanced Usage
--------------

NNI model compression leaves simple interface for users to customize a new compression algorithm. The design philosophy of the interface is making users focus on the compression logic while hiding framework specific implementation details from users. Users can learn more about our compression framework and customize a new compression algorithm (pruning algorithm or quantization algorithm) based on our framework. Moreover, users could leverage NNI's auto tuning power to automatically compress a model. Please refer to `here <./advanced.rst>`__ for more details.


Reference and Feedback
----------------------


* To `report a bug <https://github.com/microsoft/nni/issues/new?template=bug-report.rst>`__ for this feature in GitHub;
* To `file a feature or improvement request <https://github.com/microsoft/nni/issues/new?template=enhancement.rst>`__ for this feature in GitHub;
* To know more about `Feature Engineering with NNI <../FeatureEngineering/Overview.rst>`__\ ;
* To know more about `NAS with NNI <../NAS/Overview.rst>`__\ ;
* To know more about `Hyperparameter Tuning with NNI <../Tuner/BuiltinTuner.rst>`__\ ;
