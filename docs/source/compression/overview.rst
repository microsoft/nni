Model Compression Overview
==========================

Deep neural networks (DNNs) have achieved great success in many tasks.
However, typical neural networks are both computationally expensive and energy-intensive,
can be difficult to be deployed on devices with low computation resources or with strict latency requirements.
Therefore, a natural thought is to perform model compression to reduce model size and accelerate model training/inference without losing performance significantly.
Model compression techniques can be divided into two categories: pruning and quantization.
The pruning methods explore the redundancy in the model weights and try to remove/prune the redundant and uncritical weights.
Quantization refers to compressing models by reducing the number of bits required to represent weights or activations.

NNI provides an easy-to-use toolkit to help users design and use model pruning and quantization algorithms.
For users to compress their models, they only need to add several lines in their code.
There are some popular model compression algorithms built-in in NNI.
Users could further use NNI’s auto-tuning power to find the best compressed model, which is detailed in Auto Model Compression.
On the other hand, users could easily customize their new compression algorithms using NNI’s interface.

There are several core features supported by NNI model compression:

* Support many popular pruning and quantization algorithms.
* Automate model pruning and quantization process with state-of-the-art strategies and NNI's auto tuning power.
* Speed up a compressed model to make it have lower inference latency and also make it become smaller.
* Provide friendly and easy-to-use compression utilities for users to dive into the compression process and results.
* Concise interface for users to customize their own compression algorithms.


Compression Pipeline
--------------------

.. image:: ../../img/compression_flow.jpg
   :target: ../../img/compression_flow.jpg
   :alt: 

The overall compression pipeline in NNI. For compressing a pretrained model, pruning and quantization can be used alone or in combination. 

.. note::
  Since NNI compression algorithms are not meant to compress model while NNI speedup tool can truly compress model and reduce latency.
  To obtain a truly compact model, users should conduct `model speedup <../tutorials/pruning_speed_up>`__.
  The interface and APIs are unified for both PyTorch and TensorFlow, currently only PyTorch version has been supported, TensorFlow version will be supported in future.


Supported Algorithms
--------------------

The algorithms include pruning algorithms and quantization algorithms.

Pruning Algorithms
^^^^^^^^^^^^^^^^^^

Pruning algorithms compress the original network by removing redundant weights or channels of layers, which can reduce model complexity and mitigate the over-fitting issue.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Introduction of Algorithm
   * - `Level Pruner <pruner.rst#level-pruner>`__
     - Pruning the specified ratio on each weight based on absolute values of weights
   * - `L1 Norm Pruner <pruner.rst#l1-norm-pruner>`__
     - Pruning output channels with the smallest L1 norm of weights (Pruning Filters for Efficient Convnets) `Reference Paper <https://arxiv.org/abs/1608.08710>`__
   * - `L2 Norm Pruner <pruner.rst#l2-norm-pruner>`__
     - Pruning output channels with the smallest L2 norm of weights
   * - `FPGM Pruner <pruner.rst#fpgm-pruner>`__
     - Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration `Reference Paper <https://arxiv.org/abs/1811.00250>`__
   * - `Slim Pruner <pruner.rst#slim-pruner>`__
     - Pruning output channels by pruning scaling factors in BN layers(Learning Efficient Convolutional Networks through Network Slimming) `Reference Paper <https://arxiv.org/abs/1708.06519>`__
   * - `Activation APoZ Rank Pruner <pruner.rst#activation-apoz-rank-pruner>`__
     - Pruning output channels based on the metric APoZ (average percentage of zeros) which measures the percentage of zeros in activations of (convolutional) layers. `Reference Paper <https://arxiv.org/abs/1607.03250>`__
   * - `Activation Mean Rank Pruner <pruner.rst#activation-mean-rank-pruner>`__
     - Pruning output channels based on the metric that calculates the smallest mean value of output activations
   * - `Taylor FO Weight Pruner <pruner.rst#taylor-fo-weight-pruner>`__
     - Pruning filters based on the first order taylor expansion on weights(Importance Estimation for Neural Network Pruning) `Reference Paper <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__
   * - `ADMM Pruner <pruner.rst#admm-pruner>`__
     - Pruning based on ADMM optimization technique `Reference Paper <https://arxiv.org/abs/1804.03294>`__
   * - `Linear Pruner <pruner.rst#linear-pruner>`__
     - Sparsity ratio increases linearly during each pruning rounds, in each round, using a basic pruner to prune the model.
   * - `AGP Pruner <pruner.rst#agp-pruner>`__
     - Automated gradual pruning (To prune, or not to prune: exploring the efficacy of pruning for model compression) `Reference Paper <https://arxiv.org/abs/1710.01878>`__
   * - `Lottery Ticket Pruner <pruner.rst#lottery-ticket-pruner>`__
     - The pruning process used by "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks". It prunes a model iteratively. `Reference Paper <https://arxiv.org/abs/1803.03635>`__
   * - `Simulated Annealing Pruner <pruner.rst#simulated-annealing-pruner>`__
     - Automatic pruning with a guided heuristic search method, Simulated Annealing algorithm `Reference Paper <https://arxiv.org/abs/1907.03141>`__
   * - `Auto Compress Pruner <pruner.rst#auto-compress-pruner>`__
     - Automatic pruning by iteratively call SimulatedAnnealing Pruner and ADMM Pruner `Reference Paper <https://arxiv.org/abs/1907.03141>`__
   * - `AMC Pruner <pruner.rst#amc-pruner>`__
     - AMC: AutoML for Model Compression and Acceleration on Mobile Devices `Reference Paper <https://arxiv.org/abs/1802.03494>`__
   * - `Movement Pruner <pruner.rst#movement-pruner>`__
     - Movement Pruning: Adaptive Sparsity by Fine-Tuning `Reference Paper <https://arxiv.org/abs/2005.07683>`__


Quantization Algorithms
^^^^^^^^^^^^^^^^^^^^^^^

Quantization algorithms compress the original network by reducing the number of bits required to represent weights or activations, which can reduce the computations and the inference time.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Introduction of Algorithm
   * - `Naive Quantizer <quantizer.rst#naive-quantizer>`__
     - Quantize weights to default 8 bits
   * - `QAT Quantizer <quantizer.rst#qat-quantizer>`__
     - Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. `Reference Paper <http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf>`__
   * - `DoReFa Quantizer <quantizer.rst#dorefa-quantizer>`__
     - DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. `Reference Paper <https://arxiv.org/abs/1606.06160>`__
   * - `BNN Quantizer <quantizer.rst#bnn-quantizer>`__
     - Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. `Reference Paper <https://arxiv.org/abs/1602.02830>`__
   * - `LSQ Quantizer <quantizer.rst#lsq-quantizer>`__
     - Learned step size quantization. `Reference Paper <https://arxiv.org/pdf/1902.08153.pdf>`__
   * - `Observer Quantizer <quantizer.rst#observer-quantizer>`__
     - Post training quantizaiton. Collect quantization information during calibration with observers.


Model Speedup
-------------

The final goal of model compression is to reduce inference latency and model size.
However, existing model compression algorithms mainly use simulation to check the performance (e.g., accuracy) of compressed model.
For example, using masks for pruning algorithms, and storing quantized values still in float32 for quantization algorithms.
Given the output masks and quantization bits produced by those algorithms, NNI can really speed up the model.
The detailed tutorial of Speed Up Model with Mask can be found `here <../tutorials/pruning_speed_up.rst>`__.
The detailed tutorial of Speed Up Model with Calibration Config can be found `here <../tutorials/quantization_speed_up.rst>`__.