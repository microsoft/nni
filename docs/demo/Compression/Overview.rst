Model Compression Overview
==========================

Deep neural networks (DNNs) have achieved great success in many tasks.
However, typical neural networks are both computationally expensive and energy intensive,
can be difficult to be deployed on devices with low computation resources or with strict latency requirements.
Therefore, a natural thought is to perform model compression to reduce model size and accelerate model training / inference without losing performance significantly.
Model compression techniques can be divided into two categories: pruning and quantization.
The pruning methods explore the redundancy in the model weights and try to remove / prune the redundant and uncritical weights.
Quantization refers to compressing models by reducing the number of bits required to represent weights or activations.

NNI provides an easy-to-use toolkit to help user design and use model pruning and quantization algorithms.
For users to compress their models, they only need to add several lines in their code.
There are some popular model compression algorithms built-in in NNI.
Users could further use NNI’s auto tuning power to find the best compressed model, which is detailed in Auto Model Compression.
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
  To obtain a truly compact model, users should conduct `model speedup <./PruningSpeedUp.rst>`__.
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
   * - `Level Pruner <Pruner.rst#level-pruner>`__
     - Pruning the specified ratio on each weight based on absolute values of weights


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


Model Speedup
-------------


The final goal of model compression is to reduce inference latency and model size.
However, existing model compression algorithms mainly use simulation to check the performance (e.g., accuracy) of compressed model.
For example, using masks for pruning algorithms, and storing quantized values still in float32 for quantization algorithms.
Given the output masks and quantization bits produced by those algorithms, NNI can really speed up the model.
The detailed tutorial of Masked Model Speedup can be found `here <./ModelSpeedup.rst>`__.
The detailed tutorial of Mixed Precision Quantization Model Speedup can be found `here <./QuantizationSpeedup.rst>`__.
