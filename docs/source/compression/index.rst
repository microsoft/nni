Model Compression with NNI
==========================

.. toctree::
    :hidden:
    :maxdepth: 2

    Pruning <pruning>
    Quantization <quantization>
    Config Specification <compression_config_list>
    Advanced Usage <advanced_usage>

.. Using rubric to prevent the section heading to be include into toc

.. rubric:: Overview

Deep neural networks (DNNs) have achieved great success in many tasks like computer vision, nature launguage processing, speech processing.
However, typical neural networks are both computationally expensive and energy-intensive,
which can be difficult to be deployed on devices with low computation resources or with strict latency requirements.
Therefore, a natural thought is to perform model compression to reduce model size and accelerate model training/inference without losing performance significantly.
Model compression techniques can be divided into two categories: pruning and quantization.
The pruning methods explore the redundancy in the model weights and try to remove/prune the redundant and uncritical weights.
Quantization refers to compress models by reducing the number of bits required to represent weights or activations.
We further elaborate on the two methods, pruning and quantization, in the following chapters. Besides, the figure below visualizes the difference between these two methods.

.. image:: ../../img/prune_quant.jpg
   :target: ../../img/prune_quant.jpg
   :scale: 40%
   :alt:

NNI provides an easy-to-use toolkit to help users design and use model pruning and quantization algorithms.
For users to compress their models, they only need to add several lines in their code.
There are some popular model compression algorithms built-in in NNI.
Users could further use NNI’s auto-tuning power to find the best-compressed model, which is detailed in Auto Model Compression.
On the other hand, users could easily customize their new compression algorithms using NNI’s interface.

There are several core features supported by NNI model compression:

* Support many popular pruning and quantization algorithms.
* Automate model pruning and quantization process with state-of-the-art strategies and NNI's auto tuning power.
* Speed up a compressed model to make it have lower inference latency and also make it smaller.
* Provide friendly and easy-to-use compression utilities for users to dive into the compression process and results.
* Concise interface for users to customize their own compression algorithms.


.. rubric:: Compression Pipeline

.. image:: ../../img/compression_flow.jpg
   :target: ../../img/compression_flow.jpg
   :alt: 

The overall compression pipeline in NNI is shown above. For compressing a pretrained model, pruning and quantization can be used alone or in combination.
If users want to apply both, a sequential mode is recommended as common practise.

.. note::
  Note that NNI pruners or quantizers are not meant to physically compact the model but for simulating the compression effect. Whereas NNI speedup tool can truly compress model by changing the network architecture and therefore reduce latency.
  To obtain a truly compact model, users should conduct :doc:`pruning speedup <../tutorials/pruning_speed_up>` or :doc:`quantizaiton speedup <../tutorials/quantization_speed_up>`. 
  The interface and APIs are unified for both PyTorch and TensorFlow. Currently only PyTorch version has been supported, and TensorFlow version will be supported in future.


.. rubric:: Supported Pruning Algorithms

Pruning algorithms compress the original network by removing redundant weights or channels of layers, which can reduce model complexity and mitigate the over-fitting issue.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Introduction of Algorithm
   * - :ref:`level-pruner`
     - Pruning the specified ratio on each weight based on absolute values of weights
   * - :ref:`l1-norm-pruner`
     - Pruning output channels with the smallest L1 norm of weights (Pruning Filters for Efficient Convnets) `Reference Paper <https://arxiv.org/abs/1608.08710>`__
   * - :ref:`l2-norm-pruner`
     - Pruning output channels with the smallest L2 norm of weights
   * - :ref:`fpgm-pruner`
     - Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration `Reference Paper <https://arxiv.org/abs/1811.00250>`__
   * - :ref:`slim-pruner`
     - Pruning output channels by pruning scaling factors in BN layers(Learning Efficient Convolutional Networks through Network Slimming) `Reference Paper <https://arxiv.org/abs/1708.06519>`__
   * - :ref:`activation-apoz-rank-pruner`
     - Pruning output channels based on the metric APoZ (average percentage of zeros) which measures the percentage of zeros in activations of (convolutional) layers. `Reference Paper <https://arxiv.org/abs/1607.03250>`__
   * - :ref:`activation-mean-rank-pruner`
     - Pruning output channels based on the metric that calculates the smallest mean value of output activations
   * - :ref:`taylor-fo-weight-pruner`
     - Pruning filters based on the first order taylor expansion on weights(Importance Estimation for Neural Network Pruning) `Reference Paper <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__
   * - :ref:`admm-pruner`
     - Pruning based on ADMM optimization technique `Reference Paper <https://arxiv.org/abs/1804.03294>`__
   * - :ref:`linear-pruner`
     - Sparsity ratio increases linearly during each pruning rounds, in each round, using a basic pruner to prune the model.
   * - :ref:`agp-pruner`
     - Automated gradual pruning (To prune, or not to prune: exploring the efficacy of pruning for model compression) `Reference Paper <https://arxiv.org/abs/1710.01878>`__
   * - :ref:`lottery-ticket-pruner`
     - The pruning process used by "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks". It prunes a model iteratively. `Reference Paper <https://arxiv.org/abs/1803.03635>`__
   * - :ref:`simulated-annealing-pruner`
     - Automatic pruning with a guided heuristic search method, Simulated Annealing algorithm `Reference Paper <https://arxiv.org/abs/1907.03141>`__
   * - :ref:`auto-compress-pruner`
     - Automatic pruning by iteratively call SimulatedAnnealing Pruner and ADMM Pruner `Reference Paper <https://arxiv.org/abs/1907.03141>`__
   * - :ref:`amc-pruner`
     - AMC: AutoML for Model Compression and Acceleration on Mobile Devices `Reference Paper <https://arxiv.org/abs/1802.03494>`__
   * - :ref:`movement-pruner`
     - Movement Pruning: Adaptive Sparsity by Fine-Tuning `Reference Paper <https://arxiv.org/abs/2005.07683>`__


.. rubric:: Supported Quantization Algorithms

Quantization algorithms compress the original network by reducing the number of bits required to represent weights or activations, which can reduce the computations and the inference time.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Introduction of Algorithm
   * - :ref:`naive-quantizer`
     - Quantize weights to default 8 bits
   * - :ref:`qat-quantizer`
     - Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. `Reference Paper <http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf>`__
   * - :ref:`dorefa-quantizer`
     - DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. `Reference Paper <https://arxiv.org/abs/1606.06160>`__
   * - :ref:`bnn-quantizer`
     - Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. `Reference Paper <https://arxiv.org/abs/1602.02830>`__
   * - :ref:`lsq-quantizer`
     - Learned step size quantization. `Reference Paper <https://arxiv.org/pdf/1902.08153.pdf>`__
   * - :ref:`observer-quantizer`
     - Post training quantizaiton. Collect quantization information during calibration with observers.


.. rubric:: Model Speedup

The final goal of model compression is to reduce inference latency and model size.
However, existing model compression algorithms mainly use simulation to check the performance (e.g., accuracy) of compressed model.
For example, using masks for pruning algorithms, and storing quantized values still in float32 for quantization algorithms.
Given the output masks and quantization bits produced by those algorithms, NNI can really speed up the model.

The following figure shows how NNI prunes and speeds up your models. 

.. image:: ../../img/pipeline_compress.jpg
   :target: ../../img/pipeline_compress.jpg
   :scale: 40%
   :alt:

The detailed tutorial of Speed Up Model with Mask can be found :doc:`here <../tutorials/pruning_speed_up>`.
The detailed tutorial of Speed Up Model with Calibration Config can be found :doc:`here <../tutorials/quantization_speed_up>`.

.. attention::

  NNI's model pruning framework has been upgraded to a more powerful version (named pruning v2 before nni v2.6).
  The old version (`named pruning before nni v2.6 <https://nni.readthedocs.io/en/v2.6/Compression/pruning.html>`_) will be out of maintenance. If for some reason you have to use the old pruning,
  v2.6 is the last nni version to support old pruning version.
