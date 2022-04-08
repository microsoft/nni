Quantizer in NNI
================

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
