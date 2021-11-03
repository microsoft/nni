Supported Quantization Algorithms on NNI
========================================

Index of supported quantization algorithms


* `Naive Quantizer <#naive-quantizer>`__
* `QAT Quantizer <#qat-quantizer>`__
* `DoReFa Quantizer <#dorefa-quantizer>`__
* `BNN Quantizer <#bnn-quantizer>`__
* `LSQ Quantizer <#lsq-quantizer>`__
* `Observer Quantizer <#observer-quantizer>`__

Naive Quantizer
---------------

We provide Naive Quantizer to quantizer weight to default 8 bits, you can use it to test quantize algorithm without any configure.

Usage
^^^^^

pytorch

.. code-block:: python

   model = nni.algorithms.compression.pytorch.quantization.NaiveQuantizer(model).compress()

----

QAT Quantizer
-------------

In `Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference <http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf>`__\ , authors Benoit Jacob and Skirmantas Kligys provide an algorithm to quantize the model with training.

..

   We propose an approach that simulates quantization effects in the forward pass of training. Backpropagation still happens as usual, and all weights and biases are stored in floating point so that they can be easily nudged by small amounts. The forward propagation pass however simulates quantized inference as it will happen in the inference engine, by implementing in floating-point arithmetic the rounding behavior of the quantization scheme


   * Weights are quantized before they are convolved with the input. If batch normalization (see [17]) is used for the layer, the batch normalization parameters are “folded into” the weights before quantization.
   * Activations are quantized at points where they would be during inference, e.g. after the activation function is applied to a convolutional or fully connected layer’s output, or after a bypass connection adds or concatenates the outputs of several layers together such as in ResNets.


Usage
^^^^^

You can quantize your model to 8 bits with the code below before your training code.

PyTorch code

.. code-block:: python

   from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
   model = Mnist()

   config_list = [{
       'quant_types': ['weight'],
       'quant_bits': {
           'weight': 8,
       }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
       'op_types':['Conv2d', 'Linear']
   }, {
       'quant_types': ['output'],
       'quant_bits': 8,
       'quant_start_step': 7000,
       'op_types':['ReLU6']
   }]
   quantizer = QAT_Quantizer(model, config_list)
   quantizer.compress()

You can view example for more information

User configuration for QAT Quantizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

common configuration needed by compression algorithms can be found at `Specification of `config_list <./QuickStart.rst>`__.

configuration needed by this algorithm :


* **quant_start_step:** int

disable quantization until model are run by certain number of steps, this allows the network to enter a more stable
state where activation quantization ranges do not exclude a signiﬁcant fraction of values, default value is 0

Batch normalization folding
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Batch normalization folding is supported in QAT quantizer. It can be easily enabled by passing an argument `dummy_input` to
the quantizer, like:

.. code-block:: python

    # assume your model takes an input of shape (1, 1, 28, 28)
    # and dummy_input must be on the same device as the model
    dummy_input = torch.randn(1, 1, 28, 28)

    # pass the dummy_input to the quantizer
    quantizer = QAT_Quantizer(model, config_list, dummy_input=dummy_input)


The quantizer will automatically detect Conv-BN patterns and simulate batch normalization folding process in the training
graph. Note that when the quantization aware training process is finished, the folded weight/bias would be restored after calling
`quantizer.export_model`.

Quantization dtype and scheme customization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Different backends on different devices use different quantization strategies (i.e. dtype (int or uint) and
scheme (per-tensor or per-channel and symmetric or affine)). QAT quantizer supports customization of mainstream dtypes and schemes.
There are two ways to set them. One way is setting them globally through a function named `set_quant_scheme_dtype` like:

.. code-block:: python

    from nni.compression.pytorch.quantization.settings import set_quant_scheme_dtype

    # This will set all the quantization of 'input' in 'per_tensor_affine' and 'uint' manner
    set_quant_scheme_dtype('input', 'per_tensor_affine', 'uint)
    # This will set all the quantization of 'output' in 'per_tensor_symmetric' and 'int' manner
    set_quant_scheme_dtype('output', 'per_tensor_symmetric', 'int')
    # This will set all the quantization of 'weight' in 'per_channel_symmetric' and 'int' manner
    set_quant_scheme_dtype('weight', 'per_channel_symmetric', 'int')


The other way is more detailed. You can customize the dtype and scheme in each quantization config list like:

.. code-block:: python

    config_list = [{
       'quant_types': ['weight'],
       'quant_bits':  8,
       'op_types':['Conv2d', 'Linear'],
       'quant_dtype': 'int',
       'quant_scheme': 'per_channel_symmetric'
   }, {
       'quant_types': ['output'],
       'quant_bits': 8,
       'quant_start_step': 7000,
       'op_types':['ReLU6'],
       'quant_dtype': 'uint',
       'quant_scheme': 'per_tensor_affine'
   }]

Multi-GPU training
^^^^^^^^^^^^^^^^^^^
QAT quantizer natively supports multi-gpu training (DataParallel and DistributedDataParallel). Note that the quantizer
instantiation should happen before you wrap your model with DataParallel or DistributedDataParallel. For example:

.. code-block:: python

    from torch.nn.parallel import DistributedDataParallel as DDP
    from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer

    model = define_your_model()

    model = QAT_Quantizer(model, **other_params)  # <--- QAT_Quantizer instantiation

    model = DDP(model)

    for i in range(epochs):
        train(model)
        eval(model)


----

LSQ Quantizer
-------------

In `LEARNED STEP SIZE QUANTIZATION <https://arxiv.org/pdf/1902.08153.pdf>`__\ , authors Steven K. Esser and Jeffrey L. McKinstry provide an algorithm to train the scales with gradients.

..

   The authors introduce a novel means to estimate and scale the task loss gradient at each weight and activation layer’s quantizer step size, such that it can be learned in conjunction with other network parameters.


Usage
^^^^^
You can add codes below before your training codes. Three things must be done:


1. configure which layer to be quantized and which tensor (input/output/weight) of that layer to be quantized.
2. construct the lsq quantizer
3. call the `compress` API


PyTorch code

.. code-block:: python

    from nni.algorithms.compression.pytorch.quantization import LsqQuantizer
    model = Mnist()

    configure_list = [{
            'quant_types': ['weight', 'input'],
            'quant_bits': {
                'weight': 8,
                'input': 8,
            },
            'op_names': ['conv1']
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output': 8,},
            'op_names': ['relu1']
    }]

    quantizer = LsqQuantizer(model, configure_list, optimizer)
    quantizer.compress()

You can view example for more information. :githublink:`examples/model_compress/quantization/LSQ_torch_quantizer.py <examples/model_compress/quantization/LSQ_torch_quantizer.py>`

User configuration for LSQ Quantizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

common configuration needed by compression algorithms can be found at `Specification of `config_list <./QuickStart.rst>`__.

configuration needed by this algorithm :


----

DoReFa Quantizer
----------------

In `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`__\ , authors Shuchang Zhou and Yuxin Wu provide an algorithm named DoReFa to quantize the weight, activation and gradients with training.

Usage
^^^^^

To implement DoReFa Quantizer, you can add code below before your training code

PyTorch code

.. code-block:: python

   from nni.algorithms.compression.pytorch.quantization import DoReFaQuantizer
   config_list = [{ 
       'quant_types': ['weight'],
       'quant_bits': 8, 
       'op_types': ['default'] 
   }]
   quantizer = DoReFaQuantizer(model, config_list)
   quantizer.compress()

You can view example for more information

User configuration for DoReFa Quantizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

common configuration needed by compression algorithms can be found at `Specification of ``config_list`` <./QuickStart.rst>`__.

configuration needed by this algorithm :

----

BNN Quantizer
-------------

In `Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1 <https://arxiv.org/abs/1602.02830>`__\ , 

..

   We introduce a method to train Binarized Neural Networks (BNNs) - neural networks with binary weights and activations at run-time. At training-time the binary weights and activations are used for computing the parameters gradients. During the forward pass, BNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations, which is expected to substantially improve power-efficiency.


Usage
^^^^^

PyTorch code

.. code-block:: python

   from nni.algorithms.compression.pytorch.quantization import BNNQuantizer
   model = VGG_Cifar10(num_classes=10)

   configure_list = [{
       'quant_bits': 1,
       'quant_types': ['weight'],
       'op_types': ['Conv2d', 'Linear'],
       'op_names': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'classifier.0', 'classifier.3']
   }, {
       'quant_bits': 1,
       'quant_types': ['output'],
       'op_types': ['Hardtanh'],
       'op_names': ['features.6', 'features.9', 'features.13', 'features.16', 'features.20', 'classifier.2', 'classifier.5']
   }]

   quantizer = BNNQuantizer(model, configure_list)
   model = quantizer.compress()

You can view example :githublink:`examples/model_compress/quantization/BNN_quantizer_cifar10.py <examples/model_compress/quantization/BNN_quantizer_cifar10.py>` for more information.

User configuration for BNN Quantizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

common configuration needed by compression algorithms can be found at `Specification of ``config_list`` <./QuickStart.rst>`__.

configuration needed by this algorithm :

Experiment
^^^^^^^^^^

We implemented one of the experiments in `Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1 <https://arxiv.org/abs/1602.02830>`__\ , we quantized the **VGGNet** for CIFAR-10 in the paper. Our experiments results are as follows:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Model
     - Accuracy
   * - VGGNet
     - 86.93%


The experiments code can be found at :githublink:`examples/model_compress/quantization/BNN_quantizer_cifar10.py <examples/model_compress/quantization/BNN_quantizer_cifar10.py>` 


Observer Quantizer
------------------

..

   Observer quantizer is a framework of post-training quantization. It will insert observers into the place where the quantization will happen. During quantization calibration, each observer will record all the tensors it 'sees'. These tensors will be used to calculate the quantization statistics after calibration.

Usage
^^^^^

1. configure which layer to be quantized and which tensor (input/output/weight) of that layer to be quantized.
2. construct the observer quantizer.
3. do quantization calibration.
4. call the `compress` API to calculate the scale and zero point for each tensor and switch model to evaluation mode.

PyTorch code

.. code-block:: python

   from nni.algorithms.compression.pytorch.quantization import ObserverQuantizer

   def calibration(model, calib_loader):
       model.eval()
       with torch.no_grad():
           for data, _ in calib_loader:
               model(data)

   model = Mnist()

   configure_list = [{
       'quant_bits': 8,
       'quant_types': ['weight', 'input'],
       'op_names': ['conv1', 'conv2],
   }, {
       'quant_bits': 8,
       'quant_types': ['output'],
       'op_names': ['relu1', 'relu2],
   }]

   quantizer = ObserverQuantizer(model, configure_list)
   calibration(model, calib_loader)
   model = quantizer.compress()

You can view example :githublink:`examples/model_compress/quantization/observer_quantizer.py <examples/model_compress/quantization/observer_quantizer.py>` for more information.

User configuration for Observer Quantizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Common configuration needed by compression algorithms can be found at `Specification of `config_list <./QuickStart.rst>`__.


.. note::
    This quantizer is still under development for now. Some quantizer settings are hard-coded:

    - weight observer: per_tensor_symmetric, qint8
    - output observer: per_tensor_affine, quint8, reduce_range=True

    Other settings (such as quant_type and op_names) can be configured.

About the compress API
^^^^^^^^^^^^^^^^^^^^^^
Before the `compress` API is called, the model will only record tensors' statistics and no quantization process will be executed.
After the `compress` API is called, the model will NOT record tensors' statistics any more. The quantization scale and zero point will
be generated for each tensor and will be used to quantize each tensor during inference (we call it evaluation mode)

About calibration
^^^^^^^^^^^^^^^^^
Usually we pick up about 100 training/evaluation examples for calibration. If you found the accuracy is a bit low, try
to reduce the number of calibration examples.

