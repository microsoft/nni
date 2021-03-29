Tutorial
========

.. contents::

In this tutorial, we will explain more detailed usage about the model compression in NNI. 

Setup compression goal
----------------------

Specify the configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

Users can specify the configuration (i.e., ``config_list``\ ) for a compression algorithm. For example, when compressing a model, users may want to specify the sparsity ratio, to specify different ratios for different types of operations, to exclude certain types of operations, or to compress only a certain types of operations. For users to express these kinds of requirements, we define a configuration specification. It can be seen as a python ``list`` object, where each element is a ``dict`` object. 

The ``dict``\ s in the ``list`` are applied one by one, that is, the configurations in latter ``dict`` will overwrite the configurations in former ones on the operations that are within the scope of both of them. 

There are different keys in a ``dict``. Some of them are common keys supported by all the compression algorithms:

* **op_types**\ : This is to specify what types of operations to be compressed. 'default' means following the algorithm's default setting. All suported module types are defined in :githublink:`default_layers.py <nni/compression/pytorch/default_layers.py>` for pytorch.
* **op_names**\ : This is to specify by name what operations to be compressed. If this field is omitted, operations will not be filtered by it.
* **exclude**\ : Default is False. If this field is True, it means the operations with specified types and names will be excluded from the compression.

Some other keys are often specific to a certain algorithm, users can refer to `pruning algorithms <./Pruner.rst>`__ and `quantization algorithms <./Quantizer.rst>`__ for the keys allowed by each algorithm.

To prune all ``Conv2d`` layers with the sparsity of 0.6, the configuration can be write as:

.. code-block:: python

   [{
    'sparsity': 0.6,
    'op_types': ['Conv2d']
   }]

To control the sparsity of specific layers, the configuration can be writed as:

.. code-block:: python

   [{
      'sparsity': 0.8,
      'op_types': ['default']
   }, 
   {
      'sparsity': 0.6,
      'op_names': ['op_name1', 'op_name2']
   }, 
   {
      'exclude': True,
      'op_names': ['op_name3']
   }]

It means following the algorithm's default setting for compressed operations with sparsity 0.8, but for ``op_name1`` and ``op_name2`` use sparsity 0.6, and do not compress ``op_name3``.

Quantization specific keys
^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides the keys explained above, if you use quantization algorithms you need to specify more keys in ``config_list``\ , which are explained below.

* **quant_types** : list of string. 

Type of quantization you want to apply, currently support 'weight', 'input', 'output'. 'weight' means applying quantization operation
to the weight parameter of modules. 'input' means applying quantization operation to the input of module forward method. 'output' means applying quantization operation to the output of module forward method, which is often called as 'activation' in some papers.


* **quant_bits** : int or dict of {str : int}

bits length of quantization, key is the quantization type, value is the quantization bits length, eg. 

.. code-block:: bash

   {
      quant_bits: {
         'weight': 8,
         'output': 4,
         },
   }

when the value is int type, all quantization types share same bits length. eg. 

.. code-block:: bash

   {
      quant_bits: 8, # weight or output quantization are all 8 bits
   }

The following example shows a more complete ``config_list``\ , it uses ``op_names`` (or ``op_types``\ ) to specify the target layers along with the quantization bits for those layers.

.. code-block:: bash

   config_list = [{
      'quant_types': ['weight'],        
      'quant_bits': 8, 
      'op_names': ['conv1']
   }, 
   {
      'quant_types': ['weight'],
      'quant_bits': 4,
      'quant_start_step': 0,
      'op_names': ['conv2']
   }, 
   {
      'quant_types': ['weight'],
      'quant_bits': 3,
      'op_names': ['fc1']
   }, 
   {
      'quant_types': ['weight'],
      'quant_bits': 2,
      'op_names': ['fc2']
   }]

In this example, 'op_names' is the name of layer and four layers will be quantized to different quant_bits.


Export compression result
-------------------------

Export the pruned model
^^^^^^^^^^^^^^^^^^^^^^^

You can easily export the pruned model using the following API if you are pruning your model, ``state_dict`` of the sparse model weights will be stored in ``model.pth``\ , which can be loaded by ``torch.load('model.pth')``. Note that, the exported ``model.pth``\ has the same parameters as the original model except the masked weights are zero. ``mask_dict`` stores the binary value that produced by the pruning algorithm, which can be further used to speed up the model.

.. code-block:: python

   # export model weights and mask
   pruner.export_model(model_path='model.pth', mask_path='mask.pth')

   # apply mask to model
   from nni.compression.pytorch import apply_compression_results

   apply_compression_results(model, mask_file, device)


export model in ``onnx`` format(\ ``input_shape`` need to be specified):

.. code-block:: python

   pruner.export_model(model_path='model.pth', mask_path='mask.pth', onnx_path='model.onnx', input_shape=[1, 1, 28, 28])


Export the quantized model
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can export the quantized model directly by using ``torch.save`` api and the quantized model can be loaded by ``torch.load`` without any extra modification. The following example shows the normal procedure of saving, loading quantized model and get related parameters in QAT.

.. code-block:: python
   
   # Save quantized model which is generated by using NNI QAT algorithm
   torch.save(model.state_dict(), "quantized_model.pth")

   # Simulate model loading procedure
   # Have to init new model and compress it before loading
   qmodel_load = Mnist()
   optimizer = torch.optim.SGD(qmodel_load.parameters(), lr=0.01, momentum=0.5)
   quantizer = QAT_Quantizer(qmodel_load, config_list, optimizer)
   quantizer.compress()
   
   # Load quantized model
   qmodel_load.load_state_dict(torch.load("quantized_model.pth"))

   # Get scale, zero_point and weight of conv1 in loaded model
   conv1 = qmodel_load.conv1
   scale = conv1.module.scale
   zero_point = conv1.module.zero_point
   weight = conv1.module.weight


Speed up the model
------------------

Masks do not provide real speedup of your model. The model should be speeded up based on the exported masks, thus, we provide an API to speed up your model as shown below. After invoking ``apply_compression_results`` on your model, your model becomes a smaller one with shorter inference latency.

.. code-block:: python

   from nni.compression.pytorch import apply_compression_results, ModelSpeedup

   dummy_input = torch.randn(config['input_shape']).to(device)
   m_speedup = ModelSpeedup(model, dummy_input, masks_file, device)
   m_speedup.speedup_model()


Please refer to `here <ModelSpeedup.rst>`__ for detailed description. The example code for model speedup can be found :githublink:`here <examples/model_compress/pruning/model_speedup.py>`


Control the Fine-tuning process
-------------------------------

APIs to control the fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some compression algorithms control the progress of compression during fine-tuning (e.g. `AGP <../Compression/Pruner.rst#agp-pruner>`__\ ), and some algorithms need to do something after every minibatch. Therefore, we provide another two APIs for users to invoke: ``pruner.update_epoch(epoch)`` and ``pruner.step()``.

``update_epoch`` should be invoked in every epoch, while ``step`` should be invoked after each minibatch. Note that most algorithms do not require calling the two APIs. Please refer to each algorithm's document for details. For the algorithms that do not need them, calling them is allowed but has no effect.

Enhance the fine-tuning process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Knowledge distillation effectively learns a small student model from a large teacher model. Users can enhance the fine-tuning process that utilize knowledge distillation to improve the performance of the compressed model. Example code can be found :githublink:`here <examples/model_compress/pruning/finetune_kd_torch.py>`
