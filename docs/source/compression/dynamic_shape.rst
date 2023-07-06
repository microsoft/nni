Compression for Models with Dynamic-shape Input
==================
Compression for models with dynamic-shape input is a novel experimental feature incorporated into NNI 3.0.
This feature makes deployment more convenient. For example, when we feed multiple images to the neural network, we don't have to create multiple models decided by their height and width. And when we feed a piece of text into a neural network, the length of the text is no longer limited by a fixed format.
This feature is mainly achieved through two steps. First, create a dynamic ONNX model, and then create a dynamic TensorRT engine through the dynamic ONNX model.
.. Note::

    NNI strives to ensure maximum compatibility among different compressors in dynamic-shape compression.
    Nevertheless, it is impossible to avoid mutual interference in model modification between different compression algorithms in some individual scenarios.
    We encourage users to integrate algorithms after acquiring a comprehensive understanding of the fundamental principles of compression methods.
    If you encounter any problems or doubts that cannot be resolved while using dynamic-shape compression, you are welcome to raise an issue for discussion.

Main API
--------

To explain how dynamic-shape compression worked, we should know that each module in the model has a corresponding wrapper in the compressor.
The wrapper stores the necessary data required for compression.
``ModelSpeedupTensorRT`` append ``dummy_input`` as a parameter instead of ``input_shape``. 
``dummy_input`` is an input that satisfies the torch model you want to deploy. It is used to create a onnx-model.
In addition, you should provide two parameters, ``dynamic_axes`` and ``dynamic_shape_setting``.
``dynamic_axes`` determine which dimensions of the model's input (or output) you set as dynamic.
``dynamic_shape_setting`` is to determine the specific range of the dynamic shape you set.

Example
-------
Quantize Bert and Deploy Model into ONNX&TensorRT with Dynamic-shape input

The full example can be found `here <https://github.com/microsoft/nni/examples/tutorials/quantization_bert_glue.py>`__.

The following code is a common pipeline with quantization first and then deployment.

.. code-block:: python
    ...
    task_name = 'rte' 
    finetune_lr = 4e-5
    quant_lr = 1e-5
    quant_method = 'ptq'
    ...
    config_list = [{
        'op_types': ['Linear'],
        'op_names_re': ['bert.encoder.layer.{}'.format(i) for i in range(12)],
        'target_names': ['weight', '_input_','_output_'],
        'quant_dtype': 'int8',
        'quant_scheme': 'symmetric',#'affine''symmetric'
        'granularity': 'default',
    }]

The same steps as the normal quantization by nni, first set the hyperparameters of the quantizer configuration.
When the fake-quantize finished, save the parameters of the quantization node as ``calibration_config``, 
and then remove the quantization node in the model by ``quantizer.unwrap_model()``.
Prepare the ``dummy_input`` required by the input model. 
In order to more accurately meet the model input requirements, it is recommended to extract ``dummy_input`` directly from the training-dataset or val-dataset of the task.
Modify the ``dummy_input`` to the ``dict`` data type through the function ``transfer_dummy_input``.

.. code-block:: python
    ...
    input_names=['input_ids','token_type_ids','attention_mask']
    dummy_input = transfer_dummy_input(dummy_input,input_names)


``dynamic_axes`` is a dict. The dict keys are names of input and output whose shape is dynamic,
 the dict values are dicts which specify dimensions are dynamic.
``dynamic_shape_setting`` requires you to provide three parameters, which are the smallest shape of your input, the commonly used shape, and the largest shape.
 These three parameters facilitate TensorRT to allocate memory space to the model.

.. code-block:: python
    ...
    dynamic_axes={'input_ids' : {1 : 'seq_len'},
                'token_type_ids' : {1 : 'seq_len'},
                'attention_mask' : {1 : 'seq_len'}}
    dynamic_shape_setting ={'min_shape' : (1,18),
                            'opt_shape' : (1,72),
                            'max_shape' : (1,360)}
    ...
.. code-block:: python
    ...
    engine = ModelSpeedupTensorRT(model, dummy_input=dummy_input, config=calibration_config, onnx_path='bert_rte.onnx',input_names=['input_ids','token_type_ids','attention_mask'],output_names=['output'],
    dynamic_axes = dynamic_axes,
    dynamic_shape_setting = dynamic_shape_setting)
    engine.compress()

After ``engine.compress()``,you get a TensorRT engine of original model. 
You can test model's output and inference time  by ``output, time_span = engine.inference(dummy_input)``
You can test model's accuracy by ``test_Accuracy(engine)``