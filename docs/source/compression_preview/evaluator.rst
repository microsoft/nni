Compression Evaluator
=====================

The ``Evaluator`` is used to package the training and evaluation process for a targeted model.
To explain why NNI needs an ``Evaluator``, let's first look at the general process of model compression in NNI.

In model pruning, some algorithms need to prune according to some intermediate variables (gradients, activations, etc.) generated during the training process,
and some algorithms need to gradually increase or adjust the sparsity of different layers during the training process,
or adjust the pruning strategy according to the performance changes of the model during the pruning process.

In model quantization, NNI has quantization-aware training algorithm,
it can adjust the scale and zero point required for model quantization from time to time during the training process,
and may achieve a better performance compare to post-training quantization.

In order to better support the above algorithms' needs and maintain the consistency of the interface,
NNI introduces the ``Evaluator`` as the carrier of the training and evaluation process.

.. note::
    For users prior to NNI v2.8: NNI previously provided APIs like ``trainer``, ``traced_optimizer``, ``criterion``, ``finetuner``.
    These APIs were maybe tedious in terms of user experience. Users need to exchange the corresponding API frequently if they want to switch compression algorithms.
    ``Evaluator`` is an alternative to the above interface, users only need to create the evaluator once and it can be used in all compressors.

For users of native PyTorch, :class:`TorchEvaluator <nni.contrib.compression.TorchEvaluator>` requires the user to encapsulate the training process as a function and exposes the specified interface,
which will bring some complexity. But don't worry, in most cases, this will not change too much code.

For users of `PyTorchLightning <https://www.pytorchlightning.ai/>`__, :class:`LightningEvaluator <nni.contrib.compression.LightningEvaluator>` can be created with only a few lines of code based on your original Lightning code.

For users of `Transformers Trainer <https://huggingface.co/docs/transformers/main_classes/trainer>`__, :class:`TransformersEvaluator <nni.contrib.compression.TransformersEvaluator>` can be created with only a few lines of code.

Here we give three examples of how to create an ``Evaluator`` for native PyTorch users, PyTorchLightning users and Huggingface Transformers users.

TorchEvaluator
--------------

:class:`TorchEvaluator <nni.contrib.compression.TorchEvaluator>` is for the users who work in a native PyTorch environment (If you are using PyTorchLightning, please refer `LightningEvaluator`_).

:class:`TorchEvaluator <nni.contrib.compression.TorchEvaluator>` has six initialization parameters ``training_func``, ``optimizers``, ``training_step``, ``lr_schedulers``,
``dummy_input``, ``evaluating_func``.

* ``training_func`` is the training loop to train the compressed model.
  It is a callable function with six input parameters ``model``, ``optimizers``,
  ``training_step``, ``lr_schedulers``, ``max_steps``, ``max_epochs``.
  Please make sure each input argument of the ``training_func`` is actually used,
  especially ``max_steps`` and ``max_epochs`` can correctly control the duration of training.
* ``optimizers`` is a single / a list of traced optimizer(s),
  please make sure using ``nni.trace`` wrapping the ``Optimizer`` class before initializing it / them.
* ``training_step`` A callable function, the first argument of inputs should be ``batch``, and the outputs should contain loss.
  Three kinds of outputs are supported: single loss, tuple with the first element is loss, a dict contains a key ``loss``.
* ``lr_schedulers`` is a single / a list of traced scheduler(s), same as ``optimizers``,
  please make sure using ``nni.trace`` wrapping the ``_LRScheduler`` class before initializing it / them.
* ``dummy_input`` is used to trace the model, same as ``example_inputs``
  in `torch.jit.trace <https://pytorch.org/docs/stable/generated/torch.jit.trace.html?highlight=torch%20jit%20trace#torch.jit.trace>`_.
* ``evaluating_func`` is a callable function to evaluate the compressed model performance. Its input is a compressed model and its output is metric.
  The format of metric should be a float number or a dict with key ``default``.

Please refer :class:`TorchEvaluator <nni.contrib.compression.TorchEvaluator>` for more details.
Here is an example of how to initialize a :class:`TorchEvaluator <nni.contrib.compression.TorchEvaluator>`.

.. code-block:: python

    pass

.. note::
    It is also worth to note that not all the arguments of :class:`TorchEvaluator <nni.contrib.compression.TorchEvaluator>` must be provided.
    Some compressors only require ``evaluate_func`` as they do not train the model, some compressors only require ``training_func``.
    Please refer to each compressor's doc to check the required arguments.
    But, it is fine to provide more arguments than the compressor's need.


A complete example of pruner using :class:`TorchEvaluator <nni.contrib.compression.TorchEvaluator>` to compress model can be found :githublink:`here <examples/model_compress/pruning/taylorfo_torch_evaluator.py>`.


LightningEvaluator
------------------
:class:`LightningEvaluator <nni.contrib.compression.LightningEvaluator>` is for the users who work with PyTorchLightning.

Only three parts users need to modify compared with the original pytorch-lightning code:

1. Wrap the ``Optimizer`` and ``_LRScheduler`` class with ``nni.trace``.
2. Wrap the ``LightningModule`` class with ``nni.trace``.
3. Wrap the ``LightningDataModule`` class with ``nni.trace``.

Please refer :class:`LightningEvaluator <nni.contrib.compression.LightningEvaluator>` for more details.
Here is an example of how to initialize a :class:`LightningEvaluator <nni.contrib.compression.LightningEvaluator>`.

.. code-block:: python

    pass

.. note::
    In ``LightningModule.configure_optimizers``, user should use traced ``torch.optim.Optimizer`` and traced ``torch.optim._LRScheduler``.
    It's for NNI can get the initialization parameters of the optimizers and lr_schedulers.

    .. code-block:: python

        class SimpleModel(pl.LightningModule):
            ...

            def configure_optimizers(self):
                optimizers = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.001)
                lr_schedulers = nni.trace(ExponentialLR)(optimizer=optimizers, gamma=0.1)
                return optimizers, lr_schedulers


A complete example of pruner using :class:`LightningEvaluator <nni.contrib.compression.LightningEvaluator>` to compress model can be found :githublink:`here <examples/model_compress/pruning/taylorfo_lightning_evaluator.py>`.


TransformersEvaluator
---------------------

TBD
