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

For users of native PyTorch, :class:`TorchEvaluator <nni.compression.TorchEvaluator>` requires the user to encapsulate the training process as a function and exposes the specified interface,
which will bring some complexity. But don't worry, in most cases, this will not change too much code.

For users of `PyTorchLightning <https://www.pytorchlightning.ai/>`__, :class:`LightningEvaluator <nni.compression.LightningEvaluator>` can be created with only a few lines of code based on your original Lightning code.

For users of `Transformers Trainer <https://huggingface.co/docs/transformers/main_classes/trainer>`__, :class:`TransformersEvaluator <nni.compression.TransformersEvaluator>` can be created with only a few lines of code.

Here we give three examples of how to create an ``Evaluator`` for native PyTorch users, PyTorchLightning users and Huggingface Transformers users.

TorchEvaluator
--------------

:class:`TorchEvaluator <nni.compression.TorchEvaluator>` is for the users who work in a native PyTorch environment (If you are using PyTorchLightning, please refer `LightningEvaluator`_).

:class:`TorchEvaluator <nni.compression.TorchEvaluator>` has six initialization parameters ``training_func``, ``optimizers``, ``training_step``, ``lr_schedulers``,
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

Please refer :class:`TorchEvaluator <nni.compression.TorchEvaluator>` for more details.
Here is an example of how to initialize a :class:`TorchEvaluator <nni.compression.TorchEvaluator>`.

.. code-block:: python

    def training_step(batch, model, *args, **kwargs):
        output = model(batch[0])
        loss = F.cross_entropy(output, batch[1])
        return loss

    def training_func(model, optimizer, training_step, lr_scheduler, max_steps, max_epochs):
        assert max_steps is not None or max_epochs is not None
        total_steps = max_steps if max_steps else max_epochs * len(train_dataloader)
        total_epochs = total_steps // len(train_dataloader) + (0 if total_steps % len(train_dataloader) == 0 else 1)

        current_step = 0
        for _ in range(total_epochs):
            for batch in train_dataloader:
                loss = training_step(batch, model)
                loss.backward()
                optimizer.step()

                # if reach the total steps, exit from the training loop
                current_step = current_step + 1
                if current_step >= total_steps:
                    return

            # if you are using a epoch-wise scheduler, call it here
            lr_scheduler.step()

    optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
    lr_scheduler = nni.trace(torch.optim.lr_scheduler.LambdaLR)(optimizer, lr_lambda=lambda epoch: 1 / epoch)

    evaluator = TorchEvaluator(training_func, optimizer, training_step, lr_scheduler)

.. note::
    It is also worth to note that not all the arguments of :class:`TorchEvaluator <nni.compression.TorchEvaluator>` must be provided.
    Some compressors only require ``evaluate_func`` as they do not train the model, some compressors only require ``training_func``.
    Please refer to each compressor's doc to check the required arguments.
    But, it is fine to provide more arguments than the compressor's need.


A complete example can be found :githublink:`here <examples/compression/evaluator/torch_evaluator.py>`.


LightningEvaluator
------------------
:class:`LightningEvaluator <nni.compression.LightningEvaluator>` is for the users who work with PyTorchLightning.

Only three parts users need to modify compared with the original pytorch-lightning code:

1. Wrap the ``Optimizer`` and ``LRScheduler`` class with ``nni.trace``.
2. Wrap the ``LightningModule`` class with ``nni.trace``.
3. Wrap the ``LightningDataModule`` class with ``nni.trace``.

Please refer :class:`LightningEvaluator <nni.compression.LightningEvaluator>` for more details.
Here is an example of how to initialize a :class:`LightningEvaluator <nni.compression.LightningEvaluator>`.

.. code-block:: python

    pl_trainer = nni.trace(pl.Trainer)(...)
    pl_data = nni.trace(MyDataModule)(...)

    evaluator = LightningEvaluator(pl_trainer, pl_data)

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


A complete example can be found :githublink:`here <examples/compression/evaluator/lightning_evaluator.py>`.


TransformersEvaluator
---------------------

:class:`TransformersEvaluator <nni.compression.TransformersEvaluator>` is for the users who work with Huggingface Transformers Trainer.

The only need is using ``nni.trace`` to wrap the Trainer class.

.. code-block:: python

    import nni
    from transformers.trainer import Trainer
    trainer = nni.trace(Trainer)(model, training_args, ...)

    from nni.compression.utils import TransformersEvaluator
    evaluator = TransformersEvaluator(trainer)

Moreover, if you are utilizing a personalized optimizer or learning rate scheduler, kindly use ``nni.trace`` to wrap their class as well.

.. code-block:: python

    optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
    lr_scheduler = nni.trace(torch.optim.lr_scheduler.LambdaLR)(optimizer, lr_lambda=lambda epoch: 1 / epoch)


A complete example of using a trainer with DeepSpeed mode under the TransformersEvaluator can be found: :githublink:`here <examples/model_compress/quantization/bert_quantization_with_ds.py>`.


DeepspeedTorchEvaluator
-----------------------

:class:`DeepspeedTorchEvaluator <nni.compression.DeepspeedTorchEvaluator>` is an evaluator designed specifically for native PyTorch users who are utilizing DeepSpeed.

:class:`DeepspeedTorchEvaluator <nni.compression.TorchEvaluator>` has eight initialization parameters ``training_func``,  ``training_step``, ``deepspeed``, ``optimizer``, ``lr_scheduler``,
``resume_from_checkpoint_args``, ``dummy_input``, ``evaluating_func``.

* ``training_func`` is the training loop to train the compressed model.
  It is a callable function with six input parameters ``model``, ``optimizers``,
  ``training_step``, ``lr_schedulers``, ``max_steps``, ``max_epochs``.
  Please make sure each input argument of the ``training_func`` is actually used,
  especially ``max_steps`` and ``max_epochs`` can correctly control the duration of training.
* ``training_step`` A callable function, the first argument of inputs should be ``batch``, and the outputs should contain loss.
  Three kinds of outputs are supported: single loss, tuple with the first element is loss, a dict contains a key ``loss``.
* ``deepspeed`` is the deepspeed configuration which Contains the parameters needed in DeepSpeed, such as train_batch_size, among others.
* ``optimizer`` is a single traced optimizer instance or a function that takes the model parameters as input and returns an optimizer instance.
  Please make sure using ``nni.trace`` wrapping the ``Optimizer`` class before initializing it / them if it is a single traced optimizer.
* ``lr_scheduler`` is a single traced lr_scheduler instance or a function that takes the model parameters and the optimizer as input and returns an lr_scheduler instance.
  Please make sure using ``nni.trace`` wrapping the ``_LRScheduler`` class before initializing it / them if it is a single traced scheduler.
* ``resume_from_checkpoint_args`` is used in the deepspeed_init process to load models saved during training with DeepSpeed.
* ``dummy_input`` is used to trace the model, same as ``example_inputs``
  in `torch.jit.trace <https://pytorch.org/docs/stable/generated/torch.jit.trace.html?highlight=torch%20jit%20trace#torch.jit.trace>`_.
* ``evaluating_func`` is a callable function to evaluate the compressed model performance. Its input is a compressed model and its output is metric.
  The format of metric should be a float number or a dict with key ``default``.

Please refer :class:`DeepspeedTorchEvaluator <nni.compression.DeepspeedTorchEvaluator>` for more details.
Here is an example of how to initialize a :class:`DeepspeedTorchEvaluator <nni.compression.DeepspeedTorchEvaluator>`.

.. code-block:: python

    def training_step(batch, model, *args, **kwargs):
        output = model(batch[0])
        loss = F.cross_entropy(output, batch[1])
        return loss

    def training_func(model, optimizer, training_step, lr_scheduler, max_steps, max_epochs):
        # here model is an instance of DeepSpeedEngine
        assert max_steps is not None or max_epochs is not None
        total_steps = max_steps if max_steps else max_epochs * len(train_dataloader)
        total_epochs = total_steps // len(train_dataloader) + (0 if total_steps % len(train_dataloader) == 0 else 1)

        current_step = 0
        for _ in range(total_epochs):
            for batch in train_dataloader:
                loss = training_step(batch, model)
                model.backward(model)
                model.step()

                # if reach the total steps, exit from the training loop
                current_step = current_step + 1
                if current_step >= total_steps:
                    return

            # if you are using a epoch-wise scheduler, call it here
            lr_scheduler.step()
    
        ds_config = {
            "gradient_accumulation_steps": 1,
            "steps_per_print": 2000,
            "wall_clock_breakdown": False,
            "train_batch_size": 128,
            "train_micro_batch_size_per_gpu": 128,
            "zero_force_ds_cpu_optimizer": False,
            "zero_allow_untested_optimizer": True
        }

    optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
    lr_scheduler = nni.trace(torch.optim.lr_scheduler.LambdaLR)(optimizer, lr_lambda=lambda epoch: 1 / epoch)

    evaluator = DeepspeedTorchEvaluator(training_func, training_step, ds_config, lr_scheduler)

.. note::
    It is also worth to note that not all the arguments of :class:`TorchEvaluator <nni.compression.TorchEvaluator>` must be provided.
    Some compressors only require ``evaluate_func`` as they do not train the model, some compressors only require ``training_func``.
    Please refer to each compressor's doc to check the required arguments.
    But, it is fine to provide more arguments than the compressor's need.


A complete example can be found :githublink:`here <examples/model_compress/quantization/quantization_with_deepspeed.py>`.