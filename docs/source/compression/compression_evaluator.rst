Compression Evaluator Creation
==============================

The evaluator is used to package the training and evaluation process for models with similar topologies.
To explain why NNI needs an evaluator, let's first look at the general process of model compression in NNI.

In model pruning, some algorithms need to prune according to some intermediate variables (gradients, activations, etc.) generated during the training process,
and some algorithms need to gradually increase or adjust the sparsity of different layers during the training process,
or adjust the pruning strategy according to the performance changes of the model during the pruning process.

In model quantization, NNI has quantization-aware training algorithm,
it can adjust the scale and zero point required for model quantization from time to time during the training process,
and may achieve a better performance compare to post-training quantization.

For the support of the above algorithms, NNI previously provided APIs like ``trainer``, ``traced_optimizer``, ``criterion``, ``finetuner``.
These APIs were maybe tedious in terms of user experience. Users need to exchange the corresponding API frequently if they want to switch compression algorithms.
After using the evaluator, users only need to consider one parameter to support NNI to complete the compression process.

For users of PytorchLightning, the evaluator can be created with only a few lines of code, and there is no need to make changes to the original code in most cases.
For users of native Pytorch, the evaluator requires the user to encapsulate the training process as a function and specifies the exposed interface,
which will bring some complexity. And don't worry, in most cases, this will not change too much code.

Here we give two examples of how to create an evaluator for both native Pytorch and PytorchLightning users.

TorchEvaluator
--------------

``TorchEvaluator`` is for the users who work in a native Pytorch environment.
Let's look at its API ``TorchEvaluator(training_func, optimizers, criterion, lr_schedulers, dummy_input, evaluating_func)``.

training_func
^^^^^^^^^^^^^
Here is an example of ``training_func``.

Training function has three required parameters, ``model``, ``optimizers`` and ``criterion``,
and three optional parameters, ``lr_schedulers``, ``max_steps``, ``max_epochs``.
Let's explain how NNI passes in these six parameters, but in most cases, users don't need to care what NNI passes in.
Users only need to treat these six parameters as the original parameters during the training process.

* The ``model`` is a wrapped model from the original model, it has a similar structure to the model to be pruned, so it can share training function with the original model.
* ``optimizers`` are reinitialized according to ``optimizers`` passed to the evaluator and the wrapped model's parameters.
* ``criterion`` also based on the ``criterion`` passed to the evaluator, it might be modified in some algorithms.
* If users use ``lr_schedulers`` in the ``training_func``, NNI will reinitialize the ``lr_schedulers`` with the reinitialized optimizers.
* ``max_steps`` is the NNI training duration limitation. An integer means that after ``max_steps`` steps, the training should stop. ``None`` means NNI doesn't limit the duration, it is up to users to decide when to stop.
* ``max_epochs`` is similar to the ``max_steps``, it controls the longest training epochs.

Note that ``optimizers`` and ``lr_schedulers`` passed to the ``training_func`` have the same type as the ``optimizers`` and ``lr_schedulers`` passed to evaluator,
a single ``torch.optim.Optimzier``/``torch.optim._LRScheduler`` instance or a list of them.
The following is a template of ``training_func``:

.. code-block:: python

    def training_func(model, optimizers, criterion, lr_schedulers=None, max_steps=None, max_epochs=None, *args, **kwargs):
        model.train()

        # prepare data
        train_dataloader = ...

        # nni may change the training duration by setting max_steps or max_epochs
        total_epochs = max_epochs if max_epochs else ...
        total_steps = max_steps if max_steps else ...
        current_steps = 0

        # training
        for _ in range(total_epochs):
            for x, y in train_dataloader:
                optimizers.zero_grad()
                ...
                loss = criterion(model(x), y)
                loss.backward()
                optimizers.step()
                ...
                current_steps += 1
                # stop the training loop when reach the total_steps
                if total_steps and current_steps == total_steps:
                    return

optimizers
^^^^^^^^^^
A single traced optimizer instance or a list of traced optimizers by ``nni.trace``.

NNI may modify the ``torch.optim.Optimizer`` member function ``step`` and/or generate new compression models,
so NNI needs to have the ability to reinitialize the optimizer. ``nni.trace`` can record the initialization parameters of a function/class,
which can then be used by NNI to reinitialize the optimizer for a new but structurally similar model.

.. code-block:: python

    import nni
    import torch

    model: torch.nn.Module = ...

    # single optimizer
    optimizers = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.001)
    # or, a list of optimizers
    optimizers = [nni.trace(torch.optim.SGD)(...), nni.trace(torch.optim.Adam)(...)]

criterion
^^^^^^^^^
``criterion`` function is used to compute the loss in the training function, the inputs of it are ``input`` and ``target``.
Sometimes, NNI needs to add additional loss to sparse the model parameters. NNI will change the ``criterion`` to do that,
e.g. in the ``training_func``, ``loss = criterion(input, target)`` will change to ``loss = patched_criterion(input, target)``.
The following is an example of ``criterion``:

.. code-block:: python

    import torch.nn.functional as F

    criterion = F.nll_loss

    # NNI maight patch this criterion function then pass it to ``training_func``.
    def patched_criterion(input, target):
        original_loss = criterion(input, target)

        # add l1 loss for all model parameters
        l1_loss = 0
        for param in model.parameters():
            l1_loss += param.norm(p=1)

        return original_loss + l1_loss

    training_func(..., criterion=patched_criterion)

lr_schedulers
^^^^^^^^^^^^^
A single traced lr_scheduler instance or a list of traced lr_schedulers by ``nni.trace``.

For the same reason with ``optimizers``, NNI needs the traced lr_scheduler to reinitialize it.
The following is an example:

.. code-block:: python

    import nni
    from torch.optim.lr_scheduler import ExponentialLR

    model: torch.nn.Module = ...

    # single lr_scheduler
    lr_schedulers = nni.trace(ExponentialLR)(optimizer=optimizers, gamma=0.1)
    # or, a list of lr_schedulers
    lr_schedulers = [nni.trace(ExponentialLR)(optimizer=optimizers, gamma=0.1), ...]

dummy_input
^^^^^^^^^^^
``dummy_input`` is used to trace the model graph, it's same with ``example_inputs`` in `torch.jit.trace <https://pytorch.org/docs/stable/generated/torch.jit.trace.html?highlight=torch%20jit%20trace#torch.jit.trace>`_.
It's only used by scheduled pruner intermediate model speedup right now.

evaluating_func
^^^^^^^^^^^^^^^
This is the function used to evaluate the compressed model performance. The input is a model and the output is a float metric or a tuple of float metric and information dict.
NNI will take the float metric as the model score, and assume the higher score means the better performance.
If you want to provide additional information, please put it into information dict.
The following is an ``evaluating_func`` example:

.. code-block:: python

    def evaluating_func(model):
        accuracy = ...
        f1 = ...
        # take the f1 score as the metric to NNI
        return f1, {'f1': f1, 'acc': accuracy}

TorchEvaluator Creation
^^^^^^^^^^^^^^^^^^^^^^^
After defining the above six parts (at least ``training_func``, ``optimizers`` and ``criterion``), the evaluator can be created.

.. code-block:: python

    from nni.compression.pytorch import TorchEvaluator

    evaluator = TorchEvaluator(training_func=training_func, optimizers=optimizers, criterion=criterion, lr_schedulers=lr_schedulers,
                            dummy_input=torch.rand(8, 1, 28, 28), evaluating_func=evaluating_func)

LightningEvaluator
------------------
``LightningEvaluator`` is for the users who work in PytorchLightning.
A few lines in the original code should be modified. The API of ``LightningEvaluator`` is ``LightningEvaluator(trainer, data_module, dummy_input)``.

Modifications in LightningModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In ``LightningModule.configure_optimizers``, user should use traced ``torch.optim.Optimizer`` and traced ``torch.optim._LRScheduler``.
It's for NNI can get the initialization parameters of the optimizers and lr_schedulers. The following is an example:

.. code-block:: python

    import nni
    import pytorch_lightning as pl

    class SimpleModel(pl.LightningModule):
        ...

        def configure_optimizers(self):
            optimizers = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.001)
            lr_schedulers = nni.trace(ExponentialLR)(optimizer=optimizers, gamma=0.1)
            return optimizers, lr_schedulers

trainer
^^^^^^^
``trainer`` is the traced ``pytorch_lightning.Trainer`` instance.
The following is an example:

.. code-block:: python

    trainer = nni.trace(pl.Trainer)(max_epochs=10)

data_module
^^^^^^^^^^^
``data_module`` is the traced ``pytorch_lightning.LightningDataModule`` instance.
The following is an example:

.. code-block:: python

    class SimpleDataModule(pl.LightningDataModule):
        ...

    data_module = nni.trace(SimpleDataModule)(...)

dummy_input
^^^^^^^^^^^
It is the same as the ``dummy_input`` in ``TorchEvaluator``.
``dummy_input`` is used to trace the model graph, it's same with ``example_inputs`` in `torch.jit.trace <https://pytorch.org/docs/stable/generated/torch.jit.trace.html?highlight=torch%20jit%20trace#torch.jit.trace>`_.
It's only used by scheduled pruner intermediate model speedup right now.

LightningEvaluator Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The evaluator can be created by the above three parts (at least ``trainer`` and ``data_module``).

.. code-block:: python

    from nni.compression.pytorch import LightningEvaluator
    evaluator = LightningEvaluator(trainer=trainer, data_module=data_module, dummy_input=torch.rand(8, 1, 28, 28))
