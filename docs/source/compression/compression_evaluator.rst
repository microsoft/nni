Compression Evaluator
=====================

The ``Evaluator`` is used to package the training and evaluation process for models with similar topologies.
To explain why NNI needs an ``Evaluator``, let's first look at the general process of model compression in NNI.

In model pruning, some algorithms need to prune according to some intermediate variables (gradients, activations, etc.) generated during the training process,
and some algorithms need to gradually increase or adjust the sparsity of different layers during the training process,
or adjust the pruning strategy according to the performance changes of the model during the pruning process.

In model quantization, NNI has quantization-aware training algorithm,
it can adjust the scale and zero point required for model quantization from time to time during the training process,
and may achieve a better performance compare to post-training quantization.

In order to better support the above algorithms' need and maintain the consistency of the interface,
NNI introduces the ``Evaluator`` as the carrier of the training and evaluation process.

.. note::
    For users prior to NNI v2.8: NNI previously provided APIs like ``trainer``, ``traced_optimizer``, ``criterion``, ``finetuner``.
    These APIs were maybe tedious in terms of user experience. Users need to exchange the corresponding API frequently if they want to switch compression algorithms.
    ``Evaluator`` is an alternative to the above interface, users only need to consider one parameter to support NNI to complete the compression process.

For users of native Pytorch, ``TorchEvaluator`` requires the user to encapsulate the training process as a function and specifies the exposed interface,
which will bring some complexity. But don't worry, in most cases, this will not change too much code.

For users of `PytorchLightning <https://www.pytorchlightning.ai/>`__, ``LightningEvaluator`` can be created with only a few lines of code, and there is no need to make changes to the original code in most cases.

Here we give two examples of how to create an ``Evaluator`` for both native Pytorch and PytorchLightning users.

TorchEvaluator
--------------

``TorchEvaluator`` is for the users who work in a native Pytorch environment. Here is an example for how to initialize a ``TorchEvaluator``.

.. code-block:: python

    from __future__ import annotations
    from typing import Callable, Any

    import torch
    from torch.optim.lr_scheduler import StepLR, _LRScheduler
    from torch.utils.data import DataLoader
    from torchvision import datasets, models

    import nni
    from nni.algorithms.compression.v2.pytorch import TorchEvaluator


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def training_func(model: torch.nn.Module, optimizers: torch.optim.Optimizer,
                      criterion: Callable[[Any, Any], torch.Tensor],
                      lr_schedulers: _LRScheduler | None = None, max_steps: int | None = None,
                      max_epochs: int | None = None, *args, **kwargs):
        model.train()

        # prepare data
        imagenet_train_data = datasets.ImageNet(root='data/imagenet', split='train', download=True)
        train_dataloader = DataLoader(imagenet_train_data, batch_size=4, shuffle=True)

        # NNI may change the training duration by setting max_steps or max_epochs.
        # To ensure that NNI has the ability to control the training duration,
        # please add max_steps and max_epochs as constraints to the training loop.
        total_epochs = max_epochs if max_epochs else 20
        total_steps = max_steps if max_steps else 1000000
        current_steps = 0

        # training loop
        for _ in range(total_epochs):
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizers.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizers.step()

                # stop the training loop when reach the total_steps
                current_steps += 1
                if total_steps and current_steps == total_steps:
                    return
            lr_schedulers.step()


    def evaluating_func(model: torch.nn.Module):
        model.eval()

        # prepare data
        imagenet_val_data = datasets.ImageNet(root='./data/imagenet', split='val', download=True)
        val_dataloader = DataLoader(imagenet_val_data, batch_size=4, shuffle=False)

        # testing loop
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(labels.view_as(preds)).sum().item()
        return correct / len(imagenet_val_data)


    # initialize the optimizer, criterion, lr_scheduler, dummy_input
    model = models.resnet18().to(device)
    # please use nni.trace wrap the optimizer class,
    # NNI will use the trace information to re-initialize the optimizer
    optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    # please use nni.trace wrap the lr_scheduler class,
    # NNI will use the trace information to re-initialize the lr_scheduler
    lr_scheduler = nni.trace(StepLR)(optimizer, step_size=5, gamma=0.1)
    dummy_input = torch.rand(4, 3, 224, 224).to(device)

    # TorchEvaluator initialization
    evaluator = TorchEvaluator(training_func=training_func, optimizers=optimizer, criterion=criterion,
                               lr_schedulers=lr_scheduler, dummy_input=dummy_input, evaluating_func=evaluating_func)


Let's introduce its API in detail.

training_func
^^^^^^^^^^^^^

Training function has three required parameters, ``model``, ``optimizers`` and ``criterion``,
and three optional parameters, ``lr_schedulers``, ``max_steps``, ``max_epochs``.
Let's explain how NNI passes in these six parameters, but in most cases, users don't need to care what NNI passes in.
Users only need to treat these six parameters as the original parameters during the training process.

* The ``model`` is a wrapped model from the original model, it has a similar structure to the model to be pruned, so it can share training function with the original model.
* ``optimizers`` are re-initialized according to ``optimizers`` passed to the evaluator and the wrapped model's parameters.
* ``criterion`` also based on the ``criterion`` passed to the evaluator, it might be modified in some algorithms.
* If users use ``lr_schedulers`` in the ``training_func``, NNI will re-initialize the ``lr_schedulers`` with the re-initialized optimizers.
* ``max_steps`` is the NNI training duration limitation. An integer means that after ``max_steps`` steps, the training should stop. ``None`` means NNI doesn't limit the duration, it is up to users to decide when to stop.
* ``max_epochs`` is similar to the ``max_steps``, it controls the longest training epochs.

Note that ``optimizers`` and ``lr_schedulers`` passed to the ``training_func`` have the same type as the ``optimizers`` and ``lr_schedulers`` passed to evaluator,
a single ``torch.optim.Optimzier``/ ``torch.optim._LRScheduler`` instance or a list of them.

optimizers
^^^^^^^^^^
A single traced optimizer instance or a list of traced optimizers by ``nni.trace``.

NNI may modify the ``torch.optim.Optimizer`` member function ``step`` and/or optimize compressed models,
so NNI needs to have the ability to re-initialize the optimizer. ``nni.trace`` can record the initialization parameters of a function/class,
which can then be used by NNI to re-initialize the optimizer for a new but structurally similar model.

criterion
^^^^^^^^^
``criterion`` function is used to compute the loss in the training function, the inputs of it are ``input`` and ``target``.
Sometimes, NNI needs to add additional loss to sparse the model parameters. NNI will change the ``criterion`` to do that,
e.g. in the ``training_func``, ``loss = criterion(input, target)`` will change to ``loss = patched_criterion(input, target)``.

lr_schedulers
^^^^^^^^^^^^^
A single traced lr_scheduler instance or a list of traced lr_schedulers by ``nni.trace``.

For the same reason with ``optimizers``, NNI needs the traced lr_scheduler to re-initialize it.

dummy_input
^^^^^^^^^^^
``dummy_input`` is used to trace the model graph, it's same with ``example_inputs`` in `torch.jit.trace <https://pytorch.org/docs/stable/generated/torch.jit.trace.html?highlight=torch%20jit%20trace#torch.jit.trace>`_.
It's only used by scheduled pruner intermediate model speedup right now.

evaluating_func
^^^^^^^^^^^^^^^
This is the function used to evaluate the compressed model performance.
The input is a model and the output is a float metric or a dict.
NNI will take the float metric as the model score, and assume the higher score means the better performance.
If you want to provide additional information, please put it into a dict and NNI will take the value of key ``default`` as evaluation metric.


LightningEvaluator
------------------
``LightningEvaluator`` is for the users who work with PytorchLightning.
Here is an example for how to initialize a ``LightningEvaluator``.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    import torch
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data import DataLoader
    from torchmetrics.functional import accuracy
    from torchvision import datasets, models

    import nni
    from nni.algorithms.compression.v2.pytorch import LightningEvaluator


    class SimpleLightningModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = models.resnet18()
            self.criterion = torch.nn.CrossEntropyLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            self.log("train_loss", loss)
            return loss

        def evaluate(self, batch, stage=None):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y)

            if stage:
                self.log(f"default", loss, prog_bar=False)
                self.log(f"{stage}_loss", loss, prog_bar=True)
                self.log(f"{stage}_acc", acc, prog_bar=True)

        def validation_step(self, batch, batch_idx):
            self.evaluate(batch, "val")

        def test_step(self, batch, batch_idx):
            self.evaluate(batch, "test")

        # please pay attention to this function,
        # using nni.trace trace the optimizer and lr_scheduler class.
        def configure_optimizers(self):
            optimizer = nni.trace(torch.optim.SGD)(
                self.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=5e-4,
            )
            scheduler_dict = {
                "scheduler": nni.trace(StepLR)(
                    optimizer,
                    step_size=5,
                    amma=0.1
                ),
                "interval": "epoch",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


    class ImageNetDataModule(pl.LightningDataModule):
        def __init__(self, data_dir: str = "./data/imagenet"):
            super().__init__()
            self.data_dir = data_dir

        def prepare_data(self):
            # download
            datasets.ImageNet(self.data_dir, split='train', download=True)
            datasets.ImageNet(self.data_dir, split='val', download=True)

        def setup(self, stage: str | None = None):
            if stage == "fit" or stage is None:
                self.imagenet_train_data = datasets.ImageNet(root='data/imagenet', split='train')
                self.imagenet_val_data = datasets.ImageNet(root='./data/imagenet', split='val')

            if stage == "test" or stage is None:
                self.imagenet_test_data = datasets.ImageNet(root='./data/imagenet', split='val')

            if stage == "predict" or stage is None:
                self.imagenet_predict_data = datasets.ImageNet(root='./data/imagenet', split='val')

        def train_dataloader(self):
            return DataLoader(self.imagenet_train_data, batch_size=4)

        def val_dataloader(self):
            return DataLoader(self.imagenet_val_data, batch_size=4)

        def test_dataloader(self):
            return DataLoader(self.imagenet_test_data, batch_size=4)

        def predict_dataloader(self):
            return DataLoader(self.imagenet_predict_data, batch_size=4)

    # please use nni.trace wrap the pl.Trainer class,
    # NNI will use the trace information to re-initialize the trainer
    pl_trainer = nni.trace(pl.Trainer)(
        accelerator='auto',
        devices=1,
        max_epochs=1,
        max_steps=50,
        logger=TensorBoardLogger('./lightning_logs', name="resnet"),
    )
    # please use nni.trace wrap the pl.LightningDataModule class,
    # NNI will use the trace information to re-initialize the datamodule
    pl_data = nni.trace(ImageNetDataModule)(data_dir='./data/imagenet')
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
