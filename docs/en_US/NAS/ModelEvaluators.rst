Model Evaluators
================

A model evaluator is for training and validating each generated model.

Usage of Model Evaluator
------------------------

In multi-trial NAS, a sampled model should be able to be executed on a remote machine or a training platform (e.g., AzureML, OpenPAI). Thus, both the model and its model evaluator should be correctly serialized. To make NNI correctly serialize model evaluator, users should apply ``serialize`` on some of their functions and objects.

.. _serializer:

`serialize <./ApiReference.rst#utilities>`__ enables re-instantiation of model evaluator in another process or machine. It is implemented by recording the initialization parameters of user instantiated evaluator.

The evaluator related APIs provided by Retiarii have already supported serialization, for example ``pl.Classification``, ``pl.DataLoader``, no need to apply ``serialize`` on them. In the following case users should use ``serialize`` API manually.

If the initialization parameters of the evaluator APIs (e.g., ``pl.Classification``, ``pl.DataLoader``) are not primitive types (e.g., ``int``, ``string``), they should be applied with  ``serialize``. If those parameters' initialization parameters are not primitive types, ``serialize`` should also be applied. In a word, ``serialize`` should be applied recursively if necessary.

Below is an example, ``transforms.Compose``, ``transforms.Normalize``, and ``MNIST`` are serialized manually using ``serialize``. ``serialize`` takes a class ``cls`` as its first argument, its following arguments are the arguments for initializing this class. ``pl.Classification`` is not applied ``serialize`` because it is already serializable as an API provided by NNI.

.. code-block:: python

  import nni.retiarii.evaluator.pytorch.lightning as pl
  from nni.retiarii import serialize
  from torchvision import transforms

  transform = serialize(transforms.Compose, [serialize(transforms.ToTensor()), serialize(transforms.Normalize, (0.1307,), (0.3081,))])
  train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
  test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)
  evaluator = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=10)

Supported Model Evaluators
--------------------------

NNI provides some commonly used model evaluators for users' convenience. These evaluators are built upon the awesome library PyTorch-Lightning. If these model evaluators do not meet users' requirement, they can customize new model evaluators following the tutorial `here <./WriteTrainer.rst>`__.

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.Classification
    :noindex:

..  autoclass:: nni.retiarii.evaluator.pytorch.lightning.Regression
    :noindex:

Customize A New Model Evaluator
-------------------------------

Model Evaluator is necessary to evaluate the performance of new explored models. A model evaluator usually includes training, validating and testing of a single model. We provide two ways for users to write a new model evaluator, which will be demonstrated below respectively.

With FunctionalEvaluator
^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to customize a new evaluator is with functional APIs, which is very easy when training code is already available. Users only need to write a fit function that wraps everything. This function takes one positional arguments (``model_cls``) and possible keyword arguments. The keyword arguments (other than ``model_cls``) are fed to FunctionEvaluator as its initialization parameters. In this way, users get everything under their control, but expose less information to the framework and thus fewer opportunities for possible optimization. An example is as belows:

.. code-block:: python

    from nni.retiarii.evaluator import FunctionalEvaluator
    from nni.retiarii.experiment.pytorch import RetiariiExperiment

    def fit(model_cls, dataloader):
        model = model_cls()
        train(model, dataloader)
        acc = test(model, dataloader)
        nni.report_final_result(acc)

    evaluator = FunctionalEvaluator(fit, dataloader=DataLoader(foo, bar))
    experiment = RetiariiExperiment(base_model, evaluator, mutators, strategy)

.. note:: Due to our current implementation limitation, the ``fit`` function should be put in another python file instead of putting it in the main file. This limitation will be fixed in future release.

.. note:: When using customized evaluators, if you want to visualize models, you need to export your model and save it into ``$NNI_OUTPUT_DIR/model.onnx`` in your evaluator.

With PyTorch-Lightning
^^^^^^^^^^^^^^^^^^^^^^

It's recommended to write training code in PyTorch-Lightning style, that is, to write a LightningModule that defines all elements needed for training (e.g., loss function, optimizer) and to define a trainer that takes (optional) dataloaders to execute the training. Before that, please read the `document of PyTorch-lightning <https://pytorch-lightning.readthedocs.io/>`__ to learn the basic concepts and components provided by PyTorch-lightning.

In practice, writing a new training module in Retiarii should inherit ``nni.retiarii.evaluator.pytorch.lightning.LightningModule``, which has a ``set_model`` that will be called after ``__init__`` to save the candidate model (generated by strategy) as ``self.model``. The rest of the process (like ``training_step``) should be the same as writing any other lightning module. Evaluators should also communicate with strategies via two API calls (``nni.report_intermediate_result`` for periodical metrics and ``nni.report_final_result`` for final metrics), added in ``on_validation_epoch_end`` and ``teardown`` respectively. 

An example is as follows:

.. code-block:: python

    from nni.retiarii.evaluator.pytorch.lightning import LightningModule  # please import this one

    @basic_unit
    class AutoEncoder(LightningModule):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 28*28)
            )

        def forward(self, x):
            embedding = self.model(x)  # let's search for encoder
            return embedding

        def training_step(self, batch, batch_idx):
            # training_step defined the train loop.
            # It is independent of forward
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.model(x)  # model is the one that is searched for
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            # Logging to TensorBoard by default
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.model(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            self.log('val_loss', loss)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

        def on_validation_epoch_end(self):
            nni.report_intermediate_result(self.trainer.callback_metrics['val_loss'].item())

        def teardown(self, stage):
            if stage == 'fit':
                nni.report_final_result(self.trainer.callback_metrics['val_loss'].item())

Then, users need to wrap everything (including LightningModule, trainer and dataloaders) into a ``Lightning`` object, and pass this object into a Retiarii experiment.

.. code-block:: python

    import nni.retiarii.evaluator.pytorch.lightning as pl
    from nni.retiarii.experiment.pytorch import RetiariiExperiment

    lightning = pl.Lightning(AutoEncoder(),
                             pl.Trainer(max_epochs=10),
                             train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                             val_dataloaders=pl.DataLoader(test_dataset, batch_size=100))
    experiment = RetiariiExperiment(base_model, lightning, mutators, strategy)
