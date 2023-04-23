Model Evaluator
===============

A model evaluator is for training and validating each generated model. They are necessary to evaluate the performance of new explored models.

.. _functional-evaluator:

Customize Evaluator with Any Function
-------------------------------------

The simplest way to customize a new evaluator is with :class:`~nni.nas.evaluator.FunctionalEvaluator`, which is very easy when training code is already available. Users only need to write a fit function that wraps everything, which usually includes training, validating and testing of a single model. This function takes one positional arguments (``model``) and possible keyword arguments. The keyword arguments (other than ``model``) are fed to :class:`~nni.nas.evaluator.FunctionalEvaluator` as its initialization parameters (note that they will be :doc:`serialized <./serialization>`). In this way, users get everything under their control, but expose less information to the framework and as a result, further optimizations like :ref:`CGO <cgo-execution-engine>` might be not feasible. An example is as belows:

.. code-block:: python

    from nni.nas.evaluator import FunctionalEvaluator
    from nni.nas.experiment import NasExperiment

    def fit(model, dataloader):
        train(model, dataloader)
        acc = test(model, dataloader)
        nni.report_final_result(acc)

    # The dataloader will be serialized, thus ``nni.trace`` is needed here.
    # See serialization tutorial for more details.
    evaluator = FunctionalEvaluator(fit, dataloader=nni.trace(DataLoader)(foo, bar))
    experiment = NasExperiment(base_model, lightning, strategy)

.. note::

   Different from the legacy Retiarii FunctionEvaluator, the new FunctionalEvaluator now accepts model instance as the first argument, rather than ``model_cls``. This makes it more intuitive and easier to use.

.. tip::

    When using customized evaluators, if you want to visualize models, you need to export your model and save it into ``$NNI_OUTPUT_DIR/model.onnx`` in your evaluator. An example here:

    .. code-block:: python

        def fit(model):
            onnx_path = Path(os.environ.get('NNI_OUTPUT_DIR', '.')) / 'model.onnx'
            onnx_path.parent.mkdir(exist_ok=True)
            dummy_input = torch.randn(10, 3, 224, 224)
            torch.onnx.export(model, dummy_input, onnx_path)
            # the rest of training code here

    If the conversion is successful, the model will be able to be visualized with powerful tools `Netron <https://netron.app/>`__.

Use Evaluators to Train and Evaluate Models
-------------------------------------------

Users can use evaluators to train or evaluate a single, concrete architecture. This is very useful when:

* Debugging your evaluator against a baseline model.
* Fully train, validate and test your model after the search process is complete.

The usage is shown below:

.. code-block:: python

   # Class definition of a model space, for example, ResNet.
   class MyModelSpace(ModelSpace):
        ...

   # Mock a model instance
   from nni.nas.space import RawFormatModelSpace
   model_container = RawFormatModelSpace.from_model(MyModelSpace())

   # Randomly sample a model
   model = model_container.random()

   # Mock a runtime so that `nni.get_next_parameter` and `nni.report_xxx_result` will work.
   with evaluator.mock_runtime(model):
       evaluator.evaluate(model.executable_model())

The underlying implementation of :meth:`~nni.nas.Evaluator.evaluate` depends on concrete evaluator that you used.
For example, if :class:`~nni.nas.evaluator.FunctionalEvaluator` is used, it will run your customized fit function.
If lightning evaluators like :class:`nni.nas.evaluator.pytorch.Classification` are used, it will invoke the ``trainer.fit()`` of Lightning.

To evaluate an architecture that is exported from experiment (i.e., from :meth:`~nni.nas.experiment.NasExperiment.export_top_models`), use :func:`nni.nas.space.model_context` to instantiate the exported model::

    with model_context(exported_model_dict):
        model = MyModelSpace()
    # Then use evaluator.evaluate
    evaluator.evaluate(model)

Another way of doing this is probably using ``freeze`` API. It will also preserve the weights at best effort if the model space has been mutated by one-shot strategies::

    MyModelSpace().freeze(exported_model_dict)

.. _lightning-evaluator:

Evaluators with PyTorch-Lightning
---------------------------------

Use Built-in Evaluators
^^^^^^^^^^^^^^^^^^^^^^^

NNI provides some commonly used model evaluators for users' convenience. These evaluators are built upon the awesome library PyTorch-Lightning. Read the :doc:`reference </reference/nas>` for their detailed usages.

* :class:`nni.nas.evaluator.pytorch.Classification`: for classification tasks.
* :class:`nni.nas.evaluator.pytorch.Regression`: for regression tasks.

We recommend to read the :doc:`serialization tutorial <serialization>` before using these evaluators. A few notes to summarize the tutorial:

1. :class:`nni.nas.evaluator.pytorch.DataLoader` should be used in place of ``torch.utils.data.DataLoader``.
2. The datasets used in data-loader should be decorated with :meth:`nni.trace` recursively.

For example,

.. code-block:: python

  import nni.nas.evaluator.pytorch.lightning as pl
  from torchvision import transforms

  transform = nni.trace(transforms.Compose, [nni.trace(transforms.ToTensor()), nni.trace(transforms.Normalize, (0.1307,), (0.3081,))])
  train_dataset = nni.trace(MNIST, root='data/mnist', train=True, download=True, transform=transform)
  test_dataset = nni.trace(MNIST, root='data/mnist', train=False, download=True, transform=transform)

  # pl.DataLoader and pl.Classification is already traced and supports serialization.
  evaluator = pl.Classification(train_dataloaders=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=10)

Customize Evaluator with PyTorch-Lightning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another approach is to write training code in PyTorch-Lightning style, that is, to write a LightningModule that defines all elements needed for training (e.g., loss function, optimizer) and to define a trainer that takes (optional) dataloaders to execute the training. Before that, please read the `document of PyTorch-lightning <https://pytorch-lightning.readthedocs.io/>`__ to learn the basic concepts and components provided by PyTorch-lightning.

In practice, writing a new training module in nas should inherit :class:`nni.nas.evaluator.pytorch.LightningModule`, which has a ``set_model`` that will be called after ``__init__`` to save the candidate model (generated by strategy) as ``self.model``. The rest of the process (like ``training_step``) should be the same as writing any other lightning module. Evaluators should also communicate with strategies via two API calls (:meth:`nni.report_intermediate_result` for periodical metrics and :meth:`nni.report_final_result` for final metrics), added in ``on_validation_epoch_end`` and ``teardown`` respectively. 

An example is as follows:

.. code-block:: python

    from nni.nas.evaluator.pytorch.lightning import LightningModule  # please import this one

    @nni.trace
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

.. note::

   If you are trying to use your customized evaluator with one-shot strategy, bear in mind that your defined methods will be reassembled into another LightningModule, which might result in extra constraints when writing the LightningModule. For example, your validation step could appear else where (e.g., in ``training_step``). This prohibits you from returning arbitrary object in ``validation_step``.

Then, users need to wrap everything (including LightningModule, trainer and dataloaders) into a :class:`nni.nas.evaluator.pytorch.Lightning` object, and pass this object into a nas experiment.

.. code-block:: python

    import nni.nas.evaluator.pytorch.lightning as pl
    from nni.nas.experiment import NasExperiment

    lightning = pl.Lightning(AutoEncoder(),
                             pl.Trainer(max_epochs=10),
                             train_dataloaders=pl.DataLoader(train_dataset, batch_size=100),
                             val_dataloaders=pl.DataLoader(test_dataset, batch_size=100))
    experiment = NasExperiment(base_model, lightning, strategy)
