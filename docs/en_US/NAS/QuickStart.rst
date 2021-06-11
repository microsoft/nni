Quick Start of Retiarii on NNI
==============================


.. contents::

In this quick start tutorial, we use multi-trial NAS as an example to show how to construct and explore a model space. There are mainly three crucial components for a neural architecture search task, namely,

* Model search space that defines the set of models to explore.
* A proper strategy as the method to explore this search space.
* A model evaluator that reports the performance of a given model.

One-shot NAS tutorial can be found `here <./OneshotTrainer.rst>`__.

.. note:: Currently, PyTorch is the only supported framework by Retiarii, and we have only tested with **PyTorch 1.6 and 1.7**. This documentation assumes PyTorch context but it should also apply to other frameworks, that is in our future plan.

Define your Model Space
-----------------------

Model space is defined by users to express a set of models that users want to explore, which contains potentially good-performing models. In this framework, a model space is defined with two parts: a base model and possible mutations on the base model.

Define Base Model
^^^^^^^^^^^^^^^^^

Defining a base model is almost the same as defining a PyTorch (or TensorFlow) model. Usually, you only need to replace the code ``import torch.nn as nn`` with ``import nni.retiarii.nn.pytorch as nn`` to use our wrapped PyTorch modules.

Below is a very simple example of defining a base model, it is almost the same as defining a PyTorch model.

.. code-block:: python

  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn
  from nni.retiarii import model_wrapper

  class BasicBlock(nn.Module):
    def __init__(self, const):
      self.const = const
    def forward(self, x):
      return x + self.const

  class ConvPool(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv = nn.Conv2d(32, 1, 5)  # possibly mutate this conv
      self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
      return self.pool(self.conv(x))

  @model_wrapper      # this decorator should be put on the out most PyTorch module
  class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.convpool = ConvPool()
      self.mymodule = BasicBlock(2.)
    def forward(self, x):
      return F.relu(self.convpool(self.mymodule(x)))

Define Model Mutations
^^^^^^^^^^^^^^^^^^^^^^

A base model is only one concrete model not a model space. We provide APIs and primitives for users to express how the base model can be mutated, i.e., a model space which includes many models.

We provide some APIs as shown below for users to easily express possible mutations after defining a base model. The APIs can be used just like PyTorch module. This approach is also called inline mutations.

* ``nn.LayerChoice``. It allows users to put several candidate operations (e.g., PyTorch modules), one of them is chosen in each explored model.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # declared in `__init__` method
    self.layer = nn.LayerChoice([
      ops.PoolBN('max', channels, 3, stride, 1),
      ops.SepConv(channels, channels, 3, stride, 1),
      nn.Identity()
    ]))
    # invoked in `forward` method
    out = self.layer(x)

* ``nn.InputChoice``. It is mainly for choosing (or trying) different connections. It takes several tensors and chooses ``n_chosen`` tensors from them.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # declared in `__init__` method
    self.input_switch = nn.InputChoice(n_chosen=1)
    # invoked in `forward` method, choose one from the three
    out = self.input_switch([tensor1, tensor2, tensor3])

* ``nn.ValueChoice``. It is for choosing one value from some candidate values. It can only be used as input argument of basic units, that is, modules in ``nni.retiarii.nn.pytorch`` and user-defined modules decorated with ``@basic_unit``.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # used in `__init__` method
    self.conv = nn.Conv2d(XX, XX, kernel_size=nn.ValueChoice([1, 3, 5])
    self.op = MyOp(nn.ValueChoice([0, 1]), nn.ValueChoice([-1, 1]))

All the APIs have an optional argument called ``label``, mutations with the same label will share the same choice. A typical example is,

  .. code-block:: python

    self.net = nn.Sequential(
        nn.Linear(10, nn.ValueChoice([32, 64, 128], label='hidden_dim'),
        nn.Linear(nn.ValueChoice([32, 64, 128], label='hidden_dim'), 3)
    )

Detailed API description and usage can be found `here <./ApiReference.rst>`__\. Example of using these APIs can be found in :githublink:`Darts base model <test/retiarii_test/darts/darts_model.py>`. We are actively enriching the set of inline mutation APIs, to make it easier to express a new search space. Please refer to `here <./construct_space.rst>`__ for more tutorials about how to express complex model spaces.

Explore the Defined Model Space
-------------------------------

There are basically two exploration approaches: (1) search by evaluating each sampled model independently and (2) one-shot weight-sharing based search. We demonstrate the first approach below in this tutorial. Users can refer to `here <./OneshotTrainer.rst>`__ for the second approach.

Users can choose a proper exploration strategy to explore the model space, and use a chosen or user-defined model evaluator to evaluate the performance of each sampled model.

Pick a search strategy
^^^^^^^^^^^^^^^^^^^^^^^^

Retiarii supports many `exploration strategies <./ExplorationStrategies.rst>`__.

Simply choosing (i.e., instantiate) an exploration strategy as below.

.. code-block:: python

  import nni.retiarii.strategy as strategy

  search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted

Pick or write a model evaluator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the NAS process, the exploration strategy repeatedly generates new models. A model evaluator is for training and validating each generated model. The obtained performance of a generated model is collected and sent to the exploration strategy for generating better models.

In the context of PyTorch, Retiarii has provided two built-in model evaluators, designed for simple use cases: classification and regression. These two evaluators are built upon the awesome library PyTorch-Lightning.

An example here creates a simple evaluator that runs on MNIST dataset, trains for 10 epochs, and reports its validation accuracy.

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

As the model evaluator is running in another process (possibly in some remote machines), the defined evaluator, along with all its parameters, needs to be correctly serialized. For example, users should use the dataloader that has been already wrapped as a serializable class defined in ``nni.retiarii.evaluator.pytorch.lightning``. For the arguments used in dataloader, recursive serialization needs to be done, until the arguments are simple types like int, str, float.

Detailed descriptions and usages of model evaluators can be found `here <./ApiReference.rst>`__ .

If the built-in model evaluators do not meet your requirement, or you already wrote the training code and just want to use it, you can follow `the guide to write a new model evaluator <./WriteTrainer.rst>`__ .

.. note:: In case you want to run the model evaluator locally for debug purpose, you can directly run the evaluator via ``evaluator._execute(Net)`` (note that it has to be ``Net``, not ``Net()``). However, this API is currently internal and subject to change.

.. warning:: Mutations on the parameters of model evaluator (known as hyper-parameter tuning) is currently not supported but will be supported in the future.

.. warning:: To use PyTorch-lightning with Retiarii, currently you need to install PyTorch-lightning v1.1.x (v1.2 is not supported).

Launch an Experiment
--------------------

After all the above are prepared, it is time to start an experiment to do the model search. An example is shown below.

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, None, simple_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnasnet_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 10
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)

The complete code of a simple MNIST example can be found :githublink:`here <examples/nas/multi-trial/mnist/search.py>`. Users can also run Retiarii Experiment on `different training services <../training_services.rst>`__ besides ``local`` training service.

Visualize the Experiment
------------------------

Users can visualize their experiment in the same way as visualizing a normal hyper-parameter tuning experiment. For example, open ``localhost::8081`` in your browser, 8081 is the port that you set in ``exp.run``. Please refer to `here <../../Tutorial/WebUI.rst>`__ for details.

Export Top Models
-----------------

Users can export top models after the exploration is done using ``export_top_models``.

.. code-block:: python

  for model_code in exp.export_top_models(formatter='dict'):
    print(model_code)
