Quick Start of Retiarii on NNI
==============================


.. contents::

In this quick start tutorial, we use multi-trial NAS as an example to show how to construct and explore a model space. There are mainly three crucial components for a neural architecture search task, namely,

* Model search space that defines the set of models to explore.
* A proper strategy as the method to explore this search space.
* A model evaluator that reports the performance of a given model.

One-shot NAS tutorial can be found `here <./OneshotTrainer.rst>`__.

.. note:: Currently, PyTorch is the only supported framework by Retiarii, and we have only tested with **PyTorch 1.6 to 1.9**. This documentation assumes PyTorch context but it should also apply to other frameworks, that is in our future plan.

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

A base model is only one concrete model not a model space. We provide `APIs and primitives <./MutationPrimitives.rst>`__ for users to express how the base model can be mutated. That is, to build a model space which includes many models.

Based on the above base model, we can define a model space as below. 

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
  -    self.conv = nn.Conv2d(32, 1, 5)
  +    self.conv = nn.LayerChoice([
  +                  nn.Conv2d(32, 1, 5),
  +                  nn.Conv2d(32, 1, 5, bias=False)
  +                ]) 
  -    self.pool = nn.MaxPool2d(kernel_size=2)
  +    self.pool = nn.MaxPool2d(kernel_size=nn.ValueChoice([2, 4]))
    def forward(self, x):
      return self.pool(self.conv(x))

  @model_wrapper      # this decorator should be put on the out most PyTorch module
  class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.convpool = ConvPool()
  -    self.mymodule = BasicBlock(2.)
  +    self.mymodule = BasicBlock(nn.ValueChoice([2., 3.]))
    def forward(self, x):
      return F.relu(self.convpool(self.mymodule(x)))

This example uses two mutation APIs, `nn.LayerChoice` and `nn.ValueChoice`. `nn.LayerChoice` takes a list of candidate modules, one will be chosen for each sampled model. It can be used just like PyTorch module. `nn.ValueChoice` takes a list of candidate values, one will be chosen to take effect for each sampled model.

More detailed API description and usage can be found `here <./construct_space.rst>`__\.

.. note:: We are actively enriching the mutation APIs, to facilitate easy construction of model space. If the currently supported mutation APIs cannot express your model space, please refer to `this doc <./Mutators.rst>`__ for customizing mutators.

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

Users can visualize their experiment in the same way as visualizing a normal hyper-parameter tuning experiment. For example, open ``localhost::8081`` in your browser, 8081 is the port that you set in ``exp.run``. Please refer to `here <../Tutorial/WebUI.rst>`__ for details.

We support visualizing models with 3rd-party visualization engines (like `Netron <https://netron.app/>`__). This can be used by clicking ``Visualization`` in detail panel for each trial. Note that current visualization is based on `onnx <https://onnx.ai/>`__ . Built-in evaluators (e.g., Classification) will automatically export the model into a file, for your own evaluator, you need to save your file into ``$NNI_OUTPUT_DIR/model.onnx`` to make this work.

Export Top Models
-----------------

Users can export top models after the exploration is done using ``export_top_models``.

.. code-block:: python

  for model_code in exp.export_top_models(formatter='dict'):
    print(model_code)
