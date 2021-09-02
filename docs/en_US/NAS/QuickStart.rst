Quick Start of Retiarii on NNI
==============================


.. contents::

In this quick start, we use multi-trial NAS as an example to show how to construct and explore a model space. There are mainly three crucial components for a neural architecture search task, namely,

* Model search space that defines a set of models to explore.
* A proper strategy as the method to explore this model space.
* A model evaluator that reports the performance of every model in the space.

The tutorial for One-shot NAS can be found `here <./OneshotTrainer.rst>`__.

.. note:: Currently, PyTorch is the only supported framework by Retiarii, and we have only tested **PyTorch 1.6 to 1.9**. This documentation assumes PyTorch context but it should also apply to other frameworks, which is in our future plan.

Define your Model Space
-----------------------

Model space is defined by users to express a set of models that users want to explore, which contains potentially good-performing models. In this framework, a model space is defined with two parts: a base model and possible mutations on the base model.

Define Base Model
^^^^^^^^^^^^^^^^^

Defining a base model is almost the same as defining a PyTorch (or TensorFlow) model. Usually, you only need to replace the code ``import torch.nn as nn`` with ``import nni.retiarii.nn.pytorch as nn`` to use our wrapped PyTorch modules.

Below is a very simple example of defining a base model.

.. code-block:: python

  import torch
  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn
  from nni.retiarii import model_wrapper

  @model_wrapper      # this decorator should be put on the out most
  class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout(0.25)
      self.dropout2 = nn.Dropout(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(self.conv2(x), 2)
      x = torch.flatten(self.dropout1(x), 1)
      x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
      output = F.log_softmax(x, dim=1)
      return output

Define Model Mutations
^^^^^^^^^^^^^^^^^^^^^^

A base model is only one concrete model not a model space. We provide `APIs and primitives <./MutationPrimitives.rst>`__ for users to express how the base model can be mutated. That is, to build a model space which includes many models.

Based on the above base model, we can define a model space as below. 

.. code-block:: diff

  import torch
  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn
  from nni.retiarii import model_wrapper

  @model_wrapper
  class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
  -   self.conv2 = nn.Conv2d(32, 64, 3, 1)
  +   self.conv2 = nn.LayerChoice([
  +       nn.Conv2d(32, 64, 3, 1),
  +       DepthwiseSeparableConv(32, 64)
  +   ])
  -   self.dropout1 = nn.Dropout(0.25)
  +   self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))
      self.dropout2 = nn.Dropout(0.5)
  -   self.fc1 = nn.Linear(9216, 128)
  -   self.fc2 = nn.Linear(128, 10)
  +   feature = nn.ValueChoice([64, 128, 256])
  +   self.fc1 = nn.Linear(9216, feature)
  +   self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(self.conv2(x), 2)
      x = torch.flatten(self.dropout1(x), 1)
      x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
      output = F.log_softmax(x, dim=1)
      return output

This example uses two mutation APIs, ``nn.LayerChoice`` and ``nn.ValueChoice``. ``nn.LayerChoice`` takes a list of candidate modules (two in this example), one will be chosen for each sampled model. It can be used like normal PyTorch module. ``nn.ValueChoice`` takes a list of candidate values, one will be chosen to take effect for each sampled model.

More detailed API description and usage can be found `here <./construct_space.rst>`__\.

.. note:: We are actively enriching the mutation APIs, to facilitate easy construction of model space. If the currently supported mutation APIs cannot express your model space, please refer to `this doc <./Mutators.rst>`__ for customizing mutators.

Explore the Defined Model Space
-------------------------------

There are basically two exploration approaches: (1) search by evaluating each sampled model independently, which is the search approach in multi-trial NAS and (2) one-shot weight-sharing based search, which is used in one-shot NAS. We demonstrate the first approach in this tutorial. Users can refer to `here <./OneshotTrainer.rst>`__ for the second approach.

First, users need to pick a proper exploration strategy to explore the defined model space. Second, users need to pick or customize a model evaluator to evaluate the performance of each explored model.

Pick an exploration strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Retiarii supports many `exploration strategies <./ExplorationStrategies.rst>`__.

Simply choosing (i.e., instantiate) an exploration strategy as below.

.. code-block:: python

  import nni.retiarii.strategy as strategy

  search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted

Pick or customize a model evaluator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the exploration process, the exploration strategy repeatedly generates new models. A model evaluator is for training and validating each generated model to obtain the model's performance. The performance is sent to the exploration strategy for the strategy to generate better models.

Retiarii has provided two built-in model evaluators, designed for simple use cases: classification and regression. These two evaluators are built upon the awesome library PyTorch-Lightning.

An example here creates a simple evaluator that runs on MNIST dataset, trains for 2 epochs, and reports its validation accuracy.

.. code-block:: python

  import nni.retiarii.evaluator.pytorch.lightning as pl
  from nni.retiarii import serialize
  from torchvision import transforms

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
  test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)
  trainer = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                              val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                              max_epochs=2)

``serialize`` is for serializing the objects to make model evaluator executable on another process or another machine (e.g., on remote training service). Retiarii provided model evaluators and other classes are already serializable. Other objects should be applied ``serialize``, for example, ``MNIST`` in the above example.

Detailed descriptions and usages of model evaluators can be found `here <./ApiReference.rst>`__ .

If the built-in model evaluators do not meet your requirement, or you already wrote the training code and just want to use it, you can follow `the guide to write a new model evaluator <./WriteTrainer.rst>`__ .

.. warning:: Mutations on the parameters of model evaluator is currently not supported but will be supported in the future.

Launch an Experiment
--------------------

After all the above are prepared, it is time to start an experiment to do the model search. An example is shown below.

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, [], simple_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnist_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 20
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)

The complete code of this example can be found :githublink:`here <examples/nas/multi-trial/mnist/search.py>`. Users can also run Retiarii Experiment with `different training services <../training_services.rst>`__ besides ``local`` training service.

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

The output is `json` object which records the mutation actions of the top model. If users want to output source code of the top model, they can use graph-based execution engine for the experiment, by simply adding the following two lines.

.. code-block:: python

  exp_config.execution_engine = 'base'
  export_formatter = 'code'
