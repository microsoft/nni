Quick Start of Retiarii on NNI
==============================


.. contents::

In this quick start, we use multi-trial NAS as an example to show how to construct and explore a model space. There are mainly three crucial components for a neural architecture search task, namely,

* Model search space that defines a set of models to explore.
* A proper strategy as the method to explore this model space.
* A model evaluator that reports the performance of every model in the space.

The tutorial for One-shot NAS can be found `here <./OneshotTrainer.rst>`__.

Currently, PyTorch is the only supported framework by Retiarii, and we have only tested **PyTorch 1.7 to 1.10**. This documentation assumes PyTorch context but it should also apply to other frameworks, which is in our future plan.

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

.. tip:: Always keep in mind that you should use ``import nni.retiarii.nn.pytorch as nn`` and :meth:`nni.retiarii.model_wrapper`. Many mistakes are a result of forgetting one of those. Also, please use ``torch.nn`` for submodules of ``nn.init``, e.g., ``torch.nn.init`` instead of ``nn.init``. 

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

More detailed API description and usage can be found `here <./construct_space.rst>`__ .

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

Retiarii has provided `built-in model evaluators <./ModelEvaluators.rst>`__, but to start with, it is recommended to use ``FunctionalEvaluator``, that is, to wrap your own training and evaluation code with one single function. This function should receive one single model class and uses ``nni.report_final_result`` to report the final score of this model.

An example here creates a simple evaluator that runs on MNIST dataset, trains for 2 epochs, and reports its validation accuracy.

..  code-block:: python

    def evaluate_model(model_cls):
      # "model_cls" is a class, need to instantiate
      model = model_cls()

      optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
      transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
      train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
      test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)

      device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

      for epoch in range(3):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

      # report final test result
      nni.report_final_result(accuracy)

    # Create the evaluator
    evaluator = nni.retiarii.evaluator.FunctionalEvaluator(evaluate_model)

The ``train_epoch`` and ``test_epoch`` here can be any customized function, where users can write their own training recipe. See :githublink:`examples/nas/multi-trial/mnist/search.py` for the full example.

It is recommended that the ``evaluate_model`` here accepts no additional arguments other than ``model_cls``. However, in the `advanced tutorial <./ModelEvaluators.rst>`__, we will show how to use additional arguments in case you actually need those. In future, we will support mutation on the arguments of evaluators, which is commonly called "Hyper-parmeter tuning".

Launch an Experiment
--------------------

After all the above are prepared, it is time to start an experiment to do the model search. An example is shown below.

.. code-block:: python

  exp = RetiariiExperiment(base_model, evaluator, [], search_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnist_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 20
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)

The complete code of this example can be found :githublink:`here <examples/nas/multi-trial/mnist/search.py>`. Users can also run Retiarii Experiment with `different training services <../training_services.rst>`__ besides ``local`` training service.

Visualize the Experiment
------------------------

Users can visualize their experiment in the same way as visualizing a normal hyper-parameter tuning experiment. For example, open ``localhost:8081`` in your browser, 8081 is the port that you set in ``exp.run``. Please refer to `here <../Tutorial/WebUI.rst>`__ for details.

We support visualizing models with 3rd-party visualization engines (like `Netron <https://netron.app/>`__). This can be used by clicking ``Visualization`` in detail panel for each trial. Note that current visualization is based on `onnx <https://onnx.ai/>`__ , thus visualization is not feasible if the model cannot be exported into onnx.

Built-in evaluators (e.g., Classification) will automatically export the model into a file. For your own evaluator, you need to save your file into ``$NNI_OUTPUT_DIR/model.onnx`` to make this work. For instance,

.. code-block:: python

  def evaluate_model(model_cls):
    model = model_cls()
    # dump the model into an onnx
    if 'NNI_OUTPUT_DIR' in os.environ:
      torch.onnx.export(model, (dummy_input, ),
                        Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')
    # the rest are training and evaluation

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
