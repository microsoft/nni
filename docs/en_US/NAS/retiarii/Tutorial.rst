Neural Architecture Search with Retiarii (Experimental)
=======================================================

`Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ is a new framework to support neural architecture search and hyper-parameter tuning. It allows users to express various search space with high flexibility, to reuse many SOTA search algorithms, and to leverage system level optimizations to speed up the search process. This framework provides the following new user experiences.

* Search space can be expressed directly in user model code. A tuning space can be expressed along defining a model.
* Neural architecture candidates and hyper-parameter candidates are more friendly supported in an experiment.
* The experiment can be launched directly from python code.

*We are working on migrating* `our previous NAS framework <../Overview.rst>`__ *to Retiarii framework. Thus, this feature is still experimental. We recommend users to try the new framework and provide your valuable feedback for us to improve it. The old framework is still supported for now.*

.. contents::

There are mainly two steps to start an experiment for your neural architecture search task. First, define the model space you want to explore. Second, choose a search method to explore your defined model space.

Define your Model Space
-----------------------

Model space is defined by users to express a set of models that users want to explore, and believe good-performing models are included in those models. In this framework, a model space is defined with two parts: a base model and possible mutations on the base model.

Define Base Model
^^^^^^^^^^^^^^^^^

Defining a base model is almost the same as defining a PyTorch (or TensorFlow) model. There are only two small differences.

* Replace the code ``import torch.nn as nn`` with ``import nni.retiarii.nn.pytorch as nn`` for PyTorch modules, such as ``nn.Conv2d``, ``nn.ReLU``.
* Some **user-defined** modules should be decorated with ``@blackbox_module``. For example, user-defined module used in ``LayerChoice`` should be decorated. Users can refer to `here <#blackbox-module>`__ for detailed usage instruction of ``@blackbox_module``.

Below is a very simple example of defining a base model, it is almost the same as defining a PyTorch model.

.. code-block:: python

  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn

  class MyModule(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv = nn.Conv2d(32, 1, 5)
      self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
      return self.pool(self.conv(x))

  class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.mymodule = MyModule()
    def forward(self, x):
      return F.relu(self.mymodule(x))

Users can refer to :githublink:`Darts base model <test/retiarii_test/darts/darts_model.py>` and :githublink:`Mnasnet base model <test/retiarii_test/mnasnet/base_mnasnet.py>` for more complicated examples.

Define Model Mutations
^^^^^^^^^^^^^^^^^^^^^^

A base model is only one concrete model not a model space. We provide APIs and primitives for users to express how the base model can be mutated, i.e., a model space which includes many models.

**Express mutations in an inlined manner**

For easy usability and also backward compatibility, we provide some APIs for users to easily express possible mutations after defining a base model. The APIs can be used just like PyTorch module.

* ``nn.LayerChoice``. It allows users to put several candidate operations (e.g., PyTorch modules), one of them is chosen in each explored model. *Note that if the candidate is a user-defined module, it should be decorated as `blackbox module <#blackbox-module>`__. In the following example, ``ops.PoolBN`` and ``ops.SepConv`` should be decorated.*

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # declared in `__init__`
    self.layer = nn.LayerChoice([
      ops.PoolBN('max', channels, 3, stride, 1),
      ops.SepConv(channels, channels, 3, stride, 1),
      nn.Identity()
    ]))
    # invoked in `forward` function
    out = self.layer(x)

* ``nn.InputChoice``. It is mainly for choosing (or trying) different connections. It takes several tensors and chooses ``n_chosen`` tensors from them.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # declared in `__init__`
    self.input_switch = nn.InputChoice(n_chosen=1)
    # invoked in `forward` function, choose one from the three
    out = self.input_switch([tensor1, tensor2, tensor3])

* ``nn.ValueChoice``. It is for choosing one value from some candidate values. It can only be used as input argument of the modules in ``nn.modules`` and ``@blackbox_module`` decorated user-defined modules. *Note that it has not been officially supported.*

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # used in `__init__`
    self.conv = nn.Conv2d(XX, XX, kernel_size=nn.ValueChoice([1, 3, 5])
    self.op = MyOp(nn.ValueChoice([0, 1], nn.ValueChoice([-1, 1]))

Detailed API description and usage can be found `here <./ApiReference.rst>`__\. Example of using these APIs can be found in :githublink:`Darts base model <test/retiarii_test/darts/darts_model.py>`.

**Express mutations with mutators**

Though easy-to-use, inline mutations have limited expressiveness, some model spaces cannot be expressed. To improve expressiveness and flexibility, we provide primitives for users to write *Mutator* to express how they want to mutate base model more flexibly. Mutator stands above base model, thus has full ability to edit the model.

Users can instantiate several mutators as below, the mutators will be sequentially applied to the base model one after another for sampling a new model.

.. code-block:: python

  applied_mutators = []
  applied_mutators.append(BlockMutator('mutable_0'))
  applied_mutators.append(BlockMutator('mutable_1'))

``BlockMutator`` is defined by users to express how to mutate the base model. User-defined mutator should inherit ``Mutator`` class, and implement mutation logic in the member function ``mutate``.

.. code-block:: python

  from nni.retiarii import Mutator
  class BlockMutator(Mutator):
    def __init__(self, target: str, candidates: List):
        super(BlockMutator, self).__init__()
        self.target = target
        self.candidate_op_list = candidates

    def mutate(self, model):
      nodes = model.get_nodes_by_label(self.target)
      for node in nodes:
        chosen_op = self.choice(self.candidate_op_list)
        node.update_operation(chosen_op.type, chosen_op.params)

The input of ``mutate`` is graph IR of the base model (please refer to `here <./ApiReference.rst>`__ for the format and APIs of the IR), users can mutate the graph with its member functions (e.g., ``get_nodes_by_label``, ``update_operation``). The mutation operations can be combined with the API ``self.choice``, in order to express a set of possible mutations. In the above example, the node's operation can be changed to any operation from ``candidate_op_list``.

Use placehoder to make mutation easier: ``nn.Placeholder``. If you want to mutate a subgraph or node of your model, you can define a placeholder in this model to represent the subgraph or node. Then, use mutator to mutate this placeholder to make it real modules.

.. code-block:: python

  ph = nn.Placeholder(label='mutable_0',
    related_info={
      'kernel_size_options': [1, 3, 5],
      'n_layer_options': [1, 2, 3, 4],
      'exp_ratio': exp_ratio,
      'stride': stride
    }
  )

``label`` is used by mutator to identify this placeholder, ``related_info`` is the information that are required by mutator. As ``related_info`` is a dict, it could include any information that users want to put to pass it to user defined mutator. The complete example code can be found in :githublink:`Mnasnet base model <test/retiarii_test/mnasnet/base_mnasnet.py>`.

Explore the Defined Model Space
-------------------------------

After model space is defined, it is time to explore this model space. Users can choose proper search and training approach to explore the model space.

Create a Trainer and Exploration Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Classic search approach:**
In this approach, trainer is for training each explored model, while strategy is for sampling the models. Both trainer and strategy are required to explore the model space. We recommend PyTorch-Lightning to write the full training process.

**Oneshot (weight-sharing) search approach:**
In this approach, users only need a oneshot trainer, because this trainer takes charge of both search and training.

In the following table, we listed the available trainers and strategies.

.. list-table::
  :header-rows: 1
  :widths: auto

  * - Trainer
    - Strategy
    - Oneshot Trainer
  * - Classification
    - TPEStrategy
    - DartsTrainer
  * - Regression
    - RandomStrategy
    - EnasTrainer
  * - 
    - 
    - ProxylessTrainer
  * - 
    - 
    - SinglePathTrainer (RandomTrainer)

There usage and API document can be found `here <./ApiReference>`__\.

Here is a simple example of using trainer and strategy.

.. code-block:: python

  import nni.retiarii.trainer.pytorch.lightning as pl
  from nni.retiarii import blackbox
  from torchvision import transforms

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  train_dataset = blackbox(MNIST, root='data/mnist', train=True, download=True, transform=transform)
  test_dataset = blackbox(MNIST, root='data/mnist', train=False, download=True, transform=transform)
  lightning = pl.Lightning(pl.Classification(),
                           pl.Trainer(max_epochs=10),
                           train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                           val_dataloaders=pl.DataLoader(test_dataset, batch_size=100))

.. Note:: For NNI to capture the dataset and dataloader and distribute it across different runs, please wrap your dataset with ``blackbox`` and use ``pl.DataLoader`` instead of ``torch.utils.data.DataLoader``. See ``blackbox_module`` section below for details.

Users can refer to `this document <./WriteTrainer.rst>`__ for how to write a new trainer, and refer to `this document <./WriteStrategy.rst>`__ for how to write a new strategy.

Set up an Experiment
^^^^^^^^^^^^^^^^^^^^

After all the above are prepared, it is time to start an experiment to do the model search. We design unified interface for users to start their experiment. An example is shown below

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, applied_mutators, simple_startegy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnasnet_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 10
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)

This code starts an NNI experiment. Note that if inlined mutation is used, ``applied_mutators`` should be ``None``.

The complete code of a simple MNIST example can be found :githublink:`here <test/retiarii_test/mnist/test.py>`.

Visualize your experiment
^^^^^^^^^^^^^^^^^^^^^^^^^

Users can visualize their experiment in the same way as visualizing a normal hyper-parameter tuning experiment. For example, open ``localhost::8081`` in your browser, 8081 is the port that you set in ``exp.run``. Please refer to `here <../../Tutorial/WebUI.rst>`__ for details. If users are using oneshot trainer, they can refer to `here <../Visualization.rst>`__ for how to visualize their experiments.

Export the best model found in your experiment
----------------------------------------------

If you are using *classic search approach*, you can simply find out the best one from WebUI.

If you are using *oneshot (weight-sharing) search approach*, you can invole ``exp.export_top_models`` to output several best models that are found in the experiment.

Advanced and FAQ
----------------

.. _blackbox-module:

**Blackbox Module**

To understand the decorator ``blackbox_module``, we first briefly explain how our framework works: it converts user-defined model to a graph representation (called graph IR), each instantiated module is converted to a subgraph. Then user-defined mutations are applied to the graph to generate new graphs. Each new graph is then converted back to PyTorch code and executed. ``@blackbox_module`` here means the module will not be converted to a subgraph but is converted to a single graph node. That is, the module will not be unfolded anymore. Users should/can decorate a user-defined module class in the following cases:

* When a module class cannot be successfully converted to a subgraph due to some implementation issues. For example, currently our framework does not support adhoc loop, if there is adhoc loop in a module's forward, this class should be decorated as blackbox module. The following ``MyModule`` should be decorated.

  .. code-block:: python

    @blackbox_module
    class MyModule(nn.Module):
      def __init__(self):
        ...
      def forward(self, x):
        for i in range(10): # <- adhoc loop
          ...

* The candidate ops in ``LayerChoice`` should be decorated as blackbox module. For example, ``self.op = nn.LayerChoice([Op1(...), Op2(...), Op3(...)])``, where ``Op1``, ``Op2``, ``Op3`` should be decorated if they are user defined modules.
* When users want to use ``ValueChoice`` in a module's input argument, the module should be decorated as blackbox module. For example, ``self.conv = MyConv(kernel_size=nn.ValueChoice([1, 3, 5]))``, where ``MyConv`` should be decorated.
* If no mutation is targeted on a module, this module *can be* decorated as a blackbox module.