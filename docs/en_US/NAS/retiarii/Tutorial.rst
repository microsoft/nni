Neural Architecture Search with Retiarii (Experimental)
=======================================================

`Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ is a new framework to support neural architecture search and hyper-parameter tuning. It allows users to express various search space with high flexibility, to reuse many SOTA search algorithms, and to leverage system level optimizations to speed up the search process. This framework provides the following new user experiences.

* Search space can be expressed directly in user model code. A tuning space can be expressed along programming a model.
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

* Use our wrapped ``nn`` for PyTorch modules instead of ``torch.nn``. Specifically, users can simply replace the code ``import torch.nn as nn`` with ``import nni.retiarii.nn.pytorch as nn``
* Add the decorator ``@blackbox_module`` to some module classes. Below we explain why this decorator is needed and what module classes should be decorated.

**@blackbox_module**: Our framework works as follows: it converts user defined model to a graph representation (called graph IR), each instantiated module is converted to a subgraph. Then user defined mutations are applied to the graph to generate new graphs, each new graph is then converted back to PyTorch code and executed. ``@blackbox_module`` here means the module will not be converted to a subgraph but is converted to a single graph node. That is, the module will not be unfolded anymore. Users should/can decorate a module class in the following cases:

* When a module class cannot be successfully converted to a subgraph. Currently, our framework does not support adhoc loop, if there is adhoc loop in a module's forward, this class should be decorated as blackbox module.
* The candidate ops in ``LayerChoice`` should be decorated as blackbox module.
* When users want to use ``ValueChoice`` in a module's input argument, the module should be decorated as blackbox module.
* If no mutation is targeted on a module, this module *can be* decorated as a blackbox module.

Below is a simple example code of how to define a base model.

.. code-block:: python

  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn
  from nni.retiarii import register_module

  @register_module()
  class MyModule(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv = nn.Conv2d(32, 1, 5)
      self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
      return self.pool(self.conv(x))

  @register_module()
  class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.mymodule = MyModule()
    def forward(self, x):
      return F.relu(self.mymodule(x))

Users can refer to :githublink:`Darts base model <test/retiarii_test/darts/darts_model.py>` and :githublink:`Mnasnet base model <test/retiarii_test/mnasnet/base_mnasnet.py>` for more complete examples.

Define Model Mutations
^^^^^^^^^^^^^^^^^^^^^^

A base model is only one concrete model not a model space. To define model space, we provide APIs and primitives for users to express how the base model can be mutated.

**Express mutations in an inlined manner**

For easy usability and also backward compatibility, we provide some APIs for users to easily express possible mutations during defining a base model. The APIs can be used just like PyTorch module.

* ``nn.LayerChoice``. It allows users to put several candidate operations (e.g., PyTorch modules), one is chosen in each explored model.

.. code-block:: python

  # declared in `__init__`
  self.layer = nn.LayerChoice([
    ops.PoolBN('max', channels, 3, stride, 1),
    ops.SepConv(channels, channels, 3, stride, 1),
    nn.Identity()
  ]))
  # invoked in `forward` function
  out = self.layer(x)

* ``nn.InputChoice``. It is mainly for choosing (trying) different connections. It takes several tensors and chooses one.

.. code-block:: python

  # declared in `__init__`
  self.input_switch = nn.InputChoice(n_chosen=1)
  # invoked in `forward` function, choose one from the three
  out = self.input_switch([tensor1, tensor2, tensor3])

* ``nn.ValueChoice``. Will be supported soon.

Detailed API description can be found `here <>`__\. Example of using these APIs can be found in :githublink:`Darts base model <test/retiarii_test/darts/darts_model.py>`.

**Express mutations with mutators**

Inline mutations have limited expressiveness, as it has to be embedded in model definition. Therefore, we provide primitives for users to write *Mutator* to flexibly express how they want to mutate base model. Mutator stands above base model, thus has full ability to edit the model.

Users can instantiate several mutators as below, the mutators will be sequentially applied to the base model one after another to generate a new model during experiment running.

.. code-block:: python

  applied_mutators = []
  applied_mutators.append(BlockMutator('mutable_0'))
  applied_mutators.append(BlockMutator('mutable_1'))

``BlockMutator`` could be defined by users to express how to mutate the base model. User defined mutator should inherit ``Mutator`` class, and implement mutation logic in member function ``mutate``.

.. code-block:: python

  class BlockMutator(Mutator):
    def __init__(self, target: str):
        super(BlockMutator, self).__init__()
        self.target = target

    def mutate(self, model):
      nodes = model.get_nodes_by_label(self.target)
      for node in nodes:
        chosen_op = self.choice(candidate_op_list)
        node.update_operation(chosen_op.type, chosen_op.params)

The input of ``mutate`` is a model IR (please refer to `here <>`__ for the format and APIs of the IR), users can mutate the model with its member functions (e.g., ``get_nodes_by_label``, ``update_operation``). The mutation operations can be combined with the API ``self.choice``, in order to express a set of mutations. In the above example, the node's operation can be changed to each operation from ``candidate_op_list``.

For mutator to easily target on a node (i.e., PyTorch module), we provide a placeholder module called ``nn.Placeholder``. If you want to mutate a module, you can define this module with ``nn.Placeholder``, and use mutator to mutate this placeholder to give it real computation operation.

.. code-block:: python

  ph = nn.Placeholder(label=f'mutable_{count}',
    related_info={
      'kernel_size_options': [1, 3, 5],
      'n_layer_options': [1, 2, 3, 4],
      'exp_ratio': exp_ratio,
      'stride': stride
    }
  )

``label`` is used by mutator to identify this placeholder, ``related_info`` is included in this placeholder node for mutator to get more mutation related information. A complete example code can be found in :githublink:`Mnasnet base model <test/retiarii_test/mnasnet/base_mnasnet.py>`.

Explore the Defined Model Space
-------------------------------

After model space is defined, it is time to explore this model space efficiently. Users can choose proper search and training approach to explore the model space.

Create a Trainer and Exploration Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Classic search approach:**
In this approach, trainer is for training each explored model, while strategy is for sampling the models. Both trainer and strategy are required to explore the model space.

**Weight-sharing search approach:**
In this approach, users only need a weight-sharing trainer, because this trainer takes charge of both search and training.

In the following table, we listed the available trainers and strategies.

TODO: table here.

Users can write their own trainer and strategy, please refer to `here <>`__ for tutorial.

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
  exp.run(exp_config, 8081, debug=True)

This code starts an NNI experiment. Note that if inlined mutation is used, ``applied_mutators`` should be ``None``.

FAQ
---

TBD
