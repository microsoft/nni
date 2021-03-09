Advanced Tutorial
=================

This document includes two parts. The first part explains the design decision of ``@basic_unit`` and ``serializer``. The second part is the tutorial of how to write a model space with mutators.

``@basic_unit`` and ``serializer``
----------------------------------

.. _serializer:

``@basic_unit`` and ``serialize`` can be viewed as some kind of serializer. They are designed for making the whole model (including training) serializable to be executed on another process or machine.

**@basic_unit** annotates that a module is a basic unit, i.e, no need to understand the details of this module. The effect is that it prevents Retiarii to parse this module. To understand this, we first briefly explain how Retiarii works: it converts user-defined model to a graph representation (called graph IR) using `TorchScript <https://pytorch.org/docs/stable/jit.html>`__, each instantiated module in the model is converted to a subgraph. Then mutations are applied to the graph to generate new graphs. Each new graph is then converted back to PyTorch code and executed. ``@basic_unit`` here means the module will not be converted to a subgraph, instead, it is converted to a single graph node as a basic unit. That is, the module will not be unfolded anymore. When the module is not unfolded, mutations on initialization parameters of this module becomes easier.

``@basic_unit`` is usually used in the following cases:

* When users want to tune initialization parameters of a module using ``ValueChoice``, then decorate the module with ``@basic_unit``. For example, ``self.conv = MyConv(kernel_size=nn.ValueChoice([1, 3, 5]))``, here ``MyConv`` should be decorated.

* When a module cannot be successfully parsed to a subgraph, decorate the module with ``@basic_unit``. The parse failure could be due to complex control flow. Currently Retiarii does not support adhoc loop, if there is adhoc loop in a module's forward, this class should be decorated as serializable module. For example, the following ``MyModule`` should be decorated.

  .. code-block:: python

    @basic_unit
    class MyModule(nn.Module):
      def __init__(self):
        ...
      def forward(self, x):
        for i in range(10): # <- adhoc loop
          ...

* Some inline mutation APIs require their handled module to be decorated with ``@basic_unit``. For example, user-defined module that is provided to ``LayerChoice`` as a candidate op should be decorated.

**serialize** is mainly used for serializing model training logic. It enables re-instantiation of model evaluator in another process or machine. Re-instantiation is necessary because most of time model and evaluator should be sent to training services. ``serialize`` is implemented by recording the initialization parameters of user instantiated evaluator.

The evaluator related APIs provided by Retiarii have already supported serialization, for example ``pl.Classification``, ``pl.DataLoader``, no need to apply ``serialize`` on them. In the following case users should use ``serialize`` API manually.

If the initialization parameters of the evaluator APIs (e.g., ``pl.Classification``, ``pl.DataLoader``) are not primitive types (e.g., ``int``, ``string``), they should be applied with  ``serialize``. If those parameters' initialization parameters are not primitive types, ``serialize`` should also be applied. In a word, ``serialize`` should be applied recursively if necessary.


Express Mutations with Mutators
-------------------------------

Besides inline mutations which have been demonstrated `here <./Tutorial.rst>`__, Retiarii provides a more general approach to express a model space: *Mutator*. Inline mutations APIs are also implemented with mutator, which can be seen as a special case of model mutation.

.. note:: Mutator and inline mutation APIs cannot be used together.

A mutator is a piece of logic to express how to mutate a given model. Users are free to write their own mutators. Then a model space is expressed with a base model and a list of mutators. A model in the model space is sampled by applying the mutators on the base model one after another. An example is shown below.

.. code-block:: python

  applied_mutators = []
  applied_mutators.append(BlockMutator('mutable_0'))
  applied_mutators.append(BlockMutator('mutable_1'))

``BlockMutator`` is defined by users to express how to mutate the base model. 

Write a mutator
^^^^^^^^^^^^^^^

User-defined mutator should inherit ``Mutator`` class, and implement mutation logic in the member function ``mutate``.

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

The input of ``mutate`` is graph IR (Intermediate Representation) of the base model (please refer to `here <./ApiReference.rst>`__ for the format and APIs of the IR), users can mutate the graph using the graph's member functions (e.g., ``get_nodes_by_label``, ``update_operation``). The mutation operations can be combined with the API ``self.choice``, in order to express a set of possible mutations. In the above example, the node's operation can be changed to any operation from ``candidate_op_list``.

Use placehoder to make mutation easier: ``nn.Placeholder``. If you want to mutate a subgraph or node of your model, you can define a placeholder in this model to represent the subgraph or node. Then, use mutator to mutate this placeholder to make it real modules.

.. code-block:: python

  ph = nn.Placeholder(
    label='mutable_0',
    kernel_size_options=[1, 3, 5],
    n_layer_options=[1, 2, 3, 4],
    exp_ratio=exp_ratio,
    stride=stride
  )

``label`` is used by mutator to identify this placeholder. The other parameters are the information that are required by mutator. They can be accessed from ``node.operation.parameters`` as a dict, it could include any information that users want to put to pass it to user defined mutator. The complete example code can be found in :githublink:`Mnasnet base model <test/retiarii_test/mnasnet/base_mnasnet.py>`.

Starting an experiment is almost the same as using inline mutation APIs. The only difference is that the applied mutators should be passed to ``RetiariiExperiment``. Below is a simple example.

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, applied_mutators, simple_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnasnet_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 10
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)
