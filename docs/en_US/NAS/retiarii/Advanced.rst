Advanced Tutorial
=================

``@basic_unit`` and ``serializer``
----------------------------------

.. _serializer:

The APIs like ``@basic_unit`` and ``serialize`` are all part of Retiarii as serializers. A serializer is basically needed for the following purposes:

* Prevent graph-parser to parse the module. To understand this, we first briefly explain how our framework works: it converts user-defined model to a graph representation (called graph IR), each instantiated module is converted to a subgraph. Then user-defined mutations are applied to the graph to generate new graphs. Each new graph is then converted back to PyTorch code and executed. ``@basic_unit`` here means the module will not be converted to a subgraph but is converted to a single graph node. That is, the module will not be unfolded anymore. When the subgraph is not unfolded, mutations on parameters of subgraph module becomes easier.
* Function as a translator to translate intermediate arguments (e.g., ValueChoice). A wrapper of module injects the chosen value and replaces the choice with the chosen one.
* Enable the re-instantiation of the object. Sometimes, modules and evaluators needs to be replicated and sent to training services. Retiarii needs to track how to instantiate them.

Thus, serializer should be used in the following cases:

* When a module class cannot be successfully converted to a subgraph due to some implementation issues. For example, currently our framework does not support adhoc loop, if there is adhoc loop in a module's forward, this class should be decorated as serializable module. The following ``MyModule`` should be decorated.

  .. code-block:: python

    @basic_unit
    class MyModule(nn.Module):
      def __init__(self):
        ...
      def forward(self, x):
        for i in range(10): # <- adhoc loop
          ...

* The candidate ops in ``LayerChoice`` should be decorated as serializable module. For example, ``self.op = nn.LayerChoice([Op1(...), Op2(...), Op3(...)])``, where ``Op1``, ``Op2``, ``Op3`` should be decorated if they are user defined modules.
* When users want to use ``ValueChoice`` in a module's input argument, the module should be decorated as serializable module. For example, ``self.conv = MyConv(kernel_size=nn.ValueChoice([1, 3, 5]))``, where ``MyConv`` should be decorated.
* If no mutation is targeted on a module, this module *can be* decorated as a serializable module.


The underlying implementation of Retiarii converts base model into a graph, runs mutators on the converted graph, and converts the mutated graph back to a model that is supported by deep learning frameworks. Therefore, the base model needs to be supported by the converter. For example, in PyTorch, TorchScript needs to be able to recognize and parse this model.

* Deep learning models tend to be a hierachical structure and tends to be very complicated if expanded. For example, a transformer contains several encoders and decoders. An encoder contains a self-attention. An attention layer contains several linear layer. If the only intention is to mutate a parameter on the transformer, there is no need to expand the full graph. To this end, Retiarii considers mutated modules as the most basic building blocks and does not expand them any more. So far, users need to manually annotate them with ``@basic_unit`` For example, user-defined modules used in ``LayerChoice`` should be decorated. Users can refer to `here <#serializer>`__ on the detailed usages.


Express Mutations with Mutators
-------------------------------

Inline mutations: express mutations in an inlined manner

This code starts an NNI experiment. Note that if inlined mutation is used, ``applied_mutators`` should be ``None``.

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

  ph = nn.Placeholder(
    label='mutable_0',
    kernel_size_options=[1, 3, 5],
    n_layer_options=[1, 2, 3, 4],
    exp_ratio=exp_ratio,
    stride=stride
  )

``label`` is used by mutator to identify this placeholder. The other parameters are the information that are required by mutator. They can be accessed from ``node.operation.parameters`` as a dict, it could include any information that users want to put to pass it to user defined mutator. The complete example code can be found in :githublink:`Mnasnet base model <test/retiarii_test/mnasnet/base_mnasnet.py>`.

We introduce two ways to write mutations. These two methods are mutually exclusive, meaning that you cannot use both of them in one model.


