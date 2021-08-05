Express Mutations with Mutators
===============================

Besides the inline mutation APIs demonstrated `here <./MutationPrimitives.rst>`__, NNI provides a more general approach to express a model space, i.e., *Mutator*, to cover more complex model spaces. Those inline mutation APIs are also implemented with mutator in the underlying system, which can be seen as a special case of model mutation.

.. note:: Mutator and inline mutation APIs cannot be used together.

A mutator is a piece of logic to express how to mutate a given model. Users are free to write their own mutators. Then a model space is expressed with a base model and a list of mutators. A model in the model space is sampled by applying the mutators on the base model one after another. An example is shown below.

.. code-block:: python

  applied_mutators = []
  applied_mutators.append(BlockMutator('mutable_0'))
  applied_mutators.append(BlockMutator('mutable_1'))

``BlockMutator`` is defined by users to express how to mutate the base model. 

Write a mutator
---------------

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

``label`` is used by mutator to identify this placeholder. The other parameters are the information that is required by mutator. They can be accessed from ``node.operation.parameters`` as a dict, it could include any information that users want to put to pass it to user defined mutator. The complete example code can be found in :githublink:`Mnasnet base model <examples/nas/multi-trial/mnasnet/base_mnasnet.py>`.

Starting an experiment is almost the same as using inline mutation APIs. The only difference is that the applied mutators should be passed to ``RetiariiExperiment``. Below is a simple example.

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, applied_mutators, simple_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnasnet_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 10
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)
