用 Mutators 表示 Mutations
===============================

除了在 `这里 <./MutationPrimitives.rst>`__ 演示的内联突变 API，NNI 还提供了一种更通用的方法来表达模型空间，即 *突变器（Mutator）*，以涵盖更复杂的模型空间。 那些内联突变 API在底层系统中也是用突变器实现的，这可以看作是模型突变的一个特殊情况。

.. note:: Mutator 和内联突变 API 不能一起使用。

突变器是一段逻辑，用来表达如何突变一个给定的模型。 用户可以自由地编写自己的突变器。 然后用一个基础模型和一个突变器列表来表达一个模型空间。 通过在基础模型上接连应用突变器，来对模型空间中的一个模型进行采样。 示例如下：

.. code-block:: python

  applied_mutators = []
  applied_mutators.append(BlockMutator('mutable_0'))
  applied_mutators.append(BlockMutator('mutable_1'))

``BlockMutator`` 由用户定义，表示如何对基本模型进行突变。 

编写 mutator
---------------

用户定义的 Mutator 应该继承 ``Mutator`` 类，并在成员函数 ``mutate`` 中实现突变逻辑。

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

``mutate`` 的输入是基本模型的 graph IR（请参考 `这里 <./ApiReference.rst>`__ 获取 IR 的格式和 API），用户可以使用其成员函数（例如， ``get_nodes_by_label``，``update_operation``）对图进行变异。 变异操作可以与 API ``self.choice`` 相结合，以表示一组可能的突变。 在上面的示例中，节点的操作可以更改为 ``candidate_op_list`` 中的任何操作。

使用占位符使突变更容易：``nn.Placeholder``。 如果要更改模型的子图或节点，可以在此模型中定义一个占位符来表示子图或节点。 然后，使用 Mutator 对这个占位符进行变异，使其成为真正的模块。

.. code-block:: python

  ph = nn.Placeholder(
    label='mutable_0',
    kernel_size_options=[1, 3, 5],
    n_layer_options=[1, 2, 3, 4],
    exp_ratio=exp_ratio,
    stride=stride
  )

``label`` 被 Mutator 所使用，来识别此占位符。 其他参数是突变器需要的信息。 它们可以从 ``node.operations.parameters`` 作为一个 dict 被访问，包括任何用户想传递给自定义突变器的信息。 完整的示例代码可以在 :githublink:`Mnasnet base model <examples/nas/multi-trial/mnasnet/base_mnasnet.py>` 找到。

开始一个实验与使用内联突变 API 几乎是一样的。 唯一的区别是，应用的突变器应该被传递给 ``RetiariiExperiment``。 示例如下：

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, applied_mutators, simple_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnasnet_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 10
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)
