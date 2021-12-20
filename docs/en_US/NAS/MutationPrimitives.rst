Mutation Primitives
===================

To make users easily express a model space within their PyTorch/TensorFlow model, NNI provides some inline mutation APIs as shown below.

We show the most common use case here. For advanced usages, please see `reference <./ApiReference.rst>`__.

.. note:: We can actively adding more mutation primitives. If you have any suggestions, feel free to `ask here <https://github.com/microsoft/nni/issues>`__.

``nn.LayerChoice``
""""""""""""""""""

API reference: :class:`nni.retiarii.nn.pytorch.LayerChoice`

It allows users to put several candidate operations (e.g., PyTorch modules), one of them is chosen in each explored model.

..  code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # declared in `__init__` method
    self.layer = nn.LayerChoice([
      ops.PoolBN('max', channels, 3, stride, 1),
      ops.SepConv(channels, channels, 3, stride, 1),
      nn.Identity()
    ])
    # invoked in `forward` method
    out = self.layer(x)

``nn.InputChoice``
""""""""""""""""""

API reference: :class:`nni.retiarii.nn.pytorch.InputChoice`

It is mainly for choosing (or trying) different connections. It takes several tensors and chooses ``n_chosen`` tensors from them.

..  code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # declared in `__init__` method
    self.input_switch = nn.InputChoice(n_chosen=1)
    # invoked in `forward` method, choose one from the three
    out = self.input_switch([tensor1, tensor2, tensor3])

``nn.ValueChoice``
""""""""""""""""""

API reference: :class:`nni.retiarii.nn.pytorch.ValueChoice`

It is for choosing one value from some candidate values. The most common use cases are:

* Used as input arguments of `basic units <LINK_TBD>` (i.e., modules in ``nni.retiarii.nn.pytorch`` and user-defined modules decorated with ``@basic_unit``).
* Used as input arguments of evaluator (*new in v2.7*).

Examples are as follows:

..  code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # used in `__init__` method
    self.conv = nn.Conv2d(XX, XX, kernel_size=nn.ValueChoice([1, 3, 5]))
    self.op = MyOp(nn.ValueChoice([0, 1]), nn.ValueChoice([-1, 1]))

    # used in evaluator
    def train_and_evaluate(model_cls, learning_rate):
      ...

    self.evaluator = FunctionalEvaluator(train_and_evaluate, learning_rate=nn.ValueChoice([1e-3, 1e-2, 1e-1]))

.. tip::

  All the APIs have an optional argument called ``label``, mutations with the same label will share the same choice. A typical example is,

  .. code-block:: python

      self.net = nn.Sequential(
        nn.Linear(10, nn.ValueChoice([32, 64, 128], label='hidden_dim')),
        nn.Linear(nn.ValueChoice([32, 64, 128], label='hidden_dim'), 3)
      )

.. warning::

    It looks as if a specific candidate has been chosen (e.g., the way you can put ``ValueChoice`` as a parameter of ``nn.ValueChoice``), but in fact it's a syntax sugar as because the basic units and evaluators do all the underlying works. That means, you cannot assume that ``ValueChoice`` can be used in the same way as its candidates. For example, the following usage will NOT work:

    .. code-block:: python

      self.blocks = []
      for i in range(nn.ValueChoice([1, 2, 3])):
        self.blocks.append(Block())

      # NOTE: instead you should probably write
      # self.blocks = nn.Repeat(Block(), (1, 3))

``nn.Repeat``
"""""""""""""

API reference: :class:`nni.retiarii.nn.pytorch.Repeat`

Repeat a block by a variable number of times.

.. code-block:: python

  # import nni.retiarii.nn.pytorch as nn
  # used in `__init__` method

  # Block() will be deep copied and repeated 3 times
  self.blocks = nn.Repeat(Block(), 3)

  # Block() will be repeated 1, 2, or 3 times
  self.blocks = nn.Repeat(Block(), (1, 3))

  # FIXME
  # The following use cases have known issues and will be fixed in current release

  # Can be used together with layer choice
  # With deep copy, the 3 layers will have the same label, thus share the choice
  self.blocks = nn.Repeat(nn.LayerChoice([...]), (1, 3))

  # To make the three layer choices independently
  # Need a factory function that accepts index (0, 1, 2, ...) and returns the module of the `index`-th layer.
  self.blocks = nn.Repeat(lambda index: nn.LayerChoice([...], label=f'layer{index}'), (1, 3))

``nn.Cell``
"""""""""""

API reference: :class:`nni.retiarii.nn.pytorch.Cell`

This cell structure is popularly used in `NAS literature <https://arxiv.org/abs/1611.01578>`__. Specifically, the cell consists of multiple "nodes". Each node is a sum of multiple operators. Each operator is chosen from user specified candidates, and takes one input from previous nodes and predecessors. Predecessor means the input of cell. The output of cell is the concatenation of some of the nodes in the cell (currently all the nodes).
