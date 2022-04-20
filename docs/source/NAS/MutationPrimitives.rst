Mutation Primitives
===================

.. TODO: this file will be merged with API reference in future.

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

* Used as input arguments of :class:`nni.retiarii.basic_unit` (i.e., modules in ``nni.retiarii.nn.pytorch`` and user-defined modules decorated with ``@basic_unit``).
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

Value choices supports arithmetic operators, which is particularly useful when searching for a network width multiplier:

..  code-block:: python

    # init
    scale = nn.ValueChoice([1.0, 1.5, 2.0])
    self.conv1 = nn.Conv2d(3, round(scale * 16))
    self.conv2 = nn.Conv2d(round(scale * 16), round(scale * 64))
    self.conv3 = nn.Conv2d(round(scale * 64), round(scale * 256))

    # forward
    return self.conv3(self.conv2(self.conv1(x)))

Or when kernel size and padding are coupled so as to keep the output size constant:

..  code-block:: python

    # init
    ks = nn.ValueChoice([3, 5, 7])
    self.conv = nn.Conv2d(3, 16, kernel_size=ks, padding=(ks - 1) // 2)

    # forward
    return self.conv(x)

Or when several layers are concatenated for a final layer.

..  code-block:: python

    # init
    self.linear1 = nn.Linear(3, nn.ValueChoice([1, 2, 3], label='a'))
    self.linear2 = nn.Linear(3, nn.ValueChoice([4, 5, 6], label='b'))
    self.final = nn.Linear(nn.ValueChoice([1, 2, 3], label='a') + nn.ValueChoice([4, 5, 6], label='b'), 2)

    # forward
    return self.final(torch.cat([self.linear1(x), self.linear2(x)], 1))

Some advanced operators are also provided, such as ``nn.ValueChoice.max`` and ``nn.ValueChoice.cond``. See reference of :class:`nni.retiarii.nn.pytorch.ValueChoice` for more details.

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

  # Can be used together with layer choice
  # With deep copy, the 3 layers will have the same label, thus share the choice
  self.blocks = nn.Repeat(nn.LayerChoice([...]), (1, 3))

  # To make the three layer choices independently
  # Need a factory function that accepts index (0, 1, 2, ...) and returns the module of the `index`-th layer.
  self.blocks = nn.Repeat(lambda index: nn.LayerChoice([...], label=f'layer{index}'), (1, 3))

``nn.Cell``
"""""""""""

API reference: :class:`nni.retiarii.nn.pytorch.Cell`

This cell structure is popularly used in `NAS literature <https://arxiv.org/abs/1611.01578>`__. High-level speaking, literatures often use the following glossaries.

.. list-table::
   :widths: 25 75

   * - Cell
     - A cell consists of several nodes.
   * - Node
     - A node is the **sum** of several operators.
   * - Operator
     - Each operator is independently chosen from a list of user-specified candidate operators.
   * - Operator's input
     - Each operator has one input, chosen from previous nodes as well as predecessors.
   * - Predecessors
     - Input of cell. A cell can have multiple predecessors. Predecessors are sent to *preprocessor* for preprocessing.
   * - Cell's output
     - Output of cell. Usually concatenation of several nodes (possibly all nodes) in the cell. Cell's output, along with predecessors, are sent to *postprocessor* for postprocessing.
   * - Preprocessor
     - Extra preprocessing to predecessors. Usually used in shape alignment (e.g., predecessors have different shapes). By default, do nothing.
   * - Postprocessor
     - Extra postprocessing for cell's output. Usually used to chain cells with multiple Predecessors
       (e.g., the next cell wants to have the outputs of both this cell and previous cell as its input). By default, directly use this cell's output.

Example usages:

.. code-block:: python

  # import nni.retiarii.nn.pytorch as nn
  # used in `__init__` method

  # Choose between conv2d and maxpool2d.
  # The cell have 4 nodes, 1 op per node, and 2 predecessors.
  cell = nn.Cell([nn.Conv2d(32, 32, 3), nn.MaxPool2d(3)], 4, 1, 2)
  # forward
  cell([input1, input2])

  # Use `merge_op` to specify how to construct the output.
  # The output will then have dynamic shape, depending on which input has been used in the cell.
  cell = nn.Cell([nn.Conv2d(32, 32, 3), nn.MaxPool2d(3)], 4, 1, 2, merge_op='loose_end')

  # The op candidates can be callable that accepts node index in cell, op index in node, and input index.
  cell = nn.Cell([
    lambda node_index, op_index, input_index: nn.Conv2d(32, 32, 3, stride=2 if input_index < 1 else 1),
    ...
  ], 4, 1, 2)

  # predecessor example
  class Preprocessor:
    def __init__(self):
      self.conv1 = nn.Conv2d(16, 32, 1)
      self.conv2 = nn.Conv2d(64, 32, 1)

    def forward(self, x):
      return [self.conv1(x[0]), self.conv2(x[1])]

  cell = nn.Cell([nn.Conv2d(32, 32, 3), nn.MaxPool2d(3)], 4, 1, 2, preprocessor=Preprocessor())
  cell([torch.randn(1, 16, 48, 48), torch.randn(1, 64, 48, 48)])  # the two inputs will be sent to conv1 and conv2 respectively
