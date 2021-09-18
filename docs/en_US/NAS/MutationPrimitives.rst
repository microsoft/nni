Mutation Primitives
===================

To make users easily express a model space within their PyTorch/TensorFlow model, NNI provides some inline mutation APIs as shown below.

* `nn.LayerChoice <./ApiReference.rst#nni.retiarii.nn.pytorch.LayerChoice>`__. It allows users to put several candidate operations (e.g., PyTorch modules), one of them is chosen in each explored model.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # declared in `__init__` method
    self.layer = nn.LayerChoice([
      ops.PoolBN('max', channels, 3, stride, 1),
      ops.SepConv(channels, channels, 3, stride, 1),
      nn.Identity()
    ])
    # invoked in `forward` method
    out = self.layer(x)

* `nn.InputChoice <./ApiReference.rst#nni.retiarii.nn.pytorch.InputChoice>`__. It is mainly for choosing (or trying) different connections. It takes several tensors and chooses ``n_chosen`` tensors from them.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # declared in `__init__` method
    self.input_switch = nn.InputChoice(n_chosen=1)
    # invoked in `forward` method, choose one from the three
    out = self.input_switch([tensor1, tensor2, tensor3])

* `nn.ValueChoice <./ApiReference.rst#nni.retiarii.nn.pytorch.ValueChoice>`__. It is for choosing one value from some candidate values. It can only be used as input argument of basic units, that is, modules in ``nni.retiarii.nn.pytorch`` and user-defined modules decorated with ``@basic_unit``.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # used in `__init__` method
    self.conv = nn.Conv2d(XX, XX, kernel_size=nn.ValueChoice([1, 3, 5])
    self.op = MyOp(nn.ValueChoice([0, 1]), nn.ValueChoice([-1, 1]))

* `nn.Repeat <./ApiReference.rst#nni.retiarii.nn.pytorch.Repeat>`__. Repeat a block by a variable number of times.

* `nn.Cell <./ApiReference.rst#nni.retiarii.nn.pytorch.Cell>`__. `This cell structure is popularly used in NAS literature <https://arxiv.org/abs/1611.01578>`__. Specifically, the cell consists of multiple "nodes". Each node is a sum of multiple operators. Each operator is chosen from user specified candidates, and takes one input from previous nodes and predecessors. Predecessor means the input of cell. The output of cell is the concatenation of some of the nodes in the cell (currently all the nodes).


All the APIs have an optional argument called ``label``, mutations with the same label will share the same choice. A typical example is,

  .. code-block:: python

    self.net = nn.Sequential(
        nn.Linear(10, nn.ValueChoice([32, 64, 128], label='hidden_dim'),
        nn.Linear(nn.ValueChoice([32, 64, 128], label='hidden_dim'), 3)
    )
