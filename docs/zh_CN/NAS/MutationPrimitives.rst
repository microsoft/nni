Mutation 原语
===================

为了让用户在他们的 PyTorch/TensorFlow 模型中轻松表达模型空间，NNI 提供了一些内联 mutation API，如下所示。

* `nn.LayerChoice <./ApiReference.rst#nni.retiarii.nn.pytorch.LayerChoice>`__. 它允许用户放置多个候选操作（例如，PyTorch 模块），在每个探索的模型中选择其中一个。

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中声明
    self.layer = nn.LayerChoice([
      ops.PoolBN('max', channels, 3, stride, 1),
      ops.SepConv(channels, channels, 3, stride, 1),
      nn.Identity()
    ]))
    # 在 `forward` 函数中调用
    out = self.layer(x)

* `nn.InputChoice <./ApiReference.rst#nni.retiarii.nn.pytorch.InputChoice>`__. 它主要用于选择（或尝试）不同的连接。 它会从设置的几个张量中，选择 ``n_chosen`` 个张量。

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中声明
    self.input_switch = nn.InputChoice(n_chosen=1)
    # 在 `forward` 函数中调用，三者选一
    out = self.input_switch([tensor1, tensor2, tensor3])

* `nn.ValueChoice <./ApiReference.rst#nni.retiarii.nn.pytorch.ValueChoice>`__. 它用于从一些候选值中选择一个值。 它只能作为基本单元的输入参数，即 ``nni.retiarii.nn.pytorch`` 中的模块和用 ``@basic_unit`` 装饰的用户定义的模块。

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中使用
    self.conv = nn.Conv2d(XX, XX, kernel_size=nn.ValueChoice([1, 3, 5])
    self.op = MyOp(nn.ValueChoice([0, 1]), nn.ValueChoice([-1, 1]))

* `nn.Repeat <./ApiReference.rst#nni.retiarii.nn.pytorch.Repeat>`__. 以可变次数重复某个块

* `nn.Cell <./ApiReference.rst#nni.retiarii.nn.pytorch.Cell>`__. `这种细胞结构在 NAS 文献中被普遍使用 <https://arxiv.org/abs/1611.01578>`__。 具体来说，Cell 由多个 "nodes"（节点）组成。 每个节点是多个运算符的总和。 每个运算符从用户指定的候选者中选择，并从以前的节点和前代（predecessor）中获取一个输入。 前代（Predecessor）指 Cell 的输入。 Cell 的输出是 Cell 中部分节点（目前是所有节点）的串联。