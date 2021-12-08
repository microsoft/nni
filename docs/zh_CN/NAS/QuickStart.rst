.. 83ce1769eb03248c40c61ccae8afe4cd

快速入门 Retiarii
==============================


.. contents::

在快速入门教程中，我们以 multi-trial NAS 为例来展示如何构建和探索模型空间。 神经网络架构搜索任务主要有三个关键组件，即：

* 模型搜索空间（Model search space），定义了要探索的模型集合。
* 一个适当的策略（strategy），作为探索这个搜索空间的方法。
* 一个模型评估器（model evaluator），报告一个给定模型的性能。

One-shot NAS 教程在 `这里 <./OneshotTrainer.rst>`__。

.. note:: 目前，PyTorch 是 Retiarii 唯一支持的框架，我们只用 **PyTorch 1.6 和 1.7** 进行了测试。 本文档基于 PyTorch 的背景，但它也应该适用于其他框架，这在我们未来的计划中。

定义模型空间
-----------------------

模型空间是由用户定义的，用来表达用户想要探索、认为包含性能良好模型的一组模型。 模型空间是由用户定义的，用来表达用户想要探索、认为包含性能良好模型的一组模型。 在这个框架中，模型空间由两部分组成：基本模型和基本模型上可能的突变。

定义基本模型
^^^^^^^^^^^^^^^^^

定义基本模型与定义 PyTorch（或 TensorFlow）模型几乎相同， 只有两个小区别。 对于 PyTorch 模块（例如 ``nn.Conv2d``, ``nn.ReLU``），将代码 ``import torch.nn as nn`` 替换为 ``import nni.retiarii.nn.pytorch as nn`` 。

下面是定义基本模型的一个简单的示例，它与定义 PyTorch 模型几乎相同。

.. code-block:: python

  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn
  from nni.retiarii import model_wrapper

  class BasicBlock(nn.Module):
    def __init__(self, const):
      self.const = const
    def forward(self, x):
      return x + self.const

  class ConvPool(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv = nn.Conv2d(32, 1, 5)  # possibly mutate this conv
      self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
      return self.pool(self.conv(x))

  @model_wrapper      # 这个装饰器应该放在最外面的 PyTorch 模块上
  class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.convpool = ConvPool()
      self.mymodule = BasicBlock(2.)
    def forward(self, x):
      return F.relu(self.convpool(self.mymodule(x)))

定义模型突变
^^^^^^^^^^^^^^^^^^^^^^

基本模型只是一个具体模型，而不是模型空间。 我们为用户提供 API 和原语，用于把基本模型变形成包含多个模型的模型空间。

用户可以按以下方式实例化多个 Mutator，这些 Mutator 将依次依次应用于基本模型来对新模型进行采样。 API 可以像 PyTorch 模块一样使用。 这种方法也被称为内联突变。

* ``nn.LayerChoice``， 它允许用户放置多个候选操作（例如，PyTorch 模块），在每个探索的模型中选择其中一个。

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

* ``nn.InputChoice``， 它主要用于选择（或尝试）不同的连接。 它会从设置的几个张量中，选择 ``n_chosen`` 个张量。

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中声明
    self.input_switch = nn.InputChoice(n_chosen=1)
    # 在 `forward` 函数中调用，三者选一
    out = self.input_switch([tensor1, tensor2, tensor3])

* ``nn.ValueChoice``， 它用于从一些候选值中选择一个值。 它只能作为基本单元的输入参数，即 ``nni.retiarii.nn.pytorch`` 中的模块和用 ``@basic_unit`` 装饰的用户定义的模块。

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中声明
    self.conv = nn.Conv2d(XX, XX, kernel_size=nn.ValueChoice([1, 3, 5])
    self.op = MyOp(nn.ValueChoice([0, 1], nn.ValueChoice([-1, 1]))

所有的API都有一个可选的参数，叫做 ``label``，具有相同标签的突变将共享相同的选择。 一个典型示例：

  .. code-block:: python

    self.net = nn.Sequential(
        nn.Linear(10, nn.ValueChoice([32, 64, 128], label='hidden_dim'),
        nn.Linear(nn.ValueChoice([32, 64, 128], label='hidden_dim'), 3)
    )

使用说明和 API 文档在 `这里 <./ApiReference>`__。 详细的 API 描述和使用说明在 `这里 <./ApiReference.rst>`__。 使用这些 API 的示例在 :githublink:`Darts base model <test/retiarii_test/darts/darts_model.py>`。 我们正在积极丰富内联突变 API，使其更容易表达一个新的搜索空间。 参考 `这里 <./construct_space.rst>`__ 获取更多关于表达复杂模型空间的教程。

探索定义的模型空间
-------------------------------

基本上有两种探索方法：(1)通过独立评估每个采样模型进行搜索；(2)基于 One-Shot 的权重共享式搜索。 我们在本教程中演示了下面的第一种方法。 第二种方法可以参考 `这里 <./OneshotTrainer.rst>`__。

用户可以选择合适的探索策略来探索模型空间，并选择或自定义模型评估器来评估每个采样模型的性能。

选择搜索策略
^^^^^^^^^^^^^^^^^^^^^^^^

Retiarii 支持许多 `探索策略（exploration strategies） <./ExplorationStrategies.rst>`__。

简单地选择（即实例化）一个探索策略：

.. code-block:: python

  import nni.retiarii.strategy as strategy

  search_strategy = strategy.Random(dedup=True)  # dedup=False 如果不希望有重复数据删除

选择或编写模型评估器
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 NAS 过程中，探索策略反复生成新模型。 模型评估器用于训练和验证每个生成的模型。 生成的模型所获得的性能被收集起来，并送至探索策略以生成更好的模型。

在 PyTorch 的上下文中，Retiarii 提供了两个内置模型评估器，为简单用例而设计：分类和回归。 这两个评估器是建立在强大的库 PyTorch-Lightning 之上。

这里的一个例子创建了一个简单的评估器，它在 MNIST 数据集上运行，训练 10 个 Epoch，并报告其验证准确性。

.. code-block:: python

  import nni.retiarii.evaluator.pytorch.lightning as pl
  from nni.retiarii import serialize
  from torchvision import transforms

  transform = serialize(transforms.Compose, [serialize(transforms.ToTensor()), serialize(transforms.Normalize, (0.1307,), (0.3081,))])
  train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
  test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)
  evaluator = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=10)

由于模型评估器是在另一个进程中运行的（可能是在一些远程机器中），定义的评估器以及它的所有参数都需要被正确序列化。 例如，用户应该使用已经被包装为在 ``nni.retiarii.evaluator.pytorch.lightning`` 中的可序列化类的 dataloader。 对于 dataloader 中使用的参数，需要进行递归序列化，直到参数为 int、str、float 等简单类型。

模型评价器的详细描述和使用方法可以在 `这里 <./ApiReference.rst>`__ 找到。

如果内置的模型评估器不符合您的要求，或者您已经编写了训练代码只是想使用它，您可以参考 `编写新模型评估器的指南 <./WriteTrainer.rst>`__ 。

.. note:: 如果您想在本地运行模型评估器以进行调试，您可以通过 ``evaluator._execute(Net)`` 直接运行评估器（注意它必须是 ``Net``，而不是 ``Net()``）。 但是，此 API 目前是内部的，可能会发生变化。

.. warning:: 目前不支持模型评估器参数的突变（也就是超参数调整），但将未来会支持。

.. warning:: 要在 Retiarii中 使用 PyTorch-lightning，目前你需要安装 PyTorch-lightning v1.1.x（不支持 v1.2）。

发起 Experiment
--------------------

上述内容准备就绪之后，就可以发起 Experiment 以进行模型搜索了。 样例如下：

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, None, simple_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnasnet_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 10
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)

一个简单 MNIST 示例的完整代码在 :githublink:`这里 <test/retiarii_test/mnist/test.py>`。 除了本地训练平台，用户还可以在 `不同的训练平台 <../training_services.rst>`__ 上运行 Retiarii 的实验。

可视化 Experiment
------------------------

用户可以像可视化普通的超参数调优 Experiment 一样可视化他们的 Experiment。 例如，在浏览器里打开 ``localhost::8081``，8081 是在 ``exp.run`` 里设置的端口。 参考 `这里 <../Tutorial/WebUI.rst>`__ 了解更多细节。

导出最佳模型
-----------------

探索完成后，用户可以使用 ``export_top_models`` 导出最佳模型。

.. code-block:: python

  for model_code in exp.export_top_models(formatter='dict'):
    print(model_code)
