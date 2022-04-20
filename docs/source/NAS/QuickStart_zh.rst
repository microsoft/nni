.. d6ad1b913b292469c647ca68ac158840

快速入门 Retiarii
==============================


.. contents::

在快速入门教程中，我们以 multi-trial NAS 为例来展示如何构建和探索模型空间。 神经网络架构搜索任务主要有三个关键组件，即：

* 模型搜索空间（Model search space），定义了要探索的模型集合。
* 一个适当的策略（strategy），作为探索这个搜索空间的方法。
* 一个模型评估器（model evaluator），报告一个给定模型的性能。

One-shot NAS 教程在 `这里 <./OneshotTrainer.rst>`__。

.. note:: 目前，PyTorch 是 Retiarii 唯一支持的框架，我们只用 **PyTorch 1.7 和 1.10** 进行了测试。 本文档基于 PyTorch 的背景，但它也应该适用于其他框架，这在我们未来的计划中。

定义模型空间
-----------------------

模型空间是由用户定义的，用来表达用户想要探索、认为包含性能良好模型的一组模型。 模型空间是由用户定义的，用来表达用户想要探索、认为包含性能良好模型的一组模型。 在这个框架中，模型空间由两部分组成：基本模型和基本模型上可能的突变。

定义基本模型
^^^^^^^^^^^^^^^^^

定义基本模型与定义 PyTorch（或 TensorFlow）模型几乎相同， 只有两个小区别。 对于 PyTorch 模块（例如 ``nn.Conv2d``, ``nn.ReLU``），将代码 ``import torch.nn as nn`` 替换为 ``import nni.retiarii.nn.pytorch as nn`` 。

下面是定义基本模型的一个简单的示例，它与定义 PyTorch 模型几乎相同。

.. code-block:: python

  import torch
  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn
  from nni.retiarii import model_wrapper

  @model_wrapper      # this decorator should be put on the out most
  class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout(0.25)
      self.dropout2 = nn.Dropout(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(self.conv2(x), 2)
      x = torch.flatten(self.dropout1(x), 1)
      x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
      output = F.log_softmax(x, dim=1)
      return output

.. tip:: 记得使用 ``import nni.retiarii.nn.pytorch as nn`` 和 :meth:`nni.retiarii.model_wrapper`. 许多错误都源于忘记使用它们。同时，对于 ``nn`` 的子模块（例如 ``nn.init``）请使用 ``torch.nn``，比如，``torch.nn.init`` 而不是 ``nn.init``。

定义模型突变
^^^^^^^^^^^^^^^^^^^^^^

基本模型只是一个具体模型，而不是模型空间。 我们为用户提供 `API 和原语 <./MutationPrimitives.rst>`__，用于把基本模型变形成包含多个模型的模型空间。

基于上面定义的基本模型，我们可以这样定义一个模型空间：

.. code-block:: diff

  import torch
  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn
  from nni.retiarii import model_wrapper

  @model_wrapper
  class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
  -   self.conv2 = nn.Conv2d(32, 64, 3, 1)
  +   self.conv2 = nn.LayerChoice([
  +       nn.Conv2d(32, 64, 3, 1),
  +       DepthwiseSeparableConv(32, 64)
  +   ])
  -   self.dropout1 = nn.Dropout(0.25)
  +   self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))
      self.dropout2 = nn.Dropout(0.5)
  -   self.fc1 = nn.Linear(9216, 128)
  -   self.fc2 = nn.Linear(128, 10)
  +   feature = nn.ValueChoice([64, 128, 256])
  +   self.fc1 = nn.Linear(9216, feature)
  +   self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(self.conv2(x), 2)
      x = torch.flatten(self.dropout1(x), 1)
      x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
      output = F.log_softmax(x, dim=1)
      return output

在这个例子中我们使用了两个突变 API， ``nn.LayerChoice`` 和 ``nn.ValueChoice``。 ``nn.LayerChoice`` 的输入参数是一个候选模块的列表（在这个例子中是两个），每个采样到的模型会选择其中的一个，然后它就可以像一般的 PyTorch 模块一样被使用。 ``nn.ValueChoice`` 输入一系列候选的值，然后对于每个采样到的模型，其中的一个值会生效。

更多的 API 描述和用法可以请阅读 `这里 <./construct_space.rst>`__ 。

.. note:: 我们正在积极的丰富突变 API，以简化模型空间的构建。如果我们提供的 API 不能满足您表达模型空间的需求，请阅读 `这个文档 <./Mutators.rst>`__ 以获得更多定制突变的资讯。

探索定义的模型空间
-------------------------------

简单来说，探索模型空间有两种方法：(1) 通过独立评估每个采样模型进行搜索；(2) 基于 One-Shot 的权重共享式搜索。 我们在本教程中演示了下面的第一种方法。 第二种方法可以参考 `这里 <./OneshotTrainer.rst>`__。

首先，用户需要选择合适的探索策略来探索模型空间。然后，用户需要选择或自定义模型评估器来评估每个采样模型的性能。

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

Retiarii 提供了诸多的 `内置模型评估器 <./ModelEvaluators.rst>`__，但是作为第一步，我们还是推荐使用 ``FunctionalEvaluator``，也就是说，将您自己的训练和测试代码用一个函数包起来。这个函数的输入参数是一个模型的类，然后使用 ``nni.report_final_result`` 来汇报模型的效果。

这里的一个例子创建了一个简单的评估器，它在 MNIST 数据集上运行，训练 2 个 Epoch，并报告其在验证集上的准确率。

..  code-block:: python

    def evaluate_model(model_cls):
      # "model_cls" 是一个类，需要初始化
      model = model_cls()

      optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
      transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
      train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
      test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)

      device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

      for epoch in range(3):
        # 训练模型，1 个 epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # 测试模型，1 个 epoch
        accuracy = test_epoch(model, device, test_loader)
        # 汇报中间结果，可以是 float 或者 dict 类型
        nni.report_intermediate_result(accuracy)

      # 汇报最终结果
      nni.report_final_result(accuracy)

    # 创建模型评估器
    evaluator = nni.retiarii.evaluator.FunctionalEvaluator(evaluate_model)

在这里 ``train_epoch`` 和 ``test_epoch`` 可以是任意自定义的函数，用户可以写自己的训练流程。完整的样例可以参见 :githublink:`examples/nas/multi-trial/mnist/search.py`。

我们建议 ``evaluate_model`` 不接受 ``model_cls`` 以外的其他参数。但是，我们在 `高级教程 <./ModelEvaluators.rst>`__ 中展示了其他参数的用法，如果您真的需要的话。另外，我们会在未来支持这些参数的突变（这通常会成为 "超参调优"）。

发起 Experiment
--------------------

一切准备就绪，就可以发起 Experiment 以进行模型搜索了。 样例如下：

.. code-block:: python

  exp = RetiariiExperiment(base_model, evaluator, [], search_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnist_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 20
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)

一个简单 MNIST 示例的完整代码在 :githublink:`这里 <examples/nas/multi-trial/mnist/search.py>`。 除了本地训练平台，用户还可以在除了本地机器以外的 `不同的训练平台 <../training_services.rst>`__ 上运行 Retiarii 的实验。

可视化 Experiment
------------------------

用户可以像可视化普通的超参数调优 Experiment 一样可视化他们的 Experiment。 例如，在浏览器里打开 ``localhost:8081``，8081 是在 ``exp.run`` 里设置的端口。 参考 `这里 <../Tutorial/WebUI.rst>`__ 了解更多细节。

我们支持使用第三方工具（例如 `Netron <https://netron.app/>`__）可视化搜索过程中采样到的模型。您可以点击每个 trial 面板下的 ``Visualization``。注意，目前的可视化是基于导出成 `onnx <https://onnx.ai/>`__ 格式的模型实现的，所以如果模型无法导出成 onnx，那么可视化就无法进行。

内置的模型评估器（比如 Classification）已经自动将模型导出成了一个文件。如果您自定义了模型，您需要将模型导出到 ``$NNI_OUTPUT_DIR/model.onnx``。例如，

.. code-block:: python

  def evaluate_model(model_cls):
    model = model_cls()
    # 把模型导出成 onnx
    if 'NNI_OUTPUT_DIR' in os.environ:
      torch.onnx.export(model, (dummy_input, ),
                        Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')
    # 剩下的就是训练和测试流程

导出最佳模型
-----------------

探索完成后，用户可以使用 ``export_top_models`` 导出最佳模型。

.. code-block:: python

  for model_code in exp.export_top_models(formatter='dict'):
    print(model_code)

导出的 `json` 记录的是最佳模型的突变记录。如果用户想要最佳模型的代码，可以简单的使用基于图的执行引擎，增加如下两行代码即可：

.. code-block:: python

  exp_config.execution_engine = 'base'
  export_formatter = 'code'
