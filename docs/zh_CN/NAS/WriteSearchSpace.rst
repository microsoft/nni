定义搜索空间
====================

通常，搜索空间是要在其中找到最好结构的候选项。 无论是经典 NAS 还是 One-Shot NAS，不同的搜索算法都需要搜索空间。 NNI 提供了统一的 API 来表达神经网络架构的搜索空间。

搜索空间可基于基础模型来构造。 这也是在已有模型上使用 NAS 的常用方法。 `MNIST on PyTorch <https://github.com/pytorch/examples/blob/master/mnist/main.py>`__ 是一个例子。 注意，NNI 为 PyTorch 和 TensorFlow 提供了同样的搜索空间 API。

.. code-block:: python

   from nni.nas.pytorch import mutables

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = mutables.LayerChoice([
               nn.Conv2d(1, 32, 3, 1),
               nn.Conv2d(1, 32, 5, 3)
           ])  # try 3x3 kernel and 5x5 kernel
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.dropout1 = nn.Dropout2d(0.25)
           self.dropout2 = nn.Dropout2d(0.5)
           self.fc1 = nn.Linear(9216, 128)
           self.fc2 = nn.Linear(128, 10)

       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           # ... same as original ...
           return output

以上示例在 conv1 上添加了 conv5x5 的选项。 修改非常简单，只需要声明 ``LayerChoice`` 并将原始的 conv3x3 和新的 conv5x5 作为参数即可。 就这么简单！ 不需要修改 forward 函数。 可将 conv1 想象为没有 NAS 的模型。

如何表示可能的连接？ 通过 ``InputChoice`` 来实现。 要在 MNIST 示例上使用跳过连接，需要增加另一层 conv3。 下面的示例中，从 conv2 的可能连接加入到了 conv3 的输出中。

.. code-block:: python

   from nni.nas.pytorch import mutables

   class Net(nn.Module):
       def __init__(self):
           # ... same ...
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.conv3 = nn.Conv2d(64, 64, 1, 1)
           # declaring that there is exactly one candidate to choose from
           # search strategy will choose one or None
           self.skipcon = mutables.InputChoice(n_candidates=1)
           # ... same ...

       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           x = self.conv2(x)
           x0 = self.skipcon([x])  # choose one or none from [x]
           x = self.conv3(x)
           if x0 is not None:  # skipconnection is open
               x += x0
           x = F.max_pool2d(x, 2)
           # ... same ...
           return output

Input Choice 可被视为可调用的模块，它接收张量数组，输出其中部分的连接、求和、平均（默认为求和），或没有选择时输出 ``None``。 就像 layer choices, input choices 应该 **用** ``__init__`` **来初始化用** ``forward`` **来回调**。 这会让搜索算法找到这些 Choice，并进行所需的准备。

``LayerChoice`` and ``InputChoice`` 都是 **mutables**。 Mutable 表示 "可变化的"。 与传统深度学习层、模型都是固定的不同，使用 Mutable 的模块，是一组可能选择的模型。

用户可以为每一个 mutable 声明一个 key。 默认情况下，NNI 会分配全局唯一的，但如果需要共享 Choice（例如，两个 ``LayerChoice`` 有同样的候选操作，希望共享同样的 Choice。即，如果一个选择了第 i 个操作，第二个也要选择第 i 个操作），那么就应该给它们相同的 key。 key 标记了此 Choice，并会在存储的检查点中使用。 如果要增加导出架构的可读性，可为每个 Mutable 的 key 指派名称。 mutables 的高级用法请参照文档 `Mutables <./NasReference.rst>`__。

定义了搜索空间后，下一步是从中找到最好的模型。 至于如何从定义的搜索空间进行搜索请参阅 `classic NAS algorithms <./ClassicNas.rst>`__ 和 `one-shot NAS algorithms <./NasGuide.rst>`__ 。
