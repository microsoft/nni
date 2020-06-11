# MNIST 示例

在深度学习中，用 CNN 来分类 MNIST 数据，就像介绍编程语言中的 `hello world` 示例。 因此，NNI 将 MNIST 作为示例来介绍功能。 示例如下：

- [MNIST 中使用 NNI API (TensorFlow v1.x)](#mnist-tfv1)
- [MNIST 中使用 NNI API (TensorFlow v2.x)](#mnist-tfv2)
- [MNIST 中使用 NNI 标记（annotation）](#mnist-annotation)
- [在 Keras 中使用 MNIST](#mnist-keras)
- [MNIST -- 用批处理 Tuner 来调优](#mnist-batch)
- [MNIST -- 用 hyperband 调优](#mnist-hyperband)
- [MNIST -- 用嵌套搜索空间调优](#mnist-nested)
- [用 Kubeflow 运行分布式的 MNIST (tensorflow)](#mnist-kubeflow-tf)
- [用 Kubeflow 运行分布式的 MNIST (PyTorch)](#mnist-kubeflow-pytorch)

<a name="mnist-tfv1"></a>
**MNIST 中使用 NNI API (TensorFlow v1.x)**

这是个简单的卷积网络，有两个卷积层，两个池化层和一个全连接层。 调优的超参包括 dropout 比率，卷积层大小，隐藏层（全连接层）大小等等。 它能用 NNI 中大部分内置的 Tuner 来调优，如 TPE，SMAC，Random。 示例的 YAML 文件也启用了评估器来提前终止一些中间结果不好的尝试。

`代码目录: examples/trials/mnist-tfv1/`

<a name="mnist-tfv2"></a>
**MNIST 中使用 NNI API (TensorFlow v2.x)**

与上述示例的网络相同，但使用了 TensorFlow v2.x Keras API。

`代码目录: examples/trials/mnist-tfv2/`

<a name="mnist-annotation"></a>
**MNIST 中使用 NNI 标记（annotation）**

此示例与上例类似，上例使用的是 NNI API 来指定搜索空间并返回结果，而此例使用的是 NNI 标记。

`代码目录: examples/trials/mnist-annotation/`

<a name="mnist-keras"></a>
**在 Keras 中使用 MNIST**

此示例由 Keras 实现。 这也是 MNIST 数据集的网络，包括两个卷积层，一个池化层和两个全连接层。

`代码目录: examples/trials/mnist-keras/`

<a name="mnist-batch"></a>
**MNIST -- 用批处理 Tuner 来调优**

此示例演示了如何使用批处理 Tuner。 只需要在搜索空间文件中列出所有要尝试的配置， NNI 会逐个尝试。

`代码目录: examples/trials/mnist-batch-tune-keras/`

<a name="mnist-hyperband"></a>
**MNIST -- 用 hyperband 调优**

此示例演示了如何使用 hyperband 来调优模型。 在尝试收到的配置中，有个主键叫做 `STEPS`，尝试要用它来控制运行多长时间（例如，控制迭代的次数）。

`代码目录: examples/trials/mnist-hyperband/`

<a name="mnist-nested"></a>
**MNIST -- 用嵌套搜索空间调优**

此示例演示了 NNI 如何支持嵌套的搜索空间。 搜索空间文件示了如何定义嵌套的搜索空间。

`代码目录: examples/trials/mnist-nested-search-space/`

<a name="mnist-kubeflow-tf"></a>
**用 Kubeflow 运行分布式的 MNIST (tensorflow)**

此示例展示了如何通过 NNI 来在 Kubeflow 上运行分布式训练。 只需要简单的提供分布式训练代码，并在配置文件中指定 kubeflow 模式。 例如，运行 ps 和 worker 的命令行，以及各自需要的资源。 此示例使用了 Tensorflow 来实现，因而，需要使用 Kubeflow 的 tf-operator。

`代码目录: examples/trials/mnist-distributed/`

<a name="mnist-kubeflow-pytorch"></a>
**用 Kubeflow 运行分布式的 MNIST (PyTorch)**

与前面的示例类似，不同之处是此示例是 Pytorch 实现的，因而需要使用 Kubeflow 的 pytorch-operator。

`代码目录: examples/trials/mnist-distributed-pytorch/`