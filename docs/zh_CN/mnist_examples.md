# MNIST 样例

在深度学习中，用 CNN 来分类 MNIST 数据，就像介绍编程语言中的 `hello world` 样例。 因此，NNI 将 MNIST 作为样例来介绍功能。 样例如下：

- [MNIST 中使用 NNI API](#mnist)
- [MNIST 中使用 NNI 标记（annotation）](#mnist-annotation)
- [在 Keras 中使用 MNIST](#mnist-keras)
- [MNIST -- 用批处理调参器来调优](#mnist-batch)
- [MNIST -- 用 hyperband 调优](#mnist-hyperband)
- [MNIST -- 用嵌套搜索空间调优](#mnist-nested)
- [用 Kubeflow 运行分布式的 MNIST (tensorflow)](#mnist-kubeflow-tf)
- [用 Kubeflow 运行分布式的 MNIST (PyTorch)](#mnist-kubeflow-pytorch)

<a name="mnist"></a>
**MNIST 中使用 NNI API**

这是个简单的卷积网络，有两个卷积层，两个池化层和一个全连接层。 调优的超参包括 dropout 比率，卷积层大小，隐藏层（全连接层）大小等等。 它能用 NNI 中大部分内置的调参器来调优，如 TPE，SMAC，Random。 样例的 YAML 文件也启用了评估器来提前终止一些中间结果不好的尝试。

`代码目录: examples/trials/mnist/`

<a name="mnist-annotation"></a>
**MNIST 中使用 NNI 标记（annotation）**

此样例与上例类似，上例使用的是 NNI API 来指定搜索空间并返回结果，而此例使用的是 NNI 标记。

`代码目录: examples/trials/mnist-annotation/`

<a name="mnist-keras"></a>
**在 Keras 中使用 MNIST**

此样例由 Keras 实现。 这也是 MNIST 数据集的网络，包括两个卷积层，一个池化层和两个全连接层。

`代码目录: examples/trials/mnist-keras/`

<a name="mnist-batch"></a>
**MNIST -- 用批处理调参器来调优**

此样例演示了如何使用批处理调参器。 只需要在搜索空间文件中列出所有要尝试的配置， NNI 会逐个尝试。

`代码目录: examples/trials/mnist-batch-tune-keras/`

<a name="mnist-hyperband"></a>
**MNIST -- 用 hyperband 调优**

此样例演示了如何使用 hyperband 来调优模型。 There is one more key `STEPS` in the received configuration for trials to control how long it can run (e.g., number of iterations).

`code directory: examples/trials/mnist-hyperband/`

<a name="mnist-nested"></a>
**MNIST -- tuning within a nested search space**

This example is to show that NNI also support nested search space. The search space file is an example of how to define nested search space.

`code directory: examples/trials/mnist-cascading-search-space/`

<a name="mnist-kubeflow-tf"></a>
**distributed MNIST (tensorflow) using kubeflow**

This example is to show how to run distributed training on kubeflow through NNI. Users can simply provide distributed training code and a configure file which specifies the kubeflow mode. For example, what is the command to run ps and what is the command to run worker, and how many resources they consume. This example is implemented in tensorflow, thus, uses kubeflow tensorflow operator.

`code directory: examples/trials/mnist-distributed/`

<a name="mnist-kubeflow-pytorch"></a>
**distributed MNIST (pytorch) using kubeflow**

Similar to the previous example, the difference is that this example is implemented in pytorch, thus, it uses kubeflow pytorch operator.

`code directory: examples/trials/mnist-distributed-pytorch/`