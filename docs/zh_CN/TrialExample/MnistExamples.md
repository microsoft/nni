# MNIST 样例

在深度学习中，用 CNN 来分类 MNIST 数据，就像介绍编程语言中的 `hello world` 样例。 因此，NNI 将 MNIST 作为样例来介绍功能。 样例如下：

- [MNIST with NNI API (TensorFlow v1.x)](#mnist-tfv1)
- [MNIST with NNI API (TensorFlow v2.x)](#mnist-tfv2)
- [MNIST with NNI annotation](#mnist-annotation)
- [MNIST in keras](#mnist-keras)
- [MNIST -- tuning with batch tuner](#mnist-batch)
- [MNIST -- tuning with hyperband](#mnist-hyperband)
- [MNIST -- tuning within a nested search space](#mnist-nested)
- [distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)
- [distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)

<a name="mnist-tfv1"></a>
**MNIST with NNI API (TensorFlow v1.x)**

这是个简单的卷积网络，有两个卷积层，两个池化层和一个全连接层。 调优的超参包括 dropout 比率，卷积层大小，隐藏层（全连接层）大小等等。 它能用 NNI 中大部分内置的 Tuner 来调优，如 TPE，SMAC，Random。 样例的 YAML 文件也启用了评估器来提前终止一些中间结果不好的尝试。

`code directory: examples/trials/mnist-tfv1/`

<a name="mnist-tfv2"></a>
**MNIST with NNI API (TensorFlow v2.x)**

Same network to the example above, but written in TensorFlow v2.x Keras API.

`code directory: examples/trials/mnist-tfv2/`

<a name="mnist-annotation"></a>
**MNIST with NNI annotation**

This example is similar to the example above, the only difference is that this example uses NNI annotation to specify search space and report results, while the example above uses NNI apis to receive configuration and report results.

`code directory: examples/trials/mnist-annotation/`

<a name="mnist-keras"></a>
**MNIST in keras**

This example is implemented in keras. It is also a network for MNIST dataset, with two convolution layers, one pooling layer, and two fully connected layers.

`code directory: examples/trials/mnist-keras/`

<a name="mnist-batch"></a>
**MNIST -- tuning with batch tuner**

This example is to show how to use batch tuner. Users simply list all the configurations they want to try in the search space file. NNI will try all of them.

`code directory: examples/trials/mnist-batch-tune-keras/`

<a name="mnist-hyperband"></a>
**MNIST -- tuning with hyperband**

This example is to show how to use hyperband to tune the model. There is one more key `STEPS` in the received configuration for trials to control how long it can run (e.g., number of iterations).

`code directory: examples/trials/mnist-hyperband/`

<a name="mnist-nested"></a>
**MNIST -- tuning within a nested search space**

This example is to show that NNI also support nested search space. The search space file is an example of how to define nested search space.

`code directory: examples/trials/mnist-nested-search-space/`

<a name="mnist-kubeflow-tf"></a>
**distributed MNIST (tensorflow) using kubeflow**

This example is to show how to run distributed training on kubeflow through NNI. Users can simply provide distributed training code and a configure file which specifies the kubeflow mode. For example, what is the command to run ps and what is the command to run worker, and how many resources they consume. This example is implemented in tensorflow, thus, uses kubeflow tensorflow operator.

`code directory: examples/trials/mnist-distributed/`

<a name="mnist-kubeflow-pytorch"></a>
**distributed MNIST (pytorch) using kubeflow**

Similar to the previous example, the difference is that this example is implemented in pytorch, thus, it uses kubeflow pytorch operator.

`code directory: examples/trials/mnist-distributed-pytorch/`