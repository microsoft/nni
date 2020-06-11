# MNIST examples

CNN MNIST classifier for deep learning is similar to `hello world` for programming languages. Thus, we use MNIST as example to introduce different features of NNI. The examples are listed below:

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

This is a simple network which has two convolutional layers, two pooling layers and a fully connected layer. We tune hyperparameters, such as dropout rate, convolution size, hidden size, etc. It can be tuned with most NNI built-in tuners, such as TPE, SMAC, Random. We also provide an exmaple YAML file which enables assessor.

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
