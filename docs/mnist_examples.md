# MNIST examples

CNN MNIST classifier for deep learning is similar to `hello world` for programming languages. Thus, we use MNIST as example to introduce different features of NNI. The examples are listed below:

 - [mnist](#mnist)
 - [mnist with NNI annotation](#mnist-annotation)
 - [mnist in keras](#mnist-keras)
 - [mnist -- tuning with batch tuner](#mnist-batch)
 - [mnist -- tuning with hyperband](#mnist-hyperband)
 - [mnist -- tuning within a nested search space](#mnist-nested)
 - [distributed mnist (tensorflow) using kubeflow](#mnist-kubeflow-tf)
 - [distributed mnist (pytorch) using kubeflow](#mnist-kubeflow-pytorch)

<a name="mnist"></a>
**mnist**

This is a simple network which has two convolutional layers, two pooling layers and a fully connected layer. We tune hyperparameters, such as dropout rate, convolution size, hidden size, etc. It can be tuned with most NNI built-in tuners, such as TPE, SMAC, Random. We also provide an exmaple yaml file which enables assessor.

`code directory: nni/examples/trials/mnist/`

<a name="mnist-annotation"></a>
**mnist with NNI annotation**

This example is similar to the example above, the only difference is that this example uses NNI annotation to specify search space and report results, while the example above uses NNI apis to receive configuration and report results.

`code directory: nni/examples/trials/mnist-annotation/`

<a name="mnist-keras"></a>
**mnist in keras**

This example is implemented in keras. It is also a network for MNIST dataset, with two convolution layers, one pooling layer, and two fully connected layers.

`code directory: nni/examples/trials/mnist-keras/`

<a name="mnist-batch"></a>
**mnist -- tuning with batch tuner**

This example is to show how to use batch tuner. Users simply list all the configurations they want to try in the search space file. NNI will try all of them.

`code directory: nni/examples/trials/mnist-batch-tune-keras/`

<a name="mnist-hyperband"></a>
**mnist -- tuning with hyperband**

This example is to show how to use hyperband to tune the model. There is one more key `STEPS` in the received configuration for trials to control how long it can run (e.g., number of iterations).

`code directory: nni/examples/trials/mnist-hyperband/`

<a name="mnist-nested"></a>
**mnist -- tuning within a nested search space**

This example is to show that NNI also support nested search space. The search space file is an example of how to define nested search space.

`code directory: nni/examples/trials/mnist-cascading-search-space/`

<a name="mnist-kuberflow-tf"></a>
**distributed mnist (tensorflow) using kubeflow**

This example is to show how to run distributed training on kubeflow through NNI. Users can simply provide distributed training code and a configure file which specifies the kubeflow mode. For example, what is the command to run ps and what is the command to run worker, and how many resources they consume. This example is implemented in tensorflow, thus, uses kubeflow tensorflow operator.

`code directory: nni/examples/trials/mnist-distributed/`

<a name="mnist-kuberflow-pytorch"></a>
**distributed mnist (pytorch) using kubeflow**

Similar to the previous example, the difference is that this example is implemented in pytorch, thus, it uses kubeflow pytorch operator.

`code directory: nni/examples/trials/mnist-distributed-pytorch/`