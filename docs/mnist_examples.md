# MNIST examples

## Overview
CNN MNIST classifier for deep learning is similar to `hello world` for programming languages. The [MNIST](http://yann.lecun.com/exdb/mnist/) database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples.Thus, we use MNIST as example to introduce different features of NNI.
### **Goals**

NNI offers a variety of features to support different search space modes, different training platforms, different tuners and different deeplearning frameworks. The goal of this tutorial is to summarize the different features that make it easy for users to start the task that suits them.

In this example, we have selected the following common aspects:

> different search space modes ： "API", "annotation" <br>
> different training platforms ： "local", "remote", "pai", "kubeflow" <br>
> different tuners ："tpe", "batch-tune", "cascading", "hyperband" <br>
> different deeplearning frameworks ： "tensorflow", "pytorch", "keras" 

<table>
<tr><td rowspan="2"> <b>different search space modes</b><br/>
<td>annotation</td><td><a name= "#mnist" >MNIST with NNI API</a></td></tr>
<tr><td>API</td><td>2.7</td></tr>

<tr><td rowspan="2"> <b>different training platforms</b><br/>
<td>local, remote, pai</td><td> </td></tr>
<tr><td>kubeflow</td><td>2.7</td></tr>

<tr><td rowspan="4"> <b>different tuners</b><br/>
<td>tpe</td><td> </td></tr>
<tr><td>batch-tune</td><td>2.7</td></tr>
<tr><td>cascading</td><td>2.7</td></tr>
<tr><td>hyperband</td><td>2.7</td></tr>

<tr><td rowspan="3"><b> different deeplearning frameworks</b><br/>
<td>keras</td><td> </td></tr>
<tr><td>tensorflow</td><td>2.7</td></tr>
<tr><td>pytorch</td><td>2.7</td></tr>
</table>

 The examples are listed below:


|item name|search space|platform|framework|tuner|code directory
|---|---|---|---|---|---|
|[MNIST with NNI API](#mnist)|json|local,remote,pai,kubeflow|tensorflow|tpe|[examples/trials/mnist/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist)
|[MNIST with NNI annotation](#mnist-annotation)|annotation|local,remote,pai,kubeflow|tensorflow|tpe|[examples/trials/mnist-annotation/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-annotation)
|[MNIST in keras](#mnist-keras)|json|local,remote,pai|keras|tpe|[examples/trials/mnist-keras/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-keras)
|[MNIST -- tuning with batch tuner](#mnist-batch)|json|local,pai,remote|keras|batch|[examples/trials/mnist-batch-tune-keras/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-batch-tune-keras)
|[MNIST -- tuning with hyperband](#mnist-hyperband)|json|local,pai,remote|tensorflow|hyperband|[examples/trials/mnist-hyperband/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-hyperband)
|[MNIST -- tuning within a nested search space](#mnist-nested)|json|local,pai,remote|tensorflow|cascading|[examples/trials/mnist-cascading-search-space/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-cascading-search-space)
|[distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)|json|local,remote,pai,kubeflow|tensorflow|tpe|[examples/trials/mnist-distributed/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed)
|[distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)|json|local,remote,pai,kubeflow|pytorch|tpe|[examples/trials/mnist-distributed-pytorch/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)

### **Experimental**
#### **different search space modes**
<a name="mnist"></a>
**MNIST with NNI API**

This is a simple network which has two convolutional layers, two pooling layers and a fully connected layer. We tune hyperparameters, such as dropout rate, convolution size, hidden size, etc. It can be tuned with most NNI built-in tuners, such as TPE, SMAC, Random. We also provide an exmaple yaml file which enables assessor.

`code directory: examples/trials/mnist/`

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

`code directory: examples/trials/mnist-cascading-search-space/`

<a name="mnist-kubeflow-tf"></a>
**distributed MNIST (tensorflow) using kubeflow**

This example is to show how to run distributed training on kubeflow through NNI. Users can simply provide distributed training code and a configure file which specifies the kubeflow mode. For example, what is the command to run ps and what is the command to run worker, and how many resources they consume. This example is implemented in tensorflow, thus, uses kubeflow tensorflow operator.

`code directory: examples/trials/mnist-distributed/`

<a name="mnist-kubeflow-pytorch"></a>
**distributed MNIST (pytorch) using kubeflow**

Similar to the previous example, the difference is that this example is implemented in pytorch, thus, it uses kubeflow pytorch operator.

`code directory: examples/trials/mnist-distributed-pytorch/`



 - [MNIST with NNI API](#mnist)
 - [MNIST with NNI annotation](#mnist-annotation)
 - [MNIST in keras](#mnist-keras)
 - [MNIST -- tuning with batch tuner](#mnist-batch)
 - [MNIST -- tuning with hyperband](#mnist-hyperband)
 - [MNIST -- tuning within a nested search space](#mnist-nested)
 - [distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)
 - [distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)
