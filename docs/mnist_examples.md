# MNIST examples

## Overview
CNN MNIST classifier for deep learning is similar to `hello world` for programming languages. The [MNIST](http://yann.lecun.com/exdb/mnist/) database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples.Thus, we use MNIST as example to introduce different features of NNI.
## Goals

NNI offers a variety of features to support different search space modes, different training platforms, different tuners and different deeplearning frameworks. The goal of this tutorial is to summarize the different features that make it easy for users to start the task that suits them.

In this example, we have selected the following common aspects:

* different search space modes ： "api", "annotation" <br>

|type|item name|code directory|
|---|---|---|
|annotation|[MNIST with NNI annotation](#mnist-annotation)|[examples/trials/mnist-annotation/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-annotation)
|api|[MNIST with NNI API](#mnist)<br>[MNIST in keras](#mnist-keras)<br>[MNIST -- tuning with batch tuner](#mnist-batch)<br>[MNIST -- tuning with hyperband](#mnist-hyperband)<br>[MNIST -- tuning within a nested search space](#mnist-nested)<br>[distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)<br>[distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)|[examples/trials/mnist/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist)<br>[examples/trials/mnist-keras/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-keras)<br>[/examples/trials/mnist-batch-tune-keras](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-batch-tune-keras)<br>[examples/trials/mnist-hyperband/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-hyperband)<br>[examples/trials/mnist-cascading-search-space/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-cascading-search-space)<br>[examples/trials/mnist-distributed/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed)<br>[examples/trials/mnist-distributed-pytorch/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)



* different training platforms ： "local", "remote", "pai", "kubeflow" <br>
|type|item name|code directory|
|---|---|---|
|local,remote,pai|
|local,remote,pai,kubeflow|


* different tuners ："tpe", "batch-tune", "hyperband" <br>
|type|item name|code directory|
|---|---|---|
|tpe|[MNIST with NNI API](#mnist)<br>[MNIST with NNI annotation](#mnist-annotation)<br>[MNIST in keras](#mnist-keras)<br>[MNIST -- tuning within a nested search space](#mnist-nested)<br>[distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)<br>[distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)|<br>[examples/trials/mnist/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist)<br>[examples/trials/mnist-annotation/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-annotation)<br>[examples/trials/mnist-keras/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-keras)<br>[examples/trials/mnist-cascading-search-space/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-cascading-search-space)<br>[examples/trials/mnist-distributed/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed)<br>[examples/trials/mnist-distributed-pytorch/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)|
|batch|[MNIST -- tuning with batch tuner](#mnist-batch)|[/examples/trials/mnist-batch-tune-keras](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-batch-tune-keras)|
|hyperband|[MNIST -- tuning with hyperband](#mnist-hyperband)|[examples/trials/mnist-hyperband/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-hyperband)


* different deeplearning frameworks ： "tensorflow", "pytorch", "keras" 
|type|item name|code directory|
|---|---|---|
|tensorflow|
|pytorch|[distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)|[examples/trials/mnist-distributed-pytorch/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)|
|keras|[MNIST in keras](#mnist-keras)<br>[MNIST -- tuning with batch tuner](#mnist-batch)|[examples/trials/mnist-keras/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-keras)<br>


## Experimental

### Search Space

As we stated in the target, we target to find out the best `optimizer` for training. When using different optimizers and different hyperparameters ,we write the choices into json file or in codes as annotation.

### Trial

The code for CNN training of each hyperparameters set, paying particular attention to the following points are specific for NNI:
* Use `nni.get_next_parameter()` to get next training hyperparameter set.
* Use `nni.report_intermediate_result(acc)` to report the intermedian result after finish each epoch.
* Use `nni.report_intermediate_result(acc)` to report the final result before the trial end.

### Config

Here are some examples of running this experiment on local,pai,remote or kubeflow.

### Lauch the experiment

We are ready for the experiment, let's now **run the config.yml file from your command line to start the experiment**.

 ```bash
    nnictl create --config nni/examples/trials/*mnist*/*config*.yml
```

## Examples

The examples are listed below:


|item name|search space|platform|framework|tuner|code directory
|---|---|---|---|---|---|
|[MNIST with NNI API](#mnist)|json|local,remote,pai,kubeflow|tensorflow|tpe|[examples/trials/mnist/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist)
|[MNIST with NNI annotation](#mnist-annotation)|annotation|local,remote,pai,kubeflow|tensorflow|tpe|[examples/trials/mnist-annotation/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-annotation)
|[MNIST in keras](#mnist-keras)|json|local,remote,pai|keras|tpe|[examples/trials/mnist-keras/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-keras)
|[MNIST -- tuning with batch tuner](#mnist-batch)|json|local,pai,remote|keras|batch|[/examples/trials/mnist-batch-tune-keras](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-batch-tune-keras)
|[MNIST -- tuning with hyperband](#mnist-hyperband)|json|local,pai,remote|tensorflow|hyperband|[examples/trials/mnist-hyperband/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-hyperband)
|[MNIST -- tuning within a nested search space](#mnist-nested)|json|local,pai,remote|tensorflow|tpe|[examples/trials/mnist-cascading-search-space/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-cascading-search-space)
|[distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)|json|local,remote,pai,kubeflow|tensorflow|tpe|[examples/trials/mnist-distributed/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed)
|[distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)|json|local,remote,pai,kubeflow|pytorch|tpe|[examples/trials/mnist-distributed-pytorch/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)



<a name ="mnist"></a>
**MNIST with NNI API**

This is a simple network which has two convolutional layers, two pooling layers and a fully connected layer. We tune hyperparameters, such as dropout rate, convolution size, hidden size, etc. It can be tuned with most NNI built-in tuners, such as TPE, SMAC, Random. We also provide an exmaple yaml file which enables assessor.

As we stated in the target, we target to find out the best `optimizer` for training . When using different optimizers, we also need to adjust `dropout_rate` ,`conv_size`,`hidden_size`,`batch_size`and `learning_rate` accordingly. so we chose these three parameters as hyperparameters and write the following search space.

```json
{
    "dropout_rate":{"_type":"uniform","_value":[0.5, 0.9]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "batch_size": {"_type":"choice", "_value": [1, 4, 8, 16, 32]},
    "learning_rate":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]}
}
```

`code directory: examples/trials/mnist/`

<a name="mnist-annotation"></a>
**MNIST with NNI annotation**

This example is similar to the example above, the only difference is that this example uses NNI annotation to specify search space and report results, while the example above uses NNI apis to receive configuration and report results.The annotation examples is as following.
```python3
"""@nni.variable(nni.choice(2, 3, 5, 7),name=self.conv_size)"""
        self.conv_size = conv_size
        """@nni.variable(nni.choice(124, 512, 1024), name=self.hidden_size)"""
        self.hidden_size = hidden_size
        self.pool_size = pool_size
        """@nni.variable(nni.loguniform(0.0001, 0.1), name=self.learning_rate)"""
        self.learning_rate = learning_rate
```
`code directory: examples/trials/mnist-annotation/`

<a name="mnist-keras"></a>
**MNIST in keras**

This example is implemented in keras. It is also a network for MNIST dataset, with two convolution layers, one pooling layer, and two fully connected layers.

`code directory: examples/trials/mnist-keras/`

<a name="mnist-batch"></a>
**MNIST -- tuning with batch tuner**

This example is to show how to use batch tuner. Users simply list all the configurations they want to try in the search space file. NNI will try all of them.If the configurations you want to try have been decided, you can list them in searchspace file (using `choice`) and run them using batch tuner.
```yaml
# config.yml
tuner:
  builtinTunerName: BatchTuner
```
`code directory: examples/trials/mnist-batch-tune-keras/`

<a name="mnist-hyperband"></a>
**MNIST -- tuning with hyperband**

This example is to show how to use hyperband to tune the model. There is one more key `STEPS` in the received configuration for trials to control how long it can run (e.g., number of iterations).
```yaml
# config.yml
advisor:
  builtinAdvisorName: Hyperband
  classArgs:
    optimize_mode: maximize
    R: 60
    eta: 3
```
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
