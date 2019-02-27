NNI offers a variety of features to support different search space modes, different training platforms, different tuners and different deeplearning frameworks. The goal of this tutorial is to summarize the different features that make it easy for users to start the task that suits them.

We have selected the following common aspects:

* different search space modes ： "api", "annotation" <br>

|type|item name|code directory|
|---|---|---|
|annotation|[MNIST with NNI annotation](#mnist-annotation)|[examples/trials/mnist-annotation/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-annotation)
|api|[MNIST with NNI API](#mnist)<br>[MNIST in keras](#mnist-keras)<br>[MNIST -- tuning with batch tuner](#mnist-batch)<br>[MNIST -- tuning with hyperband](#mnist-hyperband)<br>[MNIST -- tuning within a nested search space](#mnist-nested)<br>[distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)<br>[distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)|[examples/trials/mnist/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist)<br>[examples/trials/mnist-keras/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-keras)<br>[examples/trials/mnist-batch-tune-keras](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-batch-tune-keras)<br>[examples/trials/mnist-hyperband/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-hyperband)<br>[examples/trials/mnist-cascading-search-space/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-cascading-search-space)<br>[examples/trials/mnist-distributed/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed)<br>[examples/trials/mnist-distributed-pytorch/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)



* different training platforms ： "local", "remote", "pai", "kubeflow" <br>


|type|item name|code directory|
|---|---|---|
|local<br>remote<br>pai<br>kubeflow|[MNIST with NNI API](#mnist)<br>[distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)<br>[distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)|[examples/trials/mnist/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist)<br>[examples/trials/mnist-distributed/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed)<br>[examples/trials/mnist-distributed-pytorch/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)
|local<br>remote<br>pai|[MNIST with NNI annotation](#mnist-annotation)<br>[MNIST in keras](#mnist-keras)<br>[MNIST -- tuning with batch tuner](#mnist-batch)<br>[MNIST -- tuning with hyperband](#mnist-hyperband)<br>[MNIST -- tuning within a nested search space](#mnist-nested)|[examples/trials/mnist-annotation/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-annotation)<br>[examples/trials/mnist-keras/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-keras)<br>[examples/trials/mnist-batch-tune-keras](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-batch-tune-keras)<br>[examples/trials/mnist-hyperband/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-hyperband)<br>[examples/trials/mnist-cascading-search-space/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-cascading-search-space)



* different tuners ："tpe", "batch-tune", "hyperband" <br>

|type|item name|code directory|
|---|---|---|
|batch|[MNIST -- tuning with batch tuner](#mnist-batch)|[examples/trials/mnist-batch-tune-keras](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-batch-tune-keras) |
|hyperband|[MNIST -- tuning with hyperband](#mnist-hyperband)|[examples/trials/mnist-hyperband](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-hyperband)|
|tpe|[MNIST with NNI API](#mnist)<br>[MNIST with NNI annotation](#mnist-annotation)<br>[MNIST in keras](#mnist-keras)<br>[MNIST -- tuning within a nested search space](#mnist-nested)<br>[distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)<br>[distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)|[examples/trials/mnist/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist)<br>[examples/trials/mnist-annotation/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-annotation)<br>[examples/trials/mnist-keras/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-keras)<br>[examples/trials/mnist-cascading-search-space/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-cascading-search-space)<br>[examples/trials/mnist-distributed/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed)<br>[examples/trials/mnist-distributed-pytorch/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)|



* different deeplearning frameworks ： "tensorflow", "pytorch", "keras" 

|type|item name|code directory|
|---|---|---|
|tensorflow|[MNIST with NNI API](#mnist)<br>[MNIST with NNI annotation](#mnist-annotation)<br>[MNIST -- tuning with hyperband](#mnist-hyperband)<br>[MNIST -- tuning within a nested search space](#mnist-nested)<br>[distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)|[examples/trials/mnist/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist)<br>[examples/trials/mnist-annotation/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-annotation)<br>[examples/trials/mnist-hyperband/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-hyperband)<br>[examples/trials/mnist-cascading-search-space/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-cascading-search-space)<br>[examples/trials/mnist-distributed/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed)
|pytorch|[distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)|[examples/trials/mnist-distributed-pytorch/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)|
|keras|[MNIST in keras](#mnist-keras)<br>[MNIST -- tuning with batch tuner](#mnist-batch)|[examples/trials/mnist-keras/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-keras)<br>

We also have some other examples :<br>
<br><br>[CIFAR-10 examples](https://github.com/Microsoft/nni/tree/master/examples/trials/cifar10_pytorch):CIFAR-10 classification is a common benchmark problem in machine learning. The CIFAR-10 dataset is the collection of images. It is one of the most widely used datasets for machine learning research which contains 60,000 32x32 color images in 10 different classes. Thus, we use CIFAR-10 classification as an example to introduce NNI usage.
<br><br>[Scikit-learn in NNI](https://github.com/Microsoft/nni/tree/master/examples/trials/sklearn):Scikit-learn is a pupular meachine learning tool for data mining and data analysis. It supports many kinds of meachine learning models like LinearRegression, LogisticRegression, DecisionTree, SVM etc. How to make the use of scikit-learn more efficiency is a valuable topic.NNI supports many kinds of tuning algorithms to search the best models and/or hyper-parameters for scikit-learn, and support many kinds of environments like local machine, remote servers and cloud.
<br><br>[Automatic Model Architecture Search for Reading Comprehension](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism):This example shows us how to use Genetic Algorithm to find good model architectures for Reading Comprehension.
<br><br>[GBDT in nni](https://github.com/Microsoft/nni/tree/master/examples/trials/auto-gbdt):Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion as other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.
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
