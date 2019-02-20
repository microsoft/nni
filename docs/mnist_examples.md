# MNIST examples

## Overview
CNN MNIST classifier for deep learning is similar to `hello world` for programming languages. The [MNIST](http://yann.lecun.com/exdb/mnist/) database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples.Thus, we use MNIST as example to introduce different features of NNI.
### **Goals**

NNI offers a variety of features to support different search space modes, different training platforms, different tuners and different deeplearning frameworks. The goal of this tutorial is to summarize the different features that make it easy for users to start the task that suits them.

In this example, we have selected the following common aspects:

> different search space modes ： "API", "annotation" <br>
> different training platforms ： "local", "remote", "pai", "kubeflow" <br>
> different tuners ： "batch-tune", "cascading", "hyperband" <br>
> different deeplearning frameworks ： "tensorflow", "pytorch", "keras" 
 
 The examples are listed below:
 

|Tuner|Brief Introduction of Algorithm|
|---|---|
|**TPE**<br>[(MNIST with NNI API)](#mnist)|The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model.|
|**Random Search**<br>[(Usage)](#Random)|In Random Search for Hyper-Parameter Optimization show that Random Search might be surprisingly simple and effective. We suggest that we could use Random Search as the baseline when we have no knowledge about the prior distribution of hyper-parameters.|
|**Anneal**<br>[(Usage)](#Anneal)|This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on the random search that leverages smoothness in the response surface. The annealing rate is not adaptive.|
|**Naive Evolution**<br>[(Usage)](#Evolution)|Naive Evolution comes from Large-Scale Evolution of Image Classifiers. It randomly initializes a population-based on search space. For each generation, it chooses better ones and does some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naive Evolution requires many trials to works, but it's very simple and easy to expand new features.|
|**SMAC**<br>[(Usage)](#SMAC)|SMAC is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by nni is a wrapper on the SMAC3 Github repo.|
|**Batch tuner**<br>[(Usage)](#Batch)|Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type choice in search space spec.|
|**Grid Search**<br>[(Usage)](#GridSearch)|Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file. Note that the only acceptable types of search space are choice, quniform, qloguniform. The number q in quniform and qloguniform has special meaning (different from the spec in search space spec). It means the number of values that will be sampled evenly from the range low and high.|
|[Hyperband](https://github.com/Microsoft/nni/tree/master/src/sdk/pynni/nni/hyperband_advisor)<br>[(Usage)](#Hyperband)|Hyperband tries to use the limited resource to explore as many configurations as possible, and finds out the promising ones to get the final result. The basic idea is generating many configurations and to run them for the small number of STEPs to find out promising one, then further training those promising ones to select several more promising one.|
|[Network Morphism](https://github.com/Microsoft/nni/blob/master/src/sdk/pynni/nni/networkmorphism_tuner/README.md)<br>[(Usage)](#NetworkMorphism)|Network Morphism provides functions to automatically search for architecture of deep learning models. Every child network inherits the knowledge from its parent network and morphs into diverse types of networks, including changes of depth, width, and skip-connection. Next, it estimates the value of a child network using the historic architecture and metric pairs. Then it selects the most promising one to train.|
|**Metis Tuner**<br>[(Usage)](#MetisTuner)|Metis offers the following benefits when it comes to tuning parameters: While most tools only predict the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guesswork. While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter.|

<br>
 
[(Usage)](#SMAC)
 
<table>
        <tr>
            <th>item name</th>
            <th>framework</th>
            <th>platform</th>
            <th>searchspace</th>
            <th>tuner</th>
        </tr>
        <tr>
            <th>
                [(MNIST with NNI API)](#mnist)
            </th>
            <th>/dev/stdin</th>
            <th>0</th>
            <th>标准输入</th>
            <th> </th>
        </tr>
        <tr>
            <th> - [MNIST with NNI annotation](#mnist-annotation)</th>
            <th>/dev/stdin</th>
            <th>0</th>
            <th>标准输入</th>
            <th> </th>
        </tr>
         <tr>
            <th> - [MNIST in keras](#mnist-keras)</th>
            <th>/dev/stdin</th>
            <th>0</th>
            <th>标准输入</th>
            <th> </th>
        </tr>
         <tr>
            <th> - [MNIST -- tuning with batch tuner](#mnist-batch)</th>
            <th>/dev/stdin</th>
            <th>0</th>
            <th>标准输入</th>
            <th> </th>
        </tr>
         <tr>
            <th> - [MNIST -- tuning with hyperband](#mnist-hyperband)</th>
            <th>/dev/stdin</th>
            <th>0</th>
            <th>标准输入</th>
            <th> </th>
        </tr>
         <tr>
            <th> - [MNIST -- tuning with hyperband](#mnist-hyperband)</th>
            <th>/dev/stdin</th>
            <th>0</th>
            <th>标准输入</th>
            <th> </th>
        </tr>
         <tr>
            <th> - [distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)</th>
            <th>/dev/stdin</th>
            <th>0</th>
            <th>标准输入</th>
            <th> </th>
        </tr>
         <tr>
            <th> - [distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)</th>
            <th>/dev/stdin</th>
            <th>0</th>
            <th>标准输入</th>
            <th> </th>
        </tr>
</table>


 - [MNIST with NNI API](#mnist)
 - [MNIST with NNI annotation](#mnist-annotation)
 - [MNIST in keras](#mnist-keras)
 - [MNIST -- tuning with batch tuner](#mnist-batch)
 - [MNIST -- tuning with hyperband](#mnist-hyperband)
 - [MNIST -- tuning within a nested search space](#mnist-nested)
 - [distributed MNIST (tensorflow) using kubeflow](#mnist-kubeflow-tf)
 - [distributed MNIST (pytorch) using kubeflow](#mnist-kubeflow-pytorch)

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
