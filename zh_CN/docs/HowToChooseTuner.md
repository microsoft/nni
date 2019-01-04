# 如何使用 NNI 支持的调参器？

目前，NNI 已支持下列调参器算法。 注意，NNI 只安装了部分下列算法，其它算法在使用前需要通过 `nnictl package install` 命令安装。 例如，安装 SMAC 算法的命令为 `nnictl package install --name=SMAC`。

* [TPE](#TPE)
* [Random Search（随机搜索）](#Random)
* [Anneal（退火算法）](#Anneal)
* [Naive Evolution（遗传算法）](#Evolution)
* [SMAC](#SMAC) (需要通过 `nnictl` 命令安装)
* [Batch Tuner](#Batch)
* [Grid Search（网格搜索）](#Grid)
* [Hyperband](#Hyperband)
* [Network Morphism](#NetworkMorphism) (需要安装 pyTorch)
    
    ## 支持的调参器算法

这里将介绍这些调参器算法的基本知识，各个调参器建议使用的场景，以及使用样例（完整的使用样例参考 [这里]()）。

<a name="TPE"></a>
**TPE**

Tree-structured Parzen Estimator (TPE) 是一种 sequential model-based optimization（SMBO，即基于序列模型优化）的方法。 SMBO 方法根据历史指标数据来按顺序构造模型，来估算超参的性能，随后基于此模型来选择新的超参。 TPE 方法对 P(x|y) 和 P(y) 建模，其中 x 表示超参，y 表示相关的评估指标。 P(x|y) 通过变换超参的生成过程来建模，用非参数密度（non-parametric densities）代替配置的先验分布。 细节可参考 [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)。 ​  
*建议场景*： TPE 作为黑盒的优化方法，能在广泛的场景中使用，通常都能得到较好的结果。 特别是在计算资源有限，只能进行少量尝试时。 从大量的实验中，我们 发现 TPE 的性能远远优于随机搜索。

*用法*:

```yaml
  # config.yaml
  tuner:
    builtinTunerName: TPE
    classArgs:
      # 可选项: maximize, minimize
      optimize_mode: maximize
```

<a name="Random"></a>
**随机搜索**

[Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) 中介绍了随机搜索惊人的简单和效果。 We suggests that we could use Random Search as baseline when we have no knowledge about the prior distribution of hyper-parameters.

*Suggested scenario*: Random search is suggested when each trial does not take too long (e.g., each trial can be completed very soon, or early stopped by assessor quickly), and you have enough computation resource. Or you want to uniformly explore the search space. Random Search could be considered as baseline of search algorithm.

*Usage*:

```yaml
  # config.yaml
  tuner:
    builtinTunerName: Random
```

<a name="Anneal"></a>
**Anneal**

This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on random search that leverages smoothness in the response surface. The annealing rate is not adaptive.

*Suggested scenario*: Anneal is suggested when each trial does not take too long, and you have enough computation resource(almost same with Random Search). Or the variables in search space could be sample from some prior distribution.

*Usage*:

```yaml
  # config.yaml
  tuner:
    builtinTunerName: Anneal
    classArgs:
      # choice: maximize, minimize
      optimize_mode: maximize
```

<a name="Evolution"></a>
**Naive Evolution**

Naive Evolution comes from [Large-Scale Evolution of Image Classifiers](https://arxiv.org/pdf/1703.01041.pdf). It randomly initializes a population based on search space. For each generation, it chooses better ones and do some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naive Evolution requires many trials to works, but it's very simple and easily to expand new features.

*Suggested scenario*: Its requirement of computation resource is relatively high. Specifically, it requires large inital population to avoid falling into local optimum. If your trial is short or leverages assessor, this tuner is a good choice. And, it is more suggested when your trial code supports weight transfer, that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training progress.

*Usage*:

```yaml
  # config.yaml
  tuner:
    builtinTunerName: Evolution
    classArgs:
      # choice: maximize, minimize
      optimize_mode: maximize
```

<a name="SMAC"></a>
**SMAC**

[SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by nni is a wrapper on [the SMAC3 github repo](https://github.com/automl/SMAC3).

Note that SMAC on nni only supports a subset of the types in [search space spec](./SearchSpaceSpec.md), including `choice`, `randint`, `uniform`, `loguniform`, `quniform(q=1)`.

*Installation*:

* Install swig first. (`sudo apt-get install swig` for Ubuntu users)
* Run `nnictl package install --name=SMAC`

*Suggested scenario*: Similar to TPE, SMAC is also a black-box tuner which can be tried in various scenarios, and is suggested when computation resource is limited. It is optimized for discrete hyperparameters, thus, suggested when most of your hyperparameters are discrete.

*Usage*:

```yaml
  # config.yaml
  tuner:
    builtinTunerName: SMAC
    classArgs:
      # choice: maximize, minimize
      optimize_mode: maximize
```

<a name="Batch"></a>
**Batch tuner**

Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type `choice` in [search space spec](./SearchSpaceSpec.md).

*Suggested sceanrio*: If the configurations you want to try have been decided, you can list them in searchspace file (using `choice`) and run them using batch tuner.

*Usage*:

```yaml
  # config.yaml
  tuner:
    builtinTunerName: BatchTuner
```

Note that the search space that BatchTuner supported like:

```json
{
    "combine_params":
    {
        "_type" : "choice",
        "_value" : [{"optimizer": "Adam", "learning_rate": 0.00001},
                    {"optimizer": "Adam", "learning_rate": 0.0001},
                    {"optimizer": "Adam", "learning_rate": 0.001},
                    {"optimizer": "SGD", "learning_rate": 0.01}, 
                    {"optimizer": "SGD", "learning_rate": 0.005}, 
                    {"optimizer": "SGD", "learning_rate": 0.0002}]
    }
}
```

The search space file including the high-level key `combine_params`. The type of params in search space must be `choice` and the `values` including all the combined-params value.

<a name="Grid"></a>
**Grid Search**

Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file. Note that the only acceptable types of search space are `choice`, `quniform`, `qloguniform`. **The number `q` in `quniform` and `qloguniform` has special meaning (different from the spec in [search space spec](./SearchSpaceSpec.md)). It means the number of values that will be sampled evenly from the range `low` and `high`.**

*Suggested scenario*: It is suggested when search space is small, it is feasible to exhaustively sweeping the whole search space.

*Usage*:

```yaml
  # config.yaml
  tuner:
    builtinTunerName: GridSearch
```

<a name="Hyperband"></a>
**Hyperband**

[Hyperband](https://arxiv.org/pdf/1603.06560.pdf) tries to use limited resource to explore as many configurations as possible, and finds out the promising ones to get the final result. The basic idea is generating many configurations and to run them for small number of STEPs to find out promising one, then further training those promising ones to select several more promising one. More detail can be referred to [here](../src/sdk/pynni/nni/hyperband_advisor/README.md).

*Suggested scenario*: It is suggested when you have limited computation resource but have relatively large search space. It performs good in the scenario that intermediate result (e.g., accuracy) can reflect good or bad of final result (e.g., accuracy) to some extent.

*Usage*:

```yaml
  # config.yaml
  advisor:
    builtinAdvisorName: Hyperband
    classArgs:
      # choice: maximize, minimize
      optimize_mode: maximize
      # R: the maximum STEPS (could be the number of mini-batches or epochs) can be
      #    allocated to a trial. Each trial should use STEPS to control how long it runs.
      R: 60
      # eta: proportion of discarded trials
      eta: 3
```

<a name="NetworkMorphism"></a>
**Network Morphism**

[Network Morphism](7) provides functions to automatically search for architecture of deep learning models. Every child network inherits the knowledge from its parent network and morphs into diverse types of networks, including changes of depth, width and skip-connection. Next, it estimates the value of child network using the history architecture and metric pairs. Then it selects the most promising one to train. More detail can be referred to [here](../src/sdk/pynni/nni/networkmorphism_tuner/README.md).

*Installation*: NetworkMorphism requires [pyTorch](https://pytorch.org/get-started/locally), so users should install it first.

*Suggested scenario*: It is suggested that you want to apply deep learning methods to your task (your own dataset) but you have no idea of how to choose or design a network. You modify the [example](../examples/trials/network_morphism/cifar10/cifar10_keras.py) to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate or optimizer. It is feasible for different tasks to find a good network architecture. Now this tuner only supports the cv domain.

*Usage*:

```yaml
  # config.yaml
  tuner:
    builtinTunerName: NetworkMorphism
    classArgs:
      #choice: maximize, minimize
      optimize_mode: maximize
      #for now, this tuner only supports cv domain
      task: cv
      #input image width
      input_width: 32
      #input image channel
      input_channel: 3
      #number of classes
      n_output_node: 10
```

# How to use Assessor that NNI supports?

For now, NNI has supported the following assessor algorithms.

* [Medianstop](#Medianstop)
* [Curvefitting](#Curvefitting)

## Supported Assessor Algorithms

<a name="Medianstop"></a>
**Medianstop**

Medianstop is a simple early stopping rule mentioned in the [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf). It stops a pending trial X at step S if the trial’s best objective value by step S is strictly worse than the median value of the running averages of all completed trials’ objectives reported up to step S.

*Suggested scenario*: It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress.

*Usage*:

```yaml
  assessor:
    builtinAssessorName: Medianstop
    classArgs:
      #choice: maximize, minimize
      optimize_mode: maximize
      # (optional) A trial is determined to be stopped or not, 

      * only after receiving start_step number of reported intermediate results.
      * The default value of start_step is 0.
      start_step: 5
```

<a name="Curvefitting"></a>
**Curvefitting**

Curve Fitting Assessor is a LPA(learning, predicting, assessing) algorithm. It stops a pending trial X at step S if the prediction of final epoch's performance worse than the best final performance in the trial history. In this algorithm, we use 12 curves to fit the accuracy curve, the large set of parametric curve models are chosen from [reference paper](http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf). The learning curves' shape coincides with our prior knowlwdge about the form of learning curves: They are typically increasing, saturating functions.

*Suggested scenario*: It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress. Even better, it's able to handle and assess curves with similar performance.

*Usage*:

```yaml
  assessor:
    builtinAssessorName: Curvefitting
    classArgs:
      # (required)The total number of epoch.
      # We need to know the number of epoch to determine which point we need to predict.
      epoch_num: 20
      # (optional) choice: maximize, minimize
      # Kindly reminds that if you choose minimize mode, please adjust the value of threshold >= 1.0 (e.g threshold=1.1)

      * The default value of optimize_mode is maximize
      optimize_mode: maximize
      # (optional) A trial is determined to be stopped or not
      # In order to save our computing resource, we start to predict when we have more than start_step(default=6) accuracy points.
      # only after receiving start_step number of reported intermediate results.
      * The default value of start_step is 6.
      start_step: 6
      # (optional) The threshold that we decide to early stop the worse performance curve.
      # For example: if threshold = 0.95, optimize_mode = maximize, best performance in the history is 0.9, then we will stop the trial which predict value is lower than 0.95 * 0.9 = 0.855.
      * The default value of threshold is 0.95.
      threshold: 0.95
```