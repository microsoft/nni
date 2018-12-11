# How to use Tuner that NNI supports?

For now, NNI has supported the following tuner algorithms. Note that NNI installation only installs a subset of those algorithms, other algorithms should be installed through `nnictl package install` before you use them. For example, for SMAC the installation command is `nnictl package install --name=SMAC`.

 - [TPE](#TPE)
 - [Random Search](#Random)
 - [Anneal](#Anneal)
 - [Naive Evolution](#Evolution)
 - [SMAC](#SMAC) (to install through `nnictl`)
 - [Batch Tuner](#Batch)
 - [Grid Search](#Grid)
 - [Hyperband](#Hyperband)

 ## Supported tuner algorithms


We will introduce some basic knowledge about the tuner algorithms, suggested scenarios for each tuner, and their example usage (for complete usage spec, please refer to [here]()).

<a name="TPE"></a>
**TPE**

The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model.
The TPE approach models P(x|y) and P(y) where x represents hyperparameters and y the associated evaluate matric. P(x|y) is modeled by transforming the generative process of hyperparameters, replacing the distributions of the configuration prior with non-parametric densities. 
This optimization approach is described in detail in [Algorithms for Hyper-Parameter Optimization][1].
    
_Suggested scenario_: TPE, as a black-box optimization, can be used in various scenarios, and shows good performance in general. Especially when you have limited computation resource and can only try a small number of trials. From a large amount of experiments, we could found that TPE is far better than Random Search.

_Usage_:
```
  # config.yaml
  tuner:
    builtinTunerName: TPE
    classArgs:
      # choice: maximize, minimize
      optimize_mode: maximize
```

<a name="Random"></a>
**Random Search**

In [Random Search for Hyper-Parameter Optimization][2] show that Random Search might be surprisingly simple and effective. We suggests that we could use Random Search as baseline when we have no knowledge about the prior distribution of hyper-parameters.

_Suggested scenario_: Random search is suggested when each trial does not take too long (e.g., each trial can be completed very soon, or early stopped by assessor quickly), and you have enough computation resource. Or you want to uniformly explore the search space. Random Search could be considered as baseline of search algorithm.

_Usage_:
```
  # config.yaml
  tuner:
    builtinTunerName: Random
```

<a name="Anneal"></a>
**Anneal**

This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on random search that leverages smoothness in the response surface. The annealing rate is not adaptive.

_Suggested scenario_: Anneal is suggested when each trial does not take too long, and you have enough computation resource(almost same with Random Search). Or the variables in search space could be sample from some prior distribution.

_Usage_:
```
  # config.yaml
  tuner:
    builtinTunerName: Anneal
    classArgs:
      # choice: maximize, minimize
      optimize_mode: maximize
```

<a name="Evolution"></a>
**Naive Evolution**

Naive Evolution comes from [Large-Scale Evolution of Image Classifiers][3]. It randomly initializes a population based on search space. For each generation, it chooses better ones and do some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naive Evolution requires many trials to works, but it's very simple and easily to expand new features.

_Suggested scenario_: Its requirement of computation resource is relatively high. Specifically, it requires large inital population to avoid falling into local optimum. If your trial is short or leverages assessor, this tuner is a good choice. And, it is more suggested when your trial code supports weight transfer, that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training progress.

_Usage_:
```
  # config.yaml
  tuner:
    builtinTunerName: Evolution
    classArgs:
      # choice: maximize, minimize
      optimize_mode: maximize
```

<a name="SMAC"></a>
**SMAC**

[SMAC][4] is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by nni is a wrapper on [the SMAC3 github repo][5].

Note that SMAC on nni only supports a subset of the types in [search space spec](./SearchSpaceSpec.md), including `choice`, `randint`, `uniform`, `loguniform`, `quniform(q=1)`.

_Installation_: 
* Install swig first
* Run `nni package install --name=smac`

_Suggested scenario_: Similar to TPE, SMAC is also a black-box tuner which can be tried in various scenarios, and is suggested when computation resource is limited. It is optimized for discrete hyperparameters, thus, suggested when most of your hyperparameters are discrete.

_Usage_:
```
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

_Suggested sceanrio_: If the configurations you want to try have been decided, you can list them in searchspace file (using `choice`) and run them using batch tuner.

_Usage_:
```
  # config.yaml
  tuner:
    builtinTunerName: BatchTuner
```

Note that the search space that BatchTuner supported like:
```
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

Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file.
Note that the only acceptable types of search space are `choice`, `quniform`, `qloguniform`. **The number `q` in `quniform` and `qloguniform` has special meaning (different from the spec in [search space spec](./SearchSpaceSpec.md)). It means the number of values that will be sampled evenly from the range `low` and `high`.**

_Suggested scenario_: It is suggested when search space is small, it is feasible to exhaustively sweeping the whole search space.

_Usage_:
```
  # config.yaml
  tuner:
    builtinTunerName: GridSearch
```

<a name="Hyperband"></a>
**Hyperband**

[Hyperband][6] tries to use limited resource to explore as many configurations as possible, and finds out the promising ones to get the final result. The basic idea is generating many configurations and to run them for small number of STEPs to find out promising one, then further training those promising ones to select several more promising one. More detail can be refered to [here](../src/sdk/pynni/nni/hyperband_advisor/README.md)

_Suggested scenario_: It is suggested when you have limited computation resource but have relatively large search space. It performs good in the scenario that intermediate result (e.g., accuracy) can reflect good or bad of final result (e.g., accuracy) to some extent.

_Usage_:
```
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


# How to use Assessor that NNI supports?

For now, NNI has supported the following assessor algorithms.

 - [Medianstop](#Medianstop)
 - Curve Extrapolation (ongoing)

## Supported Assessor Algorithms

<a name="Medianstop"></a>
**Medianstop**

Medianstop is a simple early stopping rule mentioned in the [paper][7]. It stops a pending trial X at step S if the trial’s best objective value by step S is strictly worse than the median value of the running averages of all completed trials’ objectives reported up to step S.

_Suggested scenario_: It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress.

_Usage_:
```
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

  [1]: https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
  [2]: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
  [3]: https://arxiv.org/pdf/1703.01041.pdf
  [4]: https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
  [5]: https://github.com/automl/SMAC3
  [6]: https://arxiv.org/pdf/1603.06560.pdf
  [7]: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf