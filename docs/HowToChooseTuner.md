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
 - [Network Morphism](#NetworkMorphism) (require pyTorch)
 - [Metis Tuner](#MetisTuner) (require sklearn)


 ## Supported tuner algorithms


We will introduce some basic knowledge about the tuner algorithms, suggested scenarios for each tuner, and their example usage (for complete usage spec, please refer to [here]()).

<a name="TPE"></a>
**TPE**

The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model.
The TPE approach models P(x|y) and P(y) where x represents hyperparameters and y the associated evaluate matric. P(x|y) is modeled by transforming the generative process of hyperparameters, replacing the distributions of the configuration prior with non-parametric densities. 
This optimization approach is described in detail in [Algorithms for Hyper-Parameter Optimization][1].
​    
_Suggested scenario_: TPE, as a black-box optimization, can be used in various scenarios, and shows good performance in general. Especially when you have limited computation resource and can only try a small number of trials. From a large amount of experiments, we could found that TPE is far better than Random Search.

_Usage_:
```yaml
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
```yaml
  # config.yaml
  tuner:
    builtinTunerName: Random
```

<a name="Anneal"></a>
**Anneal**

This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on random search that leverages smoothness in the response surface. The annealing rate is not adaptive.

_Suggested scenario_: Anneal is suggested when each trial does not take too long, and you have enough computation resource(almost same with Random Search). Or the variables in search space could be sample from some prior distribution.

_Usage_:
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

Naive Evolution comes from [Large-Scale Evolution of Image Classifiers][3]. It randomly initializes a population based on search space. For each generation, it chooses better ones and do some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naive Evolution requires many trials to works, but it's very simple and easily to expand new features.

_Suggested scenario_: Its requirement of computation resource is relatively high. Specifically, it requires large inital population to avoid falling into local optimum. If your trial is short or leverages assessor, this tuner is a good choice. And, it is more suggested when your trial code supports weight transfer, that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training progress.

_Usage_:
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

[SMAC][4] is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by nni is a wrapper on [the SMAC3 github repo][5].

Note that SMAC on nni only supports a subset of the types in [search space spec](./SearchSpaceSpec.md), including `choice`, `randint`, `uniform`, `loguniform`, `quniform(q=1)`.

_Installation_: 
* Install swig first. (`sudo apt-get install swig` for Ubuntu users)
* Run `nnictl package install --name=SMAC`

_Suggested scenario_: Similar to TPE, SMAC is also a black-box tuner which can be tried in various scenarios, and is suggested when computation resource is limited. It is optimized for discrete hyperparameters, thus, suggested when most of your hyperparameters are discrete.

_Usage_:
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

_Suggested sceanrio_: If the configurations you want to try have been decided, you can list them in searchspace file (using `choice`) and run them using batch tuner.

_Usage_:
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

Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file.
Note that the only acceptable types of search space are `choice`, `quniform`, `qloguniform`. **The number `q` in `quniform` and `qloguniform` has special meaning (different from the spec in [search space spec](./SearchSpaceSpec.md)). It means the number of values that will be sampled evenly from the range `low` and `high`.**

_Suggested scenario_: It is suggested when search space is small, it is feasible to exhaustively sweeping the whole search space.

_Usage_:
```yaml
  # config.yaml
  tuner:
    builtinTunerName: GridSearch
```

<a name="Hyperband"></a>
**Hyperband**

[Hyperband][6] tries to use limited resource to explore as many configurations as possible, and finds out the promising ones to get the final result. The basic idea is generating many configurations and to run them for small number of STEPs to find out promising one, then further training those promising ones to select several more promising one. More detail can be referred to [here](../src/sdk/pynni/nni/hyperband_advisor/README.md).

_Suggested scenario_: It is suggested when you have limited computation resource but have relatively large search space. It performs good in the scenario that intermediate result (e.g., accuracy) can reflect good or bad of final result (e.g., accuracy) to some extent.

_Usage_:
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

[Network Morphism][7] provides functions to automatically search for architecture of deep learning models. Every child network inherits the knowledge from its parent network and morphs into diverse types of networks, including changes of depth, width and skip-connection. Next, it estimates the value of child network using the history architecture and metric pairs. Then it selects the most promising one to train. More detail can be referred to [here](../src/sdk/pynni/nni/networkmorphism_tuner/README.md). 

_Installation_: 
NetworkMorphism requires [pyTorch](https://pytorch.org/get-started/locally), so users should install it first.


_Suggested scenario_: It is suggested that you want to apply deep learning methods to your task (your own dataset) but you have no idea of how to choose or design a network. You modify the [example](../examples/trials/network_morphism/cifar10/cifar10_keras.py) to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate or optimizer. It is feasible for different tasks to find a good network architecture. Now this tuner only supports the cv domain.

_Usage_:
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


<a name="MetisTuner"></a>
**Metis Tuner**

[Metis][10] offers the following benefits when it comes to tuning parameters:
While most tools only predicts the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guess work!

While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter.

While most tools have problems of being exploitation-heavy, Metis' search strategy balances exploration, exploitation, and (optional) re-sampling.
 
Metis belongs to the class of sequential model-based optimization (SMBO), and it is based on the Bayesian Optimization framework. To model the parameter-vs-performance space, Metis uses both Gaussian Process and GMM. Since each trial can impose a high time cost, Metis heavily trades inference computations with naive trial. At each iteration, Metis does two tasks:
*  It finds the global optimal point in the Gaussian Process space. This point represents the optimal configuration.
* It identifies the next hyper-parameter candidate. This is achieved by inferring the potential information gain of exploration, exploitation, and re-sampling.

Note that the only acceptable types of search space are `choice`, `quniform`, `uniform` and `randint`. We only support 
numerical `choice` now. More features will support later.

More details can be found in our paper: https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/
 

_Installation_: 
Metis Tuner requires [sklearn](https://scikit-learn.org/), so users should install it first. User could use `pip3 install sklearn` to install it.


_Suggested scenario_:
Similar to TPE and SMAC, Metis is a black-box tuner. If your system takes a long time to finish each trial, Metis is more favorable than other approaches such as random search. Furthermore, Metis provides guidance on the subsequent trial. Here is an [example](../examples/trials/auto-gbdt/search_space_metis.json) about the use of Metis. User only need to send the final result like `accuracy` to tuner, by calling the nni SDK.

_Usage_:
```yaml
  # config.yaml
  tuner:
    builtinTunerName: MetisTuner
    classArgs:
      #choice: maximize, minimize
      optimize_mode: maximize
```

<a name="assessor"></a>
# How to use Assessor that NNI supports?

For now, NNI has supported the following assessor algorithms.

 - [Medianstop](#Medianstop)
 - [Curvefitting](#Curvefitting)

## Supported Assessor Algorithms

<a name="Medianstop"></a>
**Medianstop**

Medianstop is a simple early stopping rule mentioned in the [paper][8]. It stops a pending trial X at step S if the trial’s best objective value by step S is strictly worse than the median value of the running averages of all completed trials’ objectives reported up to step S.

_Suggested scenario_: It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress.

_Usage_:
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

Curve Fitting Assessor is a LPA(learning, predicting, assessing) algorithm. It stops a pending trial X at step S if the prediction of final epoch's performance worse than the best final performance in the trial history. In this algorithm, we use 12 curves to fit the accuracy curve, the large set of parametric curve models are chosen from [reference paper][9]. The learning curves' shape coincides with our prior knowlwdge about the form of learning curves: They are typically increasing, saturating functions.

_Suggested scenario_: It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress. Even better, it's able to handle and assess curves with similar performance. 

_Usage_:
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

[1]: https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
[2]: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
[3]: https://arxiv.org/pdf/1703.01041.pdf
[4]: https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
[5]: https://github.com/automl/SMAC3
[6]: https://arxiv.org/pdf/1603.06560.pdf
[7]: https://arxiv.org/abs/1806.10282
[8]: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf
[9]: http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf
[10]:https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/ 
