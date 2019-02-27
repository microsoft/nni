# Built-in Tuners

NNI provides state-of-the-art tuning algorithm as our builtin-tuners and makes them easy to use. Below is the brief summary of NNI currently built-in Tuners:

Note: Click the **Tuner's name** to get a detailed description of the algorithm, click the corresponding **Usage** to get the Tuner's installation requirements, suggested scenario and using example.

Currently we support the following algorithms:
* [TPE](hyperoptTuner.md)
* [Random Search](hyperoptTuner.md)
* [Anneal](hyperoptTuner.md)
* [Naive Evolution](evolutionTuner.md)
* [SMAC](smacTuner.md)
* [Batch tuner](batchTuner.md)
* [Grid Search](gridsearchTuner.md)
* [Hyperband](hyperbandAdvisor.md)
* [Network Morphism](networkmorphismTuner.md)
* [Metis Tuner](metisTuner.md)

|Tuner|Brief Introduction of Algorithm|
|---|---|
|__TPE__ [(Usage)](#TPE)|The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. [Reference Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)|
|__Random Search__ [(Usage)](#Random)|In Random Search for Hyper-Parameter Optimization show that Random Search might be surprisingly simple and effective. We suggest that we could use Random Search as the baseline when we have no knowledge about the prior distribution of hyper-parameters. [Reference Paper](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)|
|__Anneal__ [(Usage)](#Anneal)|This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on the random search that leverages smoothness in the response surface. The annealing rate is not adaptive.|
|__Naive Evolution__ [(Usage)](#Evolution)|Naive Evolution comes from Large-Scale Evolution of Image Classifiers. It randomly initializes a population-based on search space. For each generation, it chooses better ones and does some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naive Evolution requires many trials to works, but it's very simple and easy to expand new features. [Reference paper](https://arxiv.org/pdf/1703.01041.pdf)|
|__SMAC__ [(Usage)](#SMAC)|SMAC is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by nni is a wrapper on the SMAC3 Github repo. Notice, SMAC need to be installed by `nnictl package` command. [Reference Paper,](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) [Github Repo](https://github.com/automl/SMAC3)|
|__Batch tuner__ [(Usage)](#Batch)|Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type choice in search space spec.|
|__Grid Search__ [(Usage)](#GridSearch)|Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file. Note that the only acceptable types of search space are choice, quniform, qloguniform. The number q in quniform and qloguniform has special meaning (different from the spec in search space spec). It means the number of values that will be sampled evenly from the range low and high.|
|__Hyperband__ [(Usage)](#Hyperband)|Hyperband tries to use the limited resource to explore as many configurations as possible, and finds out the promising ones to get the final result. The basic idea is generating many configurations and to run them for the small number of STEPs to find out promising one, then further training those promising ones to select several more promising one.[Reference Paper](https://arxiv.org/pdf/1603.06560.pdf)|
|__Network Morphism__ [(Usage)](#NetworkMorphism)|Network Morphism provides functions to automatically search for architecture of deep learning models. Every child network inherits the knowledge from its parent network and morphs into diverse types of networks, including changes of depth, width, and skip-connection. Next, it estimates the value of a child network using the historic architecture and metric pairs. Then it selects the most promising one to train. [Reference Paper](https://arxiv.org/abs/1806.10282)|
|__Metis Tuner__ [(Usage)](#MetisTuner)|Metis offers the following benefits when it comes to tuning parameters: While most tools only predict the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guesswork. While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter. [Reference Paper](https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/)|

<br>

## Usage of Builtin Tuners

Use builtin tuner provided by NNI SDK requires to declare the  **builtinTunerName** and **classArgs** in `config.yml` file. In this part, we will introduce the detailed usage about the suggested scenarios, classArg requirements and example for each tuner.

Note: Please follow the format when you write your `config.yml` file. Some builtin tuner need to be installed by `nnictl package`, like SMAC.

<a name="TPE"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `TPE`

> Builtin Tuner Name: **TPE**

**Suggested scenario**

TPE, as a black-box optimization, can be used in various scenarios and shows good performance in general. Especially when you have limited computation resource and can only try a small number of trials. From a large amount of experiments, we could found that TPE is far better than Random Search.

**Requirement of classArg**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will return the hyperparameter set with larger expectation. If 'minimize', tuner will return the hyperparameter set with smaller expectation.

**Usage example:**

```yaml
# config.yml
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
```

<br>

<a name="Random"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Random Search`

> Builtin Tuner Name: **Random**

**Suggested scenario**

Random search is suggested when each trial does not take too long (e.g., each trial can be completed very soon, or early stopped by assessor quickly), and you have enough computation resource. Or you want to uniformly explore the search space. Random Search could be considered as baseline of search algorithm.

**Requirement of classArg:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will return the hyperparameter set with larger expectation. If 'minimize', tuner will return the hyperparameter set with smaller expectation.

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: Random
  classArgs:
    optimize_mode: maximize
```

<br>

<a name="Anneal"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Anneal`

> Builtin Tuner Name: **Anneal**

**Suggested scenario**

Anneal is suggested when each trial does not take too long, and you have enough computation resource(almost same with Random Search). Or the variables in search space could be sample from some prior distribution.

**Requirement of classArg**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will return the hyperparameter set with larger expectation. If 'minimize', tuner will return the hyperparameter set with smaller expectation.

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: Anneal
  classArgs:
    optimize_mode: maximize
```

<br>

<a name="Evolution"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Naive Evolution`

> Builtin Tuner Name: **Evolution**

**Suggested scenario**

Its requirement of computation resource is relatively high. Specifically, it requires large initial population to avoid falling into local optimum. If your trial is short or leverages assessor, this tuner is a good choice. And, it is more suggested when your trial code supports weight transfer, that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training progress.

**Requirement of classArg**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will return the hyperparameter set with larger expectation. If 'minimize', tuner will return the hyperparameter set with smaller expectation.

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: Evolution
  classArgs:
    optimize_mode: maximize
```

<br>

<a name="SMAC"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `SMAC`

> Builtin Tuner Name: **SMAC**

**Installation**

SMAC need to be installed by following command before first use.

```bash
nnictl package install --name=SMAC
```

**Suggested scenario**

Similar to TPE, SMAC is also a black-box tuner which can be tried in various scenarios, and is suggested when computation resource is limited. It is optimized for discrete hyperparameters, thus, suggested when most of your hyperparameters are discrete.

**Requirement of classArg**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will return the hyperparameter set with larger expectation. If 'minimize', tuner will return the hyperparameter set with smaller expectation.

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: SMAC
  classArgs:
    optimize_mode: maximize
```

<br>

<a name="Batch"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Batch Tuner`

> Builtin Tuner Name: BatchTuner

**Suggested scenario**

If the configurations you want to try have been decided, you can list them in searchspace file (using `choice`) and run them using batch tuner.

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: BatchTuner
```

<br>

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

<a name="GridSearch"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Grid Search`

> Builtin Tuner Name: **Grid Search**

**Suggested scenario**

Note that the only acceptable types of search space are `choice`, `quniform`, `qloguniform`. **The number `q` in `quniform` and `qloguniform` has special meaning (different from the spec in [search space spec](./SearchSpaceSpec.md)). It means the number of values that will be sampled evenly from the range `low` and `high`.**

It is suggested when search space is small, it is feasible to exhaustively sweeping the whole search space.

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: GridSearch
```

<br>

<a name="Hyperband"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Hyperband`

> Builtin Advisor Name: **Hyperband**

**Suggested scenario**

It is suggested when you have limited computation resource but have relatively large search space. It performs well in the scenario that intermediate result (e.g., accuracy) can reflect good or bad of final result (e.g., accuracy) to some extent.

**Requirement of classArg**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will return the hyperparameter set with larger expectation. If 'minimize', tuner will return the hyperparameter set with smaller expectation.
* **R** (*int, optional, default = 60*) - the maximum STEPS (could be the number of mini-batches or epochs) can be allocated to a trial. Each trial should use STEPS to control how long it runs.
* **eta** (*int, optional, default = 3*) - `(eta-1)/eta` is the proportion of discarded trials

**Usage example**

```yaml
# config.yml
advisor:
  builtinAdvisorName: Hyperband
  classArgs:
    optimize_mode: maximize
    R: 60
    eta: 3
```

<br>

<a name="NetworkMorphism"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Network Morphism`

> Builtin Tuner Name: **NetworkMorphism**

**Installation**

NetworkMorphism requires [pyTorch](https://pytorch.org/get-started/locally), so users should install it first.

**Suggested scenario**

It is suggested that you want to apply deep learning methods to your task (your own dataset) but you have no idea of how to choose or design a network. You modify the [example](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism/cifar10/cifar10_keras.py) to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate or optimizer. It is feasible for different tasks to find a good network architecture. Now this tuner only supports the computer vision domain.

**Requirement of classArg**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will return the hyperparameter set with larger expectation. If 'minimize', tuner will return the hyperparameter set with smaller expectation.
* **task** (*('cv'), optional, default = 'cv'*) - The domain of experiment, for now, this tuner only supports the computer vision(cv) domain.
* **input_width** (*int, optional, default = 32*) - input image width
* **input_channel** (*int, optional, default = 3*) - input image channel
* **n_output_node** (*int, optional, default = 10*) - number of classes

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: NetworkMorphism
    classArgs:
      optimize_mode: maximize
      task: cv
      input_width: 32
      input_channel: 3
      n_output_node: 10
```

<br>

<a name="MetisTuner"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Metis Tuner`

> Builtin Tuner Name: **MetisTuner**

Note that the only acceptable types of search space are `choice`, `quniform`, `uniform` and `randint`.

**Installation**

Metis Tuner requires [sklearn](https://scikit-learn.org/), so users should install it first. User could use `pip3 install sklearn` to install it.

**Suggested scenario**

Similar to TPE and SMAC, Metis is a black-box tuner. If your system takes a long time to finish each trial, Metis is more favorable than other approaches such as random search. Furthermore, Metis provides guidance on the subsequent trial. Here is an [example](https://github.com/Microsoft/nni/tree/master/examples/trials/auto-gbdt/search_space_metis.json) about the use of Metis. User only need to send the final result like `accuracy` to tuner, by calling the nni SDK.

**Requirement of classArg**

* **optimize_mode** (*'maximize' or 'minimize', optional, default = 'maximize'*) - If 'maximize', tuners will return the hyperparameter set with larger expectation. If 'minimize', tuner will return the hyperparameter set with smaller expectation.

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: MetisTuner
  classArgs:
    optimize_mode: maximize
```
