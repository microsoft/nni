# Built-in Tuners for Hyperparameter Tuning

NNI provides state-of-the-art tuning algorithms as part of our built-in tuners and makes them easy to use. Below is the brief summary of NNI's current built-in tuners:

Note: Click the **Tuner's name** to get the Tuner's installation requirements, suggested scenario, and an example configuration. A link for a detailed description of each algorithm is located at the end of the suggested scenario for each tuner. Here is an [article](../CommunitySharings/HpoComparison.md) comparing different Tuners on several problems.

Currently, we support the following algorithms:

|Tuner|Brief Introduction of Algorithm|
|---|---|
|[__TPE__](#TPE)|The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. [Reference Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)|
|[__Random Search__](#Random)|In Random Search for Hyper-Parameter Optimization show that Random Search might be surprisingly simple and effective. We suggest that we could use Random Search as the baseline when we have no knowledge about the prior distribution of hyper-parameters. [Reference Paper](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)|
|[__Anneal__](#Anneal)|This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on the random search that leverages smoothness in the response surface. The annealing rate is not adaptive.|
|[__Naïve Evolution__](#Evolution)|Naïve Evolution comes from Large-Scale Evolution of Image Classifiers. It randomly initializes a population-based on search space. For each generation, it chooses better ones and does some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naïve Evolution requires many trials to work, but it's very simple and easy to expand new features. [Reference paper](https://arxiv.org/pdf/1703.01041.pdf)|
|[__SMAC__](#SMAC)|SMAC is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by NNI is a wrapper on the SMAC3 GitHub repo. Notice, SMAC needs to be installed by `nnictl package` command. [Reference Paper,](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) [GitHub Repo](https://github.com/automl/SMAC3)|
|[__Batch tuner__](#Batch)|Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type choice in search space spec.|
|[__Grid Search__](#GridSearch)|Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file. Note that the only acceptable types of search space are choice, quniform, randint. |
|[__Hyperband__](#Hyperband)|Hyperband tries to use limited resources to explore as many configurations as possible and returns the most promising ones as a final result. The basic idea is to generate many configurations and run them for a small number of trials. The half least-promising configurations are thrown out, the remaining are further trained along with a selection of new configurations. The size of these populations is sensitive to resource constraints (e.g. allotted search time). [Reference Paper](https://arxiv.org/pdf/1603.06560.pdf)|
|[__Network Morphism__](#NetworkMorphism)|Network Morphism provides functions to automatically search for deep learning architectures. It generates child networks that inherit the knowledge from their parent network which it is a morph from. This includes changes in depth, width, and skip-connections. Next, it estimates the value of a child network using historic architecture and metric pairs. Then it selects the most promising one to train. [Reference Paper](https://arxiv.org/abs/1806.10282)|
|[__Metis Tuner__](#MetisTuner)|Metis offers the following benefits when it comes to tuning parameters: While most tools only predict the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guesswork. While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter. [Reference Paper](https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/)|
|[__BOHB__](#BOHB)|BOHB is a follow-up work to Hyperband. It targets the weakness of Hyperband that new configurations are generated randomly without leveraging finished trials. For the name BOHB, HB means Hyperband, BO means Bayesian Optimization. BOHB leverages finished trials by building multiple TPE models, a proportion of new configurations are generated through these models. [Reference Paper](https://arxiv.org/abs/1807.01774)|
|[__GP Tuner__](#GPTuner)|Gaussian Process Tuner is a sequential model-based optimization (SMBO) approach with Gaussian Process as the surrogate. [Reference Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf), [Github Repo](https://github.com/fmfn/BayesianOptimization)|
|[__PPO Tuner__](#PPOTuner)|PPO Tuner is a Reinforcement Learning tuner based on PPO algorithm. [Reference Paper](https://arxiv.org/abs/1707.06347)|
|[__PBT Tuner__](#PBTTuner)|PBT Tuner is a simple asynchronous optimization algorithm which effectively utilizes a fixed computational budget to jointly optimize a population of models and their hyperparameters to maximize performance. [Reference Paper](https://arxiv.org/abs/1711.09846v1)|

## Usage of Built-in Tuners

Using a built-in tuner provided by the NNI SDK requires one to declare the  **builtinTunerName** and **classArgs** in the `config.yml` file. In this part, we will introduce each tuner along with information about usage and suggested scenarios, classArg requirements, and an example configuration.

Note: Please follow the format when you write your `config.yml` file. Some built-in tuners need to be installed using `nnictl package`, like SMAC.

<a name="TPE"></a>

### TPE

> Built-in Tuner Name: **TPE**

**Suggested scenario**

TPE, as a black-box optimization, can be used in various scenarios and shows good performance in general. Especially when you have limited computation resources and can only try a small number of trials. From a large amount of experiments, we found that TPE is far better than Random Search. [Detailed Description](./HyperoptTuner.md)


**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.

Note: We have optimized the parallelism of TPE for large-scale trial concurrency. For the principle of optimization or turn-on optimization, please refer to [TPE document](./HyperoptTuner.md).

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
```

<br>

<a name="Random"></a>

### Random Search

> Built-in Tuner Name: **Random**

**Suggested scenario**

Random search is suggested when each trial does not take very long (e.g., each trial can be completed very quickly, or early stopped by the assessor), and you have enough computational resources. It's also useful if you want to uniformly explore the search space. Random Search can be considered a baseline search algorithm. [Detailed Description](./HyperoptTuner.md)

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: Random
```

<br>

<a name="Anneal"></a>

### Anneal

> Built-in Tuner Name: **Anneal**

**Suggested scenario**

Anneal is suggested when each trial does not take very long and you have enough computation resources (very similar to Random Search). It's also useful when the variables in the search space can be sample from some prior distribution. [Detailed Description](./HyperoptTuner.md)


**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: Anneal
  classArgs:
    optimize_mode: maximize
```

<br>

<a name="Evolution"></a>

### Naïve Evolution

> Built-in Tuner Name: **Evolution**

**Suggested scenario**

Its computational resource requirements are relatively high. Specifically, it requires a large initial population to avoid falling into a local optimum. If your trial is short or leverages assessor, this tuner is a good choice. It is also suggested when your trial code supports weight transfer; that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training process. [Detailed Description](./EvolutionTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.

* **population_size** (*int value (should > 0), optional, default = 20*) - the initial size of the population (trial num) in the evolution tuner. It's suggested that `population_size` be much larger than `concurrency` so users can get the most out of the algorithm (and at least `concurrency`, or the tuner will fail on its first generation of parameters).

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: Evolution
  classArgs:
    optimize_mode: maximize
    population_size: 100
```

<br>

<a name="SMAC"></a>

### SMAC

> Built-in Tuner Name: **SMAC**

**Please note that SMAC doesn't support running on Windows currently. For the specific reason, please refer to this [GitHub issue](https://github.com/automl/SMAC3/issues/483).**

**Installation**

SMAC needs to be installed by following command before the first usage. As a reminder, `swig` is required for SMAC: for Ubuntu `swig` can be installed with `apt`.

```bash
nnictl package install --name=SMAC
```

**Suggested scenario**

Similar to TPE, SMAC is also a black-box tuner that can be tried in various scenarios and is suggested when computational resources are limited. It is optimized for discrete hyperparameters, thus, it's suggested when most of your hyperparameters are discrete. [Detailed Description](./SmacTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **config_dedup** (*True or False, optional, default = False*) - If True, the tuner will not generate a configuration that has been already generated. If False, a configuration may be generated twice, but it is rare for a relatively large search space.

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: SMAC
  classArgs:
    optimize_mode: maximize
```

<br>

<a name="Batch"></a>

### Batch Tuner

> Built-in Tuner Name: BatchTuner

**Suggested scenario**

If the configurations you want to try have been decided beforehand, you can list them in search space file (using `choice`) and run them using batch tuner. [Detailed Description](./BatchTuner.md)

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: BatchTuner
```

<br>

Note that the search space for BatchTuner should look like:

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

The search space file should include the high-level key `combine_params`. The type of params in the search space must be `choice` and the `values` must include all the combined params values.

<a name="GridSearch"></a>

### Grid Search

> Built-in Tuner Name: **Grid Search**

**Suggested scenario**

Note that the only acceptable types within the search space are `choice`, `quniform`, and `randint`.

This is suggested when the search space is small. It's suggested when it is feasible to exhaustively sweep the whole search space. [Detailed Description](./GridsearchTuner.md)

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: GridSearch
```

<br>

<a name="Hyperband"></a>

### Hyperband

> Built-in Advisor Name: **Hyperband**

**Suggested scenario**

This is suggested when you have limited computational resources but have a relatively large search space. It performs well in scenarios where intermediate results can indicate good or bad final results to some extent. For example, when models that are more accurate early on in training are also more accurate later on. [Detailed Description](./HyperbandAdvisor.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **R** (*int, optional, default = 60*) - the maximum budget given to a trial (could be the number of mini-batches or epochs). Each trial should use TRIAL_BUDGET to control how long they run.
* **eta** (*int, optional, default = 3*) - `(eta-1)/eta` is the proportion of discarded trials.

**Example Configuration:**

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

### Network Morphism

> Built-in Tuner Name: **NetworkMorphism**

**Installation**

NetworkMorphism requires [PyTorch](https://pytorch.org/get-started/locally) and [Keras](https://keras.io/#installation), so users should install them first. The corresponding requirements file is [here](https://github.com/microsoft/nni/blob/master/examples/trials/network_morphism/requirements.txt).

**Suggested scenario**

This is suggested when you want to apply deep learning methods to your task but you have no idea how to choose or design a network. You may modify this [example](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism/cifar10/cifar10_keras.py) to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate, or optimizer. Currently, this tuner only supports the computer vision domain. [Detailed Description](./NetworkmorphismTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **task** (*('cv'), optional, default = 'cv'*) - The domain of the experiment. For now, this tuner only supports the computer vision (CV) domain.
* **input_width** (*int, optional, default = 32*) - input image width
* **input_channel** (*int, optional, default = 3*) - input image channel
* **n_output_node** (*int, optional, default = 10*) - number of classes

**Example Configuration:**

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

### Metis Tuner

> Built-in Tuner Name: **MetisTuner**

Note that the only acceptable types of search space types are `quniform`, `uniform`, `randint`, and numerical `choice`. Only numerical values are supported since the values will be used to evaluate the 'distance' between different points.

**Suggested scenario**

Similar to TPE and SMAC, Metis is a black-box tuner. If your system takes a long time to finish each trial, Metis is more favorable than other approaches such as random search. Furthermore, Metis provides guidance on subsequent trials. Here is an [example](https://github.com/Microsoft/nni/tree/master/examples/trials/auto-gbdt/search_space_metis.json) on the use of Metis. Users only need to send the final result, such as `accuracy`, to the tuner by calling the NNI SDK. [Detailed Description](./MetisTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*'maximize' or 'minimize', optional, default = 'maximize'*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: MetisTuner
  classArgs:
    optimize_mode: maximize
```

<br>

<a name="BOHB"></a>

### BOHB Advisor

> Built-in Tuner Name: **BOHB**

**Installation**

BOHB advisor requires [ConfigSpace](https://github.com/automl/ConfigSpace) package. ConfigSpace can be installed using the following command.

```bash
nnictl package install --name=BOHB
```

**Suggested scenario**

Similar to Hyperband, BOHB is suggested when you have limited computational resources but have a relatively large search space. It performs well in scenarios where intermediate results can indicate good or bad final results to some extent. In this case, it may converge to a better configuration than Hyperband due to its usage of Bayesian optimization. [Detailed Description](./BohbAdvisor.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will try to maximize metrics. If 'minimize', tuner will try to minimize metrics.
* **min_budget** (*int, optional, default = 1*) - The smallest budget to assign to a trial job, (budget can be the number of mini-batches or epochs). Needs to be positive.
* **max_budget** (*int, optional, default = 3*) - The largest budget to assign to a trial job, (budget can be the number of mini-batches or epochs). Needs to be larger than min_budget.
* **eta** (*int, optional, default = 3*) - In each iteration, a complete run of sequential halving is executed. In it, after evaluating each configuration on the same subset size, only a fraction of 1/eta of them 'advances' to the next round. Must be greater or equal to 2.
* **min_points_in_model**(*int, optional, default = None*): number of observations to start building a KDE. Default 'None' means dim+1; when the number of completed trials in this budget is equal to or larger than `max{dim+1, min_points_in_model}`, BOHB will start to build a KDE model of this budget then use said KDE model to guide configuration selection. Needs to be positive. (dim means the number of hyperparameters in search space)
* **top_n_percent**(*int, optional, default = 15*): percentage (between 1 and 99) of the observations which are considered good. Good points and bad points are used for building KDE models. For example, if you have 100 observed trials and top_n_percent is 15, then the top 15% of points will be used for building the good points models "l(x)". The remaining 85% of points will be used for building the bad point models "g(x)".
* **num_samples**(*int, optional, default = 64*): number of samples to optimize EI (default 64). In this case, we will sample "num_samples" points and compare the result of l(x)/g(x). Then we will return the one with the maximum l(x)/g(x) value as the next configuration if the optimize_mode is `maximize`. Otherwise, we return the smallest one.
* **random_fraction**(*float, optional, default = 0.33*): fraction of purely random configurations that are sampled from the prior without the model.
* **bandwidth_factor**(*float, optional, default = 3.0*): to encourage diversity, the points proposed to optimize EI are sampled from a 'widened' KDE where the bandwidth is multiplied by this factor. We suggest using the default value if you are not familiar with KDE.
* **min_bandwidth**(*float, optional, default = 0.001*): to keep diversity, even when all (good) samples have the same value for one of the parameters, a minimum bandwidth (default: 1e-3) is used instead of zero. We suggest using the default value if you are not familiar with KDE.

*Please note that the float type currently only supports decimal representations. You have to use 0.333 instead of 1/3 and 0.001 instead of 1e-3.*

**Example Configuration:**

```yaml
advisor:
  builtinAdvisorName: BOHB
  classArgs:
    optimize_mode: maximize
    min_budget: 1
    max_budget: 27
    eta: 3
```

<a name="GPTuner"></a>

### GP Tuner

> Built-in Tuner Name: **GPTuner**

Note that the only acceptable types within the search space are `randint`, `uniform`, `quniform`,  `loguniform`, `qloguniform`, and numerical `choice`. Only numerical values are supported since the values will be used to evaluate the 'distance' between different points.

**Suggested scenario**

As a strategy in a Sequential Model-based Global Optimization (SMBO) algorithm, GP Tuner uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) to solve and common tools can be employed to solve it. Therefore, GP Tuner is most adequate for situations where the function to be optimized is very expensive to evaluate. GP can be used when computational resources are limited. However, GP Tuner has a computational cost that grows at *O(N^3)* due to the requirement of inverting the Gram matrix, so it's not suitable when lots of trials are needed. [Detailed Description](./GPTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*'maximize' or 'minimize', optional, default = 'maximize'*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **utility** (*'ei', 'ucb' or 'poi', optional, default = 'ei'*) - The utility function (acquisition function). 'ei', 'ucb', and 'poi' correspond to 'Expected Improvement', 'Upper Confidence Bound', and 'Probability of Improvement', respectively.
* **kappa** (*float, optional, default = 5*) - Used by the 'ucb' utility function. The bigger `kappa` is, the more exploratory the tuner will be.
* **xi** (*float, optional, default = 0*) - Used by the 'ei' and 'poi' utility functions. The bigger `xi` is, the more exploratory the tuner will be.
* **nu** (*float, optional, default = 2.5*) - Used to specify the Matern kernel. The smaller nu, the less smooth the approximated function is.
* **alpha** (*float, optional, default = 1e-6*) - Used to specify the Gaussian Process Regressor. Larger values correspond to an increased noise level in the observations.
* **cold_start_num** (*int, optional, default = 10*) - Number of random explorations to perform before the Gaussian Process. Random exploration can help by diversifying the exploration space.
* **selection_num_warm_up** (*int, optional, default = 1e5*) - Number of random points to evaluate when getting the point which maximizes the acquisition function.
* **selection_num_starting_points** (*int, optional, default = 250*) - Number of times to run L-BFGS-B from a random starting point after the warmup.

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: GPTuner
  classArgs:
    optimize_mode: maximize
    utility: 'ei'
    kappa: 5.0
    xi: 0.0
    nu: 2.5
    alpha: 1e-6
    cold_start_num: 10
    selection_num_warm_up: 100000
    selection_num_starting_points: 250
```

<a name="PPOTuner"></a>

### PPO Tuner

> Built-in Tuner Name: **PPOTuner**

Note that the only acceptable types within the search space are `layer_choice` and `input_choice`. For `input_choice`, `n_chosen` can only be 0, 1, or [0, 1]. Note, the search space file for NAS is usually automatically generated through the command [`nnictl ss_gen`](../Tutorial/Nnictl.md).

**Suggested scenario**

PPOTuner is a Reinforcement Learning tuner based on the PPO algorithm. PPOTuner can be used when using the NNI NAS interface to do neural architecture search. In general, the Reinforcement Learning algorithm needs more computing resources, though the PPO algorithm is relatively more efficient than others. It's recommended to use this tuner when you have a large amount of computional resources available. You could try it on a very simple task, such as the [mnist-nas](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nas) example. [See details](./PPOTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*'maximize' or 'minimize'*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **trials_per_update** (*int, optional, default = 20*) - The number of trials to be used for one update. It must be divisible by minibatch_size. `trials_per_update` is recommended to be an exact multiple of `trialConcurrency` for better concurrency of trials.
* **epochs_per_update** (*int, optional, default = 4*) - The number of epochs for one update.
* **minibatch_size** (*int, optional, default = 4*) - Mini-batch size (i.e., number of trials for a mini-batch) for the update. Note that trials_per_update must be divisible by minibatch_size.
* **ent_coef** (*float, optional, default = 0.0*) - Policy entropy coefficient in the optimization objective.
* **lr** (*float, optional, default = 3e-4*) - Learning rate of the model (lstm network); constant.
* **vf_coef** (*float, optional, default = 0.5*) - Value function loss coefficient in the optimization objective.
* **max_grad_norm** (*float, optional, default = 0.5*) - Gradient norm clipping coefficient.
* **gamma** (*float, optional, default = 0.99*) - Discounting factor.
* **lam** (*float, optional, default = 0.95*) - Advantage estimation discounting factor (lambda in the paper).
* **cliprange** (*float, optional, default = 0.2*) - Cliprange in the PPO algorithm, constant.

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: PPOTuner
  classArgs:
    optimize_mode: maximize
```

<a name="PBTTuner"></a>

### PBT Tuner

> Built-in Tuner Name: **PBTTuner**

**Suggested scenario**

Population Based Training (PBT) bridges and extends parallel search methods and sequential optimization methods. It requires relatively small computation resource, by inheriting weights from currently good-performing ones to explore better ones periodically. With PBTTuner, users finally get a trained model, rather than a configuration that could reproduce the trained model by training the model from scratch. This is because model weights are inherited periodically through the whole search process. PBT can also be seen as a training approach. If you don't need to get a specific configuration, but just expect a good model, PBTTuner is a good choice. [See details](./PBTTuner.md)

**classArgs requirements:**

* **optimize_mode** (*'maximize' or 'minimize'*) - If 'maximize', the tuner will target to maximize metrics. If 'minimize', the tuner will target to minimize metrics.
* **all_checkpoint_dir** (*str, optional, default = None*) - Directory for trials to load and save checkpoint, if not specified, the directory would be "~/nni/checkpoint/<exp-id>". Note that if the experiment is not local mode, users should provide a path in a shared storage which can be accessed by all the trials.
* **population_size** (*int, optional, default = 10*) - Number of trials in a population. Each step has this number of trials. In our implementation, one step is running each trial by specific training epochs set by users.
* **factors** (*tuple, optional, default = (1.2, 0.8)*) - Factors for perturbation of hyperparameters.
* **fraction** (*float, optional, default = 0.2*) - Fraction for selecting bottom and top trials.

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: PBTTuner
  classArgs:
    optimize_mode: maximize
```

Note that, to use this tuner, your trial code should be modified accordingly, please refer to [the document of PBTTuner](./PBTTuner.md) for details.


## **Reference and Feedback**
* To [report a bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md) for this feature in GitHub;
* To [file a feature or improvement request](https://github.com/microsoft/nni/issues/new?template=enhancement.md) for this feature in GitHub;
* To know more about [Feature Engineering with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/FeatureEngineering/Overview.md);
* To know more about [NAS with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/NAS/Overview.md);
* To know more about [Model Compression with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/Compressor/Overview.md);
