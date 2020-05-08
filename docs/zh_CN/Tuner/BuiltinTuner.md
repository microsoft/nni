# 内置的超参调优 Tuner

NNI 提供了先进的调优算法，使用上也很简单。 下面是内置 Tuner 的简单介绍：

注意：点击 **Tuner 的名称**可看到 Tuner 的安装需求，建议的场景以及示例。 算法的详细说明在每个 Tuner 建议场景的最后。 [本文](../CommunitySharings/HpoComparision.md)对比了不同 Tuner 在几个问题下的不同效果。

当前支持的算法：

| Tuner（调参器）                               | 算法简介                                                                                                                                                                                                                                                                                          |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**TPE**](#TPE)                          | Tree-structured Parzen Estimator (TPE) 是一种 sequential model-based optimization（SMBO，即基于序列模型优化）的方法。 SMBO 方法根据历史指标数据来按顺序构造模型，来估算超参的性能，随后基于此模型来选择新的超参。 [参考论文](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)                                                 |
| [**Random Search（随机搜索）**](#Random)       | 在超参优化时，随机搜索算法展示了其惊人的简单和效果。 建议当不清楚超参的先验分布时，采用随机搜索作为基准。 [参考论文](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)                                                                                                                                                                 |
| [**Anneal（退火算法）**](#Anneal)              | 这种简单的退火算法从先前的采样开始，会越来越靠近发现的最佳点取样。 此算法是随机搜索的简单变体，利用了反应曲面的平滑性。 退火率不是自适应的。                                                                                                                                                                                                                       |
| [**Naïve Evolution（进化算法）**](#Evolution)  | Naïve Evolution（朴素进化算法）来自于 Large-Scale Evolution of Image Classifiers。 它会基于搜索空间随机生成一个种群。 在每一代中，会选择较好的结果，并对其下一代进行一些变异（例如，改动一个超参，增加或减少一层）。 朴素进化算法需要很多次的 Trial 才能有效，但它也非常简单，也很容易扩展新功能。 [参考论文](https://arxiv.org/pdf/1703.01041.pdf)                                                              |
| [**SMAC**](#SMAC)                        | SMAC 基于 Sequential Model-Based Optimization (SMBO，即序列的基于模型优化方法)。 它利用使用过的结果好的模型（高斯随机过程模型），并将随机森林引入到 SMBO 中，来处理分类参数。 SMAC 算法包装了 Github 的 SMAC3。 注意：SMAC 需要通过 `nnictl package` 命令来安装。 [参考论文，](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) [Github 代码库](https://github.com/automl/SMAC3) |
| [**Batch Tuner（批处理 Tuner）**](#Batch)     | Batch Tuner 能让用户简单的提供几组配置（如，超参选项的组合）。 当所有配置都执行完后，Experiment 即结束。 Batch Tuner 仅支持 choice 类型。                                                                                                                                                                                                   |
| [**Grid Search（遍历搜索）**](#GridSearch)     | Grid Search 会穷举定义在搜索空间文件中的所有超参组合。 遍历搜索可以使用的类型有 choice, quniform, randint。                                                                                                                                                                                                                     |
| [**Hyperband**](#Hyperband)              | Hyperband 试图用有限的资源来探索尽可能多的组合，并发现最好的结果。 基本思想是生成许多配置，并通过少量的 Trial 来运行一部分。 一半性能不好的配置会被抛弃，剩下的部分与新选择出的配置会进行下一步的训练。 数量的多少对资源约束非常敏感（例如，分配的搜索时间）。 [参考论文](https://arxiv.org/pdf/1603.06560.pdf)                                                                                                        |
| [**Network Morphism**](#NetworkMorphism) | 网络模态（Network Morphism）提供自动搜索深度学习体系结构的功能。 它会继承父网络的知识，来生成变形的子网络。 包括深度、宽度、跳连接等变化。 然后使用历史的架构和指标，来估计子网络的值。 然后会选择最有希望的模型进行训练。 [参考论文](https://arxiv.org/abs/1806.10282)                                                                                                                              |
| [**Metis Tuner**](#MetisTuner)           | 大多数调参工具仅仅预测最优配置，而 Metis 的优势在于有两个输出：(a) 最优配置的当前预测结果， 以及 (b) 下一次 Trial 的建议。 它不进行随机取样。 大多数工具假设训练集没有噪声数据，但 Metis 会知道是否需要对某个超参重新采样。 [参考论文](https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/)                                               |
| [**BOHB**](#BOHB)                        | BOHB 是 Hyperband 算法的后续工作。 Hyperband 在生成新的配置时，没有利用已有的 Trial 结果，而本算法利用了 Trial 结果。 BOHB 中，HB 表示 Hyperband，BO 表示贝叶斯优化（Byesian Optimization）。 BOHB 会建立多个 TPE 模型，从而利用已完成的 Trial 生成新的配置。 [参考论文](https://arxiv.org/abs/1807.01774)                                                                    |
| [**GP Tuner**](#GPTuner)                 | Gaussian Process（高斯过程） Tuner 是序列化的基于模型优化（SMBO）的方法，并使用了高斯过程来替代。 [参考论文](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)，[Github 库](https://github.com/fmfn/BayesianOptimization)                                                                             |
| [**PPO Tuner**](#PPOTuner)               | PPO Tuner 是基于 PPO 算法的强化学习 Tuner。 [参考论文](https://arxiv.org/abs/1707.06347)                                                                                                                                                                                                                     |
| [**PBT Tuner**](#PBTTuner)               | PBT Tuner 是一种简单的异步优化算法，在固定的计算资源下，它能有效的联合优化一组模型及其超参来最大化性能。 [参考论文](https://arxiv.org/abs/1711.09846v1)                                                                                                                                                                                          |

## 用法

要使用 NNI 内置的 Tuner，需要在 `config.yml` 文件中添加 **builtinTunerName** 和 **classArgs**。 本部分中，将介绍每个 Tuner 的用法和建议场景、参数要求，并提供配置示例。

注意：参考示例中的格式来创建新的 `config.yml` 文件。 一些内置的 Tuner 还需要通过 `nnictl package` 命令先安装，如 SMAC。

<a name="TPE"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `TPE`

> 名称：**TPE**

**建议场景**

TPE 是一种黑盒优化方法，可以使用在各种场景中，通常情况下都能得到较好的结果。 特别是在计算资源有限，只能运行少量 Trial 的情况。 大量的实验表明，TPE 的性能远远优于随机搜索。 [详细说明](./HyperoptTuner.md)

**classArgs 要求：**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

注意：为实现大规模并发 Trial，TPE 的并行性得到了优化。 有关优化原理或开启优化，参考 [TPE 文档](HyperoptTuner.md)。

**配置示例：**

```yaml
# config.yml
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
```

<br />

<a name="Random"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Random Search`

> 名称：**Random**

**建议场景**

随机搜索，可用于每个 Trial 运行时间不长（例如，能够非常快的完成，或者很快的被 Assessor 终止），并有充足计算资源的情况下。 如果要均衡的探索搜索空间，它也很有用。 随机搜索可作为搜索算法的基准线。 [详细说明](./HyperoptTuner.md)

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: Random
```

<br />

<a name="Anneal"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Anneal`

> 名称：**Anneal**

**Suggested scenario**

Anneal is suggested when each trial does not take very long and you have enough computation resources (very similar to Random Search). It's also useful when the variables in the search space can be sample from some prior distribution. [Detailed Description](./HyperoptTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: Anneal
  classArgs:
    optimize_mode: maximize
```

<br />

<a name="Evolution"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Naïve Evolution`

> 名称：**Evolution**

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

<br />

<a name="SMAC"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `SMAC`

> 名称：**SMAC**

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

<br />

<a name="Batch"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Batch Tuner`

> 名称：BatchTuner

**Suggested scenario**

If the configurations you want to try have been decided beforehand, you can list them in search space file (using `choice`) and run them using batch tuner. [Detailed Description](./BatchTuner.md)

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: BatchTuner
```

<br />

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

![](https://placehold.it/15/1589F0/000000?text=+) `Grid Search`

> 名称：**Grid Search**

**Suggested scenario**

Note that the only acceptable types within the search space are `choice`, `quniform`, and `randint`.

This is suggested when the search space is small. It's suggested when it is feasible to exhaustively sweep the whole search space. [Detailed Description](./GridsearchTuner.md)

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: GridSearch
```

<br />

<a name="Hyperband"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Hyperband`

> 名称：**Hyperband**

**Suggested scenario**

This is suggested when you have limited computational resources but have a relatively large search space. It performs well in scenarios where intermediate results can indicate good or bad final results to some extent. For example, when models that are more accurate early on in training are also more accurate later on. [Detailed Description](./HyperbandAdvisor.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
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

<br />

<a name="NetworkMorphism"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Network Morphism`

> 名称：**NetworkMorphism**

**Installation**

NetworkMorphism requires [PyTorch](https://pytorch.org/get-started/locally) and [Keras](https://keras.io/#installation), so users should install them first. The corresponding requirements file is [here](https://github.com/microsoft/nni/blob/master/examples/trials/network_morphism/requirements.txt).

**Suggested scenario**

This is suggested when you want to apply deep learning methods to your task but you have no idea how to choose or design a network. You may modify this [example](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism/cifar10/cifar10_keras.py) to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate, or optimizer. Currently, this tuner only supports the computer vision domain. [Detailed Description](./NetworkmorphismTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
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

<br />

<a name="MetisTuner"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Metis Tuner`

> 名称：**MetisTuner**

Note that the only acceptable types of search space types are `quniform`, `uniform`, `randint`, and numerical `choice`. Only numerical values are supported since the values will be used to evaluate the 'distance' between different points.

**Suggested scenario**

Similar to TPE and SMAC, Metis is a black-box tuner. If your system takes a long time to finish each trial, Metis is more favorable than other approaches such as random search. Furthermore, Metis provides guidance on subsequent trials. Here is an [example](https://github.com/Microsoft/nni/tree/master/examples/trials/auto-gbdt/search_space_metis.json) on the use of Metis. Users only need to send the final result, such as `accuracy`, to the tuner by calling the NNI SDK. [Detailed Description](./MetisTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*'maximize' or 'minimize', optional, default = 'maximize'*) - If 'maximize', the tuner will try to maximize metrics. 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: MetisTuner
  classArgs:
    optimize_mode: maximize
```

<br />

<a name="BOHB"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `BOHB Advisor`

> 名称：**BOHB**

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

![](https://placehold.it/15/1589F0/000000?text=+) `GP Tuner`

> 名称：**GPTuner**

Note that the only acceptable types within the search space are `randint`, `uniform`, `quniform`, `loguniform`, `qloguniform`, and numerical `choice`. Only numerical values are supported since the values will be used to evaluate the 'distance' between different points.

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

![](https://placehold.it/15/1589F0/000000?text=+) `PPO Tuner`

> 内置 Tuner 名称：**PPOTuner**

Note that the only acceptable types within the search space is `mutable_layer`. `optional_input_size` can only be 0, 1, or [0, 1].

**Suggested scenario**

PPOTuner is a Reinforcement Learning tuner based on the PPO algorithm. PPOTuner can be used when using the NNI NAS interface to do neural architecture search. In general, the Reinforcement Learning algorithm needs more computing resources, though the PPO algorithm is relatively more efficient than others. It's recommended to use this tuner when you have a large amount of computional resources available. You could try it on a very simple task, such as the [mnist-nas](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nas) example. [See details](./PPOTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*'maximize' or 'minimize'*) - If 'maximize', the tuner will try to maximize metrics. 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
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

![](https://placehold.it/15/1589F0/000000?text=+) `PBT Tuner`

> 内置 Tuner 名称：**PBTTuner**

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

## **参考和反馈**

* To [report a bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md) for this feature in GitHub;
* To [file a feature or improvement request](https://github.com/microsoft/nni/issues/new?template=enhancement.md) for this feature in GitHub;
* To know more about [Feature Engineering with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/FeatureEngineering/Overview.md);
* To know more about [NAS with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/NAS/Overview.md);
* To know more about [Model Compression with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/Compressor/Overview.md);