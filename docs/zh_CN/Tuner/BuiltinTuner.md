# 内置的超参调优 Tuner

NNI 提供了先进的调优算法，使用上也很简单。 下面是内置 Tuner 的简单介绍：

注意：点击 **Tuner 的名称**可看到 Tuner 的安装需求，建议的场景以及示例。 算法的详细说明在每个 Tuner 建议场景的最后。 [本文](../CommunitySharings/HpoComparision.md)对比了不同 Tuner 在几个问题下的不同效果。

当前支持的算法：

| Tuner（调参器）                               | 算法简介                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [**TPE**](#TPE)                          | Tree-structured Parzen Estimator (TPE) 是一种 sequential model-based optimization（SMBO，即基于序列模型优化）的方法。 SMBO 方法根据历史指标数据来按顺序构造模型，来估算超参的性能，随后基于此模型来选择新的超参。 [参考论文](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)                                                                                                                                                                                                                                                                                              |
| [**Random Search（随机搜索）**](#Random)       | 在超参优化时，随机搜索算法展示了其惊人的简单和效果。 建议当不清楚超参的先验分布时，采用随机搜索作为基准。 [参考论文](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)                                                                                                                                                                                                                                                                                                                                                                                                              |
| [**Anneal（退火算法）**](#Anneal)              | 这种简单的退火算法从先前的采样开始，会越来越靠近发现的最佳点取样。 此算法是随机搜索的简单变体，利用了反应曲面的平滑性。 退火率不是自适应的。                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [**Naïve Evolution（进化算法）**](#Evolution)  | Naïve Evolution（朴素进化算法）来自于 Large-Scale Evolution of Image Classifiers。 它会基于搜索空间随机生成一个种群。 在每一代中，会选择较好的结果，并对其下一代进行一些变异（例如，改动一个超参，增加或减少一层）。 朴素进化算法需要很多次的 Trial 才能有效，但它也非常简单，也很容易扩展新功能。 [参考论文](https://arxiv.org/pdf/1703.01041.pdf)                                                                                                                                                                                                                                                                                                           |
| [**SMAC**](#SMAC)                        | SMAC 基于 Sequential Model-Based Optimization (SMBO，即序列的基于模型优化方法)。 它利用使用过的结果好的模型（高斯随机过程模型），并将随机森林引入到 SMBO 中，来处理分类参数。 SMAC 算法包装了 Github 的 SMAC3。 注意：SMAC 需要通过 `nnictl package` 命令来安装。 [参考论文，](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) [Github 代码库](https://github.com/automl/SMAC3)                                                                                                                                                                                                                                              |
| [**Batch Tuner（批处理 Tuner）**](#Batch)     | Batch Tuner 能让用户简单的提供几组配置（如，超参选项的组合）。 当所有配置都执行完后，Experiment 即结束。 Batch Tuner 仅支持 choice 类型。                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [**Grid Search（遍历搜索）**](#GridSearch)     | Grid Search 会穷举定义在搜索空间文件中的所有超参组合。 遍历搜索可以使用的类型有 choice, quniform, randint。                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [**Hyperband**](#Hyperband)              | Hyperband tries to use limited resources to explore as many configurations as possible and returns the most promising ones as a final result. The basic idea is to generate many configurations and run them for a small number of trials. The half least-promising configurations are thrown out, the remaining are further trained along with a selection of new configurations. The size of these populations is sensitive to resource constraints (e.g. allotted search time). [Reference Paper](https://arxiv.org/pdf/1603.06560.pdf) |
| [**Network Morphism**](#NetworkMorphism) | Network Morphism provides functions to automatically search for deep learning architectures. It generates child networks that inherit the knowledge from their parent network which it is a morph from. This includes changes in depth, width, and skip-connections. Next, it estimates the value of a child network using historic architecture and metric pairs. Then it selects the most promising one to train. [Reference Paper](https://arxiv.org/abs/1806.10282)                                                                    |
| [**Metis Tuner**](#MetisTuner)           | 大多数调参工具仅仅预测最优配置，而 Metis 的优势在于有两个输出：(a) 最优配置的当前预测结果， 以及 (b) 下一次 Trial 的建议。 它不进行随机取样。 大多数工具假设训练集没有噪声数据，但 Metis 会知道是否需要对某个超参重新采样。 [参考论文](https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/)                                                                                                                                                                                                                                                                                            |
| [**BOHB**](#BOHB)                        | BOHB is a follow-up work to Hyperband. Hyperband 在生成新的配置时，没有利用已有的 Trial 结果，而本算法利用了 Trial 结果。 BOHB 中，HB 表示 Hyperband，BO 表示贝叶斯优化（Byesian Optimization）。 BOHB 会建立多个 TPE 模型，从而利用已完成的 Trial 生成新的配置。 [参考论文](https://arxiv.org/abs/1807.01774)                                                                                                                                                                                                                                                                                                    |
| [**GP Tuner**](#GPTuner)                 | Gaussian Process（高斯过程） Tuner 是序列化的基于模型优化（SMBO）的方法，并使用了高斯过程来替代。 [参考论文](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)，[Github 库](https://github.com/fmfn/BayesianOptimization)                                                                                                                                                                                                                                                                                                                          |
| [**PPO Tuner**](#PPOTuner)               | PPO Tuner 是基于 PPO 算法的强化学习 Tuner。 [参考论文](https://arxiv.org/abs/1707.06347)                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

## 用法

Using a built-in tuner provided by the NNI SDK requires one to declare the **builtinTunerName** and **classArgs** in the `config.yml` file. In this part, we will introduce each tuner along with information about usage and suggested scenarios, classArg requirements, and an example configuration.

注意：参考示例中的格式来创建新的 `config.yml` 文件。 Some built-in tuners need to be installed using `nnictl package`, like SMAC.

<a name="TPE"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `TPE`

> 名称：**TPE**

**建议场景**

TPE 是一种黑盒优化方法，可以使用在各种场景中，通常情况下都能得到较好的结果。 Especially when you have limited computation resources and can only try a small number of trials. From a large amount of experiments, we found that TPE is far better than Random Search. [详细说明](./HyperoptTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.

Note: We have optimized the parallelism of TPE for large-scale trial concurrency. 有关优化原理或开启优化，参考 [TPE 文档](HyperoptTuner.md)。

**Example Configuration:**

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

Random search is suggested when each trial does not take very long (e.g., each trial can be completed very quickly, or early stopped by the assessor), and you have enough computational resources. It's also useful if you want to uniformly explore the search space. Random Search can be considered a baseline search algorithm. [详细说明](./HyperoptTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.

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

**建议场景**

Anneal is suggested when each trial does not take very long and you have enough computation resources (very similar to Random Search). It's also useful when the variables in the search space can be sample from some prior distribution. [详细说明](./HyperoptTuner.md)

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

<br />

<a name="Evolution"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Naïve Evolution`

> 名称：**Evolution**

**建议场景**

Its computational resource requirements are relatively high. Specifically, it requires a large initial population to avoid falling into a local optimum. 如果 Trial 时间很短，或者使用了 Assessor，就非常适合此算法。 It is also suggested when your trial code supports weight transfer; that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training process. [详细说明](./EvolutionTuner.md)

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

**当前 SMAC 不支持在 WIndows 下运行。 For the specific reason, please refer to this [GitHub issue](https://github.com/automl/SMAC3/issues/483).**

**安装**

SMAC needs to be installed by following command before the first usage. 注意：SMAC 依赖于 `swig`，Ubuntu 下可通过 `apt` 命令来安装 `swig`。

```bash
nnictl package install --name=SMAC
```

**建议场景**

Similar to TPE, SMAC is also a black-box tuner that can be tried in various scenarios and is suggested when computational resources are limited. It is optimized for discrete hyperparameters, thus, it's suggested when most of your hyperparameters are discrete. [详细说明](./SmacTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **config_dedup** (*True 或 False, 可选, 默认为 False*) - 如果为 True，则 Tuner 不会生成重复的配置。 If False, a configuration may be generated twice, but it is rare for a relatively large search space.

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

**建议场景**

If the configurations you want to try have been decided beforehand, you can list them in search space file (using `choice`) and run them using batch tuner. [详细说明](./BatchTuner.md)

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

**建议场景**

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

**建议场景**

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

<br />

<a name="NetworkMorphism"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Network Morphism`

> 名称：**NetworkMorphism**

**安装**

NetworkMorphism 需要先安装 [PyTorch](https://pytorch.org/get-started/locally) 和 [Keras](https://keras.io/#installation) 才能使用。 对应的 requirements 文件在[这里](https://github.com/microsoft/nni/blob/master/examples/trials/network_morphism/requirements.txt)。

**建议场景**

This is suggested when you want to apply deep learning methods to your task but you have no idea how to choose or design a network. You may modify this [example](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism/cifar10/cifar10_keras.py) to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate, or optimizer. Currently, this tuner only supports the computer vision domain. [Detailed Description](./NetworkmorphismTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **task** (*('cv'), optional, default = 'cv'*) - The domain of the experiment. For now, this tuner only supports the computer vision (CV) domain.
* **input_width** (*int, 可选, 默认为 = 32*) - 输入图像的宽度
* **input_channel** (*int, 可选, 默认为 3*) - 输入图像的通道数
* **n_output_node** (*int, 可选, 默认为 10*) - 输出分类的数量

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

Note that the only acceptable types of search space types are `quniform`, `uniform`, `randint`, and numerical `choice`. 因为数值会被用来评估点之间的距离，所以只支持数值。

**建议场景**

与 TPE 和 SMAC 类似，Metis 是黑盒 Tuner。 如果系统需要很长时间才能完成一次 Trial，Metis 就比随机搜索等其它方法要更合适。 Furthermore, Metis provides guidance on subsequent trials. Here is an [example](https://github.com/Microsoft/nni/tree/master/examples/trials/auto-gbdt/search_space_metis.json) on the use of Metis. Users only need to send the final result, such as `accuracy`, to the tuner by calling the NNI SDK. [详细说明](./MetisTuner.md)

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

<br />

<a name="BOHB"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `BOHB Adivisor`

> 名称：**BOHB**

**安装**

BOHB advisor requires [ConfigSpace](https://github.com/automl/ConfigSpace) package. ConfigSpace can be installed using the following command.

```bash
nnictl package install --name=BOHB
```

**建议场景**

Similar to Hyperband, BOHB is suggested when you have limited computational resources but have a relatively large search space. It performs well in scenarios where intermediate results can indicate good or bad final results to some extent. In this case, it may converge to a better configuration than Hyperband due to its usage of Bayesian optimization. [详细说明](./BohbAdvisor.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will try to maximize metrics. If 'minimize', tuner will try to minimize metrics.
* **min_budget** (*int, optional, default = 1*) - The smallest budget to assign to a trial job, (budget can be the number of mini-batches or epochs). 该参数必须为正数。
* **max_budget** (*int, optional, default = 3*) - The largest budget to assign to a trial job, (budget can be the number of mini-batches or epochs). 该参数必须大于“min_budget”。
* **eta** (*整数, 可选项, 默认值为3*) - 在每次迭代中，执行完整的“连续减半”算法。 在这里，当一个使用相同计算资源的子集结束后，选择表现前 1/eta 好的参数，给予更高的优先级，进入下一轮比较（会获得更多计算资源）。 该参数必须大于等于 2。
* **min_points_in_model**(*整数, 可选项, 默认值为None*): 建立核密度估计（KDE）要求的最小观察到的点。 Default 'None' means dim+1; when the number of completed trials in this budget is equal to or larger than `max{dim+1, min_points_in_model}`, BOHB will start to build a KDE model of this budget then use said KDE model to guide configuration selection. Needs to be positive. (dim means the number of hyperparameters in search space)
* **top_n_percent**(*int, optional, default = 15*): percentage (between 1 and 99) of the observations which are considered good. 区分表现好的点与坏的点是为了建立树形核密度估计模型。 For example, if you have 100 observed trials and top_n_percent is 15, then the top 15% of points will be used for building the good points models "l(x)". The remaining 85% of points will be used for building the bad point models "g(x)".
* **num_samples** (*整数, 可选项, 默认值为64*): 用于优化 EI 值的采样个数（默认值为64）。 In this case, we will sample "num_samples" points and compare the result of l(x)/g(x). Then we will return the one with the maximum l(x)/g(x) value as the next configuration if the optimize_mode is `maximize`. Otherwise, we return the smallest one.
* **random_fraction**(*浮点数, 可选项, 默认值为0.33*): 使用模型的先验（通常是均匀）来随机采样的比例。
* **bandwidth_factor**(*float, optional, default = 3.0*): to encourage diversity, the points proposed to optimize EI are sampled from a 'widened' KDE where the bandwidth is multiplied by this factor. We suggest using the default value if you are not familiar with KDE.
* **min_bandwidth**(< 1>float, 可选, 默认值 = 0.001 </em>): 为了保持多样性, 即使所有好的样本对其中一个参数具有相同的值，使用最小带宽 (默认值: 1e-3) 而不是零。 We suggest using the default value if you are not familiar with KDE.

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

Note that the only acceptable types within the search space are `randint`, `uniform`, `quniform`, `loguniform`, `qloguniform`, and numerical `choice`. 因为数值会被用来评估点之间的距离，所以只支持数值。

**建议场景**

As a strategy in a Sequential Model-based Global Optimization (SMBO) algorithm, GP Tuner uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) to solve and common tools can be employed to solve it. Therefore, GP Tuner is most adequate for situations where the function to be optimized is very expensive to evaluate. GP can be used when computational resources are limited. However, GP Tuner has a computational cost that grows at *O(N^3)* due to the requirement of inverting the Gram matrix, so it's not suitable when lots of trials are needed. [详细说明](./GPTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*'maximize' or 'minimize', optional, default = 'maximize'*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **utility** (*'ei', 'ucb' or 'poi', optional, default = 'ei'*) - The utility function (acquisition function). 'ei', 'ucb', and 'poi' correspond to 'Expected Improvement', 'Upper Confidence Bound', and 'Probability of Improvement', respectively. 
* **kappa** (*float, optional, default = 5*) - Used by the 'ucb' utility function. The bigger `kappa` is, the more exploratory the tuner will be.
* **xi** (*float, optional, default = 0*) - Used by the 'ei' and 'poi' utility functions. The bigger `xi` is, the more exploratory the tuner will be.
* **nu** (*float, optional, default = 2.5*) - Used to specify the Matern kernel. nu 越小，近似函数的平滑度越低。
* **alpha** (*float, optional, default = 1e-6*) - Used to specify the Gaussian Process Regressor. Larger values correspond to an increased noise level in the observations.
* **cold_start_num** (*int, optional, default = 10*) - Number of random explorations to perform before the Gaussian Process. 随机探索可帮助提高探索空间的广泛性。
* **selection_num_warm_up** (*int, optional, default = 1e5*) - Number of random points to evaluate when getting the point which maximizes the acquisition function.
* **selection_num_starting_points** (*int, 可选, 默认为 250*) - 预热后，从随机七十点运行 L-BFGS-B 的次数。

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

Note that the only acceptable types within the search space is `mutable_layer`. `optional_input_size` 只能是 0, 1, 或 [0, 1]。

**建议场景**

PPOTuner is a Reinforcement Learning tuner based on the PPO algorithm. PPOTuner can be used when using the NNI NAS interface to do neural architecture search. In general, the Reinforcement Learning algorithm needs more computing resources, though the PPO algorithm is relatively more efficient than others. It's recommended to use this tuner when you have a large amount of computional resources available. You could try it on a very simple task, such as the [mnist-nas](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nas) example. [查看详细信息](./PPOTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*'maximize' or 'minimize'*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **trials_per_update** (*int, 可选, 默认为 20*) - 每次更新的 Trial 数量。 此数字必须可被 minibatch_size 整除。 推荐将 `trials_per_update` 设为 `trialConcurrency` 的倍数，以提高 Trial 的并发效率。
* **epochs_per_update** (*int, 可选, 默认为 4*) - 每次更新的 Epoch 数量。
* **minibatch_size** (*int, 可选, 默认为 4*) - mini-batch 大小 (即每个 mini-batch 的 Trial 数量)。 Note that trials_per_update must be divisible by minibatch_size.
* **ent_coef** (*float, 可选, 默认为 0.0*) - 优化目标中的 Policy entropy coefficient。
* **lr** (*float, optional, default = 3e-4*) - Learning rate of the model (lstm network); constant.
* **vf_coef** (*float, 可选, 默认为 0.5*) - Value function loss coefficient in the optimization objective.
* **max_grad_norm** (*float, 可选, 默认为 0.5*) - Gradient norm clipping coefficient.
* **gamma** (*float, 可选, 默认为 0.99*) - Discounting factor.
* **lam** (*float, 可选, 默认为 0.95*) - Advantage estimation discounting factor (论文中的 lambda).
* **cliprange** (*float, 可选, 默认为 0.2*) - PPO 算法的 cliprange, 为常数。

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: PPOTuner
  classArgs:
    optimize_mode: maximize
```

## **参考和反馈**

* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)；
* 了解 NNI 中[特征工程的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/FeatureEngineering/Overview.md)；
* 了解 NNI 中[ NAS 的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/NAS/Overview.md)；
* 了解 NNI 中[模型自动压缩的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Compressor/Overview.md)；