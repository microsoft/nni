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

### TPE

> 名称：**TPE**

**Suggested scenario**

TPE, as a black-box optimization, can be used in various scenarios and shows good performance in general. Especially when you have limited computation resources and can only try a small number of trials. From a large amount of experiments, we found that TPE is far better than Random Search. [Detailed Description](./HyperoptTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

Note: We have optimized the parallelism of TPE for large-scale trial concurrency. For the principle of optimization or turn-on optimization, please refer to [TPE document](./HyperoptTuner.md).

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

### Random Search

> 名称：**Random**

**Suggested scenario**

Random search is suggested when each trial does not take very long (e.g., each trial can be completed very quickly, or early stopped by the assessor), and you have enough computational resources. It's also useful if you want to uniformly explore the search space. Random Search can be considered a baseline search algorithm. [Detailed Description](./HyperoptTuner.md)

**Example Configuration:**

```yaml
# config.yml
tuner:
  builtinTunerName: Random
```

<br />

<a name="Anneal"></a>

### Anneal

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

### Naïve Evolution

> 名称：**Evolution**

**Suggested scenario**

Its computational resource requirements are relatively high. Specifically, it requires a large initial population to avoid falling into a local optimum. If your trial is short or leverages assessor, this tuner is a good choice. It is also suggested when your trial code supports weight transfer; that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training process. [Detailed Description](./EvolutionTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

* **population_size** (*int 类型 (需要大于 0), 可选项, 默认值为 20*) - 表示遗传 Tuner 中的初始种群（Trial 数量）。 建议 `population_size` 比 `concurrency` 取值更大，这样能充分利用算法（至少要等于 `concurrency`，否则 Tuner 在生成第一代参数的时候就会失败）。

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

### SMAC

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

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **config_dedup** (*True 或 False, 可选, 默认为 False*) - 如果为 True，则 Tuner 不会生成重复的配置。 如果为 False，则配置可能会重复生成，但对于相对较大的搜索空间，此概率较小。

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

### Batch Tuner

> 名称：BatchTuner

**Suggested scenario**

If the configurations you want to try have been decided beforehand, you can list them in search space file (using `choice`) and run them using batch tuner. [Detailed Description](./BatchTuner.md)

**配置示例：**

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

### Grid Search

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

### Hyperband

> 名称：**Hyperband**

**Suggested scenario**

This is suggested when you have limited computational resources but have a relatively large search space. It performs well in scenarios where intermediate results can indicate good or bad final results to some extent. For example, when models that are more accurate early on in training are also more accurate later on. [Detailed Description](./HyperbandAdvisor.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **R** (*int, 可选, 默认为 60*) - 分配给 Trial 的最大资源（可以是 mini-batches 或 epochs 的数值）。 每个 Trial 都需要用 TRIAL_BUDGET 来控制运行的步数。
* **eta** (*int, 可选, 默认为 3*) - `(eta-1)/eta` 是丢弃 Trial 的比例。

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

### Network Morphism

> 名称：**NetworkMorphism**

**Installation**

NetworkMorphism requires [PyTorch](https://pytorch.org/get-started/locally) and [Keras](https://keras.io/#installation), so users should install them first. The corresponding requirements file is [here](https://github.com/microsoft/nni/blob/master/examples/trials/network_morphism/requirements.txt).

**建议场景**

This is suggested when you want to apply deep learning methods to your task but you have no idea how to choose or design a network. You may modify this [example](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism/cifar10/cifar10_keras.py) to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate, or optimizer. Currently, this tuner only supports the computer vision domain. [Detailed Description](./NetworkmorphismTuner.md)

**classArgs 要求：**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **task** (*('cv'), 可选, 默认为 'cv'*) - 实验的领域。 当前，此 Tuner 仅支持计算机视觉（cv）领域。
* **input_width** (*int, 可选, 默认为 = 32*) - 输入图像的宽度
* **input_channel** (*int, 可选, 默认为 3*) - 输入图像的通道数
* **n_output_node** (*int, 可选, 默认为 10*) - 输出分类的数量

**配置示例：**

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

### Metis Tuner

> 名称：**MetisTuner**

Note that the only acceptable types of search space types are `quniform`, `uniform`, `randint`, and numerical `choice`. Only numerical values are supported since the values will be used to evaluate the 'distance' between different points.

**Suggested scenario**

Similar to TPE and SMAC, Metis is a black-box tuner. If your system takes a long time to finish each trial, Metis is more favorable than other approaches such as random search. Furthermore, Metis provides guidance on subsequent trials. Here is an [example](https://github.com/Microsoft/nni/tree/master/examples/trials/auto-gbdt/search_space_metis.json) on the use of Metis. Users only need to send the final result, such as `accuracy`, to the tuner by calling the NNI SDK. [Detailed Description](./MetisTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*'maximize' 或 'minimize', 可选项, 默认值为 'maximize'*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

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

### BOHB Advisor

> 名称：**BOHB**

**Installation**

BOHB advisor requires [ConfigSpace](https://github.com/automl/ConfigSpace) package. ConfigSpace can be installed using the following command.

```bash
nnictl package install --name=BOHB
```

**Suggested scenario**

Similar to Hyperband, BOHB is suggested when you have limited computational resources but have a relatively large search space. It performs well in scenarios where intermediate results can indicate good or bad final results to some extent. In this case, it may converge to a better configuration than Hyperband due to its usage of Bayesian optimization. [Detailed Description](./BohbAdvisor.md)

**classArgs Requirements:**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 会试着最大化指标。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **min_budget** (*整数, 可选项, 默认值为 1*) - 运行一个试验给予的最低计算资源（budget），这里的计算资源通常使用mini-batches 或者 epochs。 该参数必须为正数。
* **max_budget** (*整数, 可选项, 默认值为 3*) - 运行一个试验给予的最大计算资源（budget），这里的计算资源通常使用 mini-batches 或者 epochs。 该参数必须大于“min_budget”。
* **eta** (*整数, 可选项, 默认值为3*) - 在每次迭代中，执行完整的“连续减半”算法。 在这里，当一个使用相同计算资源的子集结束后，选择表现前 1/eta 好的参数，给予更高的优先级，进入下一轮比较（会获得更多计算资源）。 该参数必须大于等于 2。
* **min_points_in_model**(*整数, 可选项, 默认值为None*): 建立核密度估计（KDE）要求的最小观察到的点。 默认值 None 表示 dim+1，当在该计算资源（budget）下试验过的参数已经大于等于`max{dim+1, min_points_in_model}` 时，BOHB 将会开始建立这个计算资源（budget）下对应的核密度估计（KDE）模型，然后用这个模型来指导参数的选取。 该参数必须为正数。 (dim 表示搜索空间中超参的数量)
* **top_n_percent**(*整数, 可选, 默认值为 15*): 认为观察点为好点的百分数 (在 1 到 99 之间)。 区分表现好的点与坏的点是为了建立树形核密度估计模型。 例如，如果有 100 个观察到的 Trial，top_n_percent 为 15，则前 15% 的点将用于构建好点模型 "l(x)"。 其余 85% 的点将用于构建坏点模型 "g(x)"。
* **num_samples** (*整数, 可选项, 默认值为64*): 用于优化 EI 值的采样个数（默认值为64）。 在这种情况下，将对 "num_samples" 点进行采样，并比较 l(x)/g(x) 的结果。 然后，如果 optimize_mode 是 `maximize`，就会返回其中 l(x)/g(x) 值最大的点作为下一个配置参数。 否则，使用值最小的点。
* **random_fraction**(*浮点数, 可选项, 默认值为0.33*): 使用模型的先验（通常是均匀）来随机采样的比例。
* **bandwidth_factor**(*浮点数, 可选, 默认值为 3.0 *): 为了鼓励多样性，把优化 EI 的点加宽，即把 KDE 中采样的点乘以这个因子，从而增加 KDE 中的带宽。 如果不熟悉 KDE，建议使用默认值。
* **min_bandwidth**(< 1>float, 可选, 默认值 = 0.001 </em>): 为了保持多样性, 即使所有好的样本对其中一个参数具有相同的值，使用最小带宽 (默认值: 1e-3) 而不是零。 如果不熟悉 KDE，建议使用默认值。

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

> 名称：**GPTuner**

Note that the only acceptable types within the search space are `randint`, `uniform`, `quniform`, `loguniform`, `qloguniform`, and numerical `choice`. Only numerical values are supported since the values will be used to evaluate the 'distance' between different points.

**Suggested scenario**

As a strategy in a Sequential Model-based Global Optimization (SMBO) algorithm, GP Tuner uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) to solve and common tools can be employed to solve it. Therefore, GP Tuner is most adequate for situations where the function to be optimized is very expensive to evaluate. GP can be used when computational resources are limited. However, GP Tuner has a computational cost that grows at *O(N^3)* due to the requirement of inverting the Gram matrix, so it's not suitable when lots of trials are needed. [Detailed Description](./GPTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*'maximize' 或 'minimize', 可选项, 默认值为 'maximize'*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **utility** (*'ei', 'ucb' 或 'poi', 可选, 默认值为 'ei'*) - 工具函数的类型（采集函数）。 'ei', 'ucb' 和 'poi' 分别对应 '期望的改进（Expected Improvement）', '上限置信度边界（Upper Confidence Bound）' 和 '改进概率（Probability of Improvement）'。 
* **kappa** (*float, 可选, 默认值为 5*) - 用于 'ucb' 函数。 `kappa` 越大， Tuner 的探索性越强。
* **xi** (*float, 可选, 默认为 0*) - 用于 'ei' 和 'poi' 工具函数。 `xi` 越大， Tuner 的探索性越强。
* **nu** (*float, 可选, 默认为 2.5*) - 用于指定 Matern 核。 nu 越小，近似函数的平滑度越低。
* **alpha** (*float, 可选, 默认值为 1e-6*) - 用于高斯过程回归器。 值越大，表示观察中的噪声水平越高。
* **cold_start_num** (*int, 可选, 默认值为 10*) - 在高斯过程前执行随机探索的数量。 随机探索可帮助提高探索空间的广泛性。
* **selection_num_warm_up** (*int, 可选, 默认为 1e5*) - 用于获得最大采集函数而评估的随机点数量。
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

### PPO Tuner

> 内置 Tuner 名称：**PPOTuner**

Note that the only acceptable types within the search space are `layer_choice` and `input_choice`. For `input_choice`, `n_chosen` can only be 0, 1, or [0, 1]. Note, the search space file for NAS is usually automatically generated through the command [`nnictl ss_gen`](../Tutorial/Nnictl.md).

**Suggested scenario**

PPOTuner is a Reinforcement Learning tuner based on the PPO algorithm. PPOTuner can be used when using the NNI NAS interface to do neural architecture search. In general, the Reinforcement Learning algorithm needs more computing resources, though the PPO algorithm is relatively more efficient than others. It's recommended to use this tuner when you have a large amount of computional resources available. You could try it on a very simple task, such as the [mnist-nas](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nas) example. [See details](./PPOTuner.md)

**classArgs Requirements:**

* **optimize_mode** (*'maximize' 或 'minimize'*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **trials_per_update** (*int, 可选, 默认为 20*) - 每次更新的 Trial 数量。 此数字必须可被 minibatch_size 整除。 推荐将 `trials_per_update` 设为 `trialConcurrency` 的倍数，以提高 Trial 的并发效率。
* **epochs_per_update** (*int, 可选, 默认为 4*) - 每次更新的 Epoch 数量。
* **minibatch_size** (*int, 可选, 默认为 4*) - mini-batch 大小 (即每个 mini-batch 的 Trial 数量)。 注意，trials_per_update 必须可被 minibatch_size 整除。
* **ent_coef** (*float, 可选, 默认为 0.0*) - 优化目标中的 Policy entropy coefficient。
* **lr** (*float, 可选, 默认为 3e-4*) - 模型的学习率（LSTM 网络），为常数。
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

<a name="PBTTuner"></a>

### PBT Tuner

> 内置 Tuner 名称：**PBTTuner**

**Suggested scenario**

Population Based Training (PBT) bridges and extends parallel search methods and sequential optimization methods. It requires relatively small computation resource, by inheriting weights from currently good-performing ones to explore better ones periodically. With PBTTuner, users finally get a trained model, rather than a configuration that could reproduce the trained model by training the model from scratch. This is because model weights are inherited periodically through the whole search process. PBT can also be seen as a training approach. If you don't need to get a specific configuration, but just expect a good model, PBTTuner is a good choice. [See details](./PBTTuner.md)

**classArgs requirements:**

* **optimize_mode** (*'maximize' 或 'minimize'*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **all_checkpoint_dir** (*str, 可选, 默认为 None*) - Trial 保存读取检查点的目录，如果不指定，其为 "~/nni/checkpoint/<exp-id>". 注意，如果 Experiment 不是本机模式，用户需要提供能被所有 Trial 所访问的共享存储。
* **population_size** (*int, 可选, 默认为 10*) - 种群的 Trial 数量。 每个步骤有此数量的 Trial。 在 NNI 的实现中，一步表示每个 Trial 运行一定次数 Epoch，此 Epoch 的数量由用户来指定。
* **factors** (*tuple, 可选, 默认为 (1.2, 0.8)*) - 超参变动量的因子。
* **fraction** (*float, 可选, 默认为 0.2*) - 选择的最低和最高 Trial 的比例。

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

* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)；
* 了解 NNI 中[特征工程的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/FeatureEngineering/Overview.md)；
* 了解 NNI 中[ NAS 的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/NAS/Overview.md)；
* 了解 NNI 中[模型自动压缩的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Compressor/Overview.md)；