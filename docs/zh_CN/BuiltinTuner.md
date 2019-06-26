# 内置 Tuner

NNI 提供了先进的调优算法，使用上也很简单。 下面是内置 Tuner 的简单介绍：

注意：点击 **Tuner 的名称**可看到 Tuner 的安装需求，建议的场景以及示例。 算法的详细说明在每个 Tuner 建议场景的最后。 [本文](./CommunitySharings/HpoComparision.md)对比了不同 Tuner 在几个问题下的不同效果。

当前支持的 Tuner：

| Tuner（调参器）                               | 算法简介                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [**TPE**](#TPE)                          | Tree-structured Parzen Estimator (TPE) 是一种 sequential model-based optimization（SMBO，即基于序列模型优化）的方法。 SMBO 方法根据历史指标数据来按顺序构造模型，来估算超参的性能，随后基于此模型来选择新的超参。 [参考论文](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)                                                                                                                                                                                                                                                                                                          |
| [**Random Search（随机搜索）**](#Random)       | 在超参优化时，随机搜索算法展示了其惊人的简单和效果。 建议当不清楚超参的先验分布时，采用随机搜索作为基准。 [参考论文](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [**Anneal（退火算法）**](#Anneal)              | 这种简单的退火算法从先前的采样开始，会越来越靠近发现的最佳点取样。 此算法是随机搜索的简单变体，利用了反应曲面的平滑性。 退火率不是自适应的。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [**Naive Evolution（进化算法）**](#Evolution)  | 朴素进化算法来自于大规模图像分类进化。 它会基于搜索空间随机生成一个种群。 在每一代中，会选择较好的结果，并对其下一代进行一些变异（例如，改动一个超参，增加或减少一层）。 进化算法需要很多次 Trial 才能有效，但它也非常简单，也很容易扩展新功能。 [参考论文](https://arxiv.org/pdf/1703.01041.pdf)                                                                                                                                                                                                                                                                                                                                                                             |
| [**SMAC**](#SMAC)                        | SMAC 基于 Sequential Model-Based Optimization (SMBO，即序列的基于模型优化方法)。 它利用使用过的结果好的模型（高斯随机过程模型），并将随机森林引入到 SMBO 中，来处理分类参数。 SMAC 算法包装了 Github 的 SMAC3。 注意：SMAC 需要通过 `nnictl package` 命令来安装。 [参考论文，](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) [Github 代码库](https://github.com/automl/SMAC3)                                                                                                                                                                                                                                                          |
| [**Batch Tuner（批量调参器）**](#Batch)         | Batch Tuner 能让用户简单的提供几组配置（如，超参选项的组合）。 当所有配置都执行完后，Experiment 即结束。 Batch Tuner 仅支持 choice 类型。                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [**Grid Search（遍历搜索）**](#GridSearch)     | Grid Search 会穷举定义在搜索空间文件中的所有超参组合。 网格搜索可以使用的类型有 choice, quniform, qloguniform。 quniform 和 qloguniform 中的数值 q 具有特别的含义（不同于搜索空间文档中的说明）。 它表示了在最高值与最低值之间采样的值的数量。                                                                                                                                                                                                                                                                                                                                                                                             |
| [**Hyperband**](#Hyperband)              | Hyperband 试图用有限的资源来探索尽可能多的组合，并发现最好的结果。 它的基本思路是生成大量的配置，并使用少量的资源来找到有可能好的配置，然后继续训练找到其中更好的配置。 [参考论文](https://arxiv.org/pdf/1603.06560.pdf)                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [**Network Morphism**](#NetworkMorphism) | Network Morphism 提供了深度学习模型的自动架构搜索功能。 每个子网络都继承于父网络的知识和形态，并变换网络的不同形态，包括深度，宽度，跨层连接（skip-connection）。 然后使用历史的架构和指标，来估计子网络的值。 然后会选择最有希望的模型进行训练。 [参考论文](https://arxiv.org/abs/1806.10282)                                                                                                                                                                                                                                                                                                                                                                    |
| [**Metis Tuner**](#MetisTuner)           | Metis offers the following benefits when it comes to tuning parameters: While most tools only predict the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guesswork. While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter. [Reference Paper](https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/) |
| [**BOHB**](#BOHB)                        | BOHB is a follow-up work of Hyperband. It targets the weakness of Hyperband that new configurations are generated randomly without leveraging finished trials. For the name BOHB, HB means Hyperband, BO means Bayesian Optimization. BOHB leverages finished trials by building multiple TPE models, a proportion of new configurations are generated through these models. [Reference Paper](https://arxiv.org/abs/1807.01774)                                                                                                                       |
| [**GP Tuner**](#GPTuner)                 | Gaussian Process Tuner is a sequential model-based optimization (SMBO) approach with Gaussian Process as the surrogate. [Reference Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf), [Github Repo](https://github.com/fmfn/BayesianOptimization)                                                                                                                                                                                                                                                              |

## Usage of Built-in Tuners

Use built-in tuner provided by NNI SDK requires to declare the **builtinTunerName** and **classArgs** in `config.yml` file. In this part, we will introduce the detailed usage about the suggested scenarios, classArg requirements and example for each tuner.

Note: Please follow the format when you write your `config.yml` file. Some built-in tuner need to be installed by `nnictl package`, like SMAC.

<a name="TPE"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `TPE`

> Built-in Tuner Name: **TPE**

**Suggested scenario**

TPE, as a black-box optimization, can be used in various scenarios and shows good performance in general. Especially when you have limited computation resource and can only try a small number of trials. From a large amount of experiments, we could found that TPE is far better than Random Search. [Detailed Description](./HyperoptTuner.md)

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**Usage example:**

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

> Built-in Tuner Name: **Random**

**Suggested scenario**

Random search is suggested when each trial does not take too long (e.g., each trial can be completed very soon, or early stopped by assessor quickly), and you have enough computation resource. Or you want to uniformly explore the search space. Random Search could be considered as baseline of search algorithm. [Detailed Description](./HyperoptTuner.md)

**Requirement of classArg:**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: Random
```

<br />

<a name="Anneal"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Anneal`

> Built-in Tuner Name: **Anneal**

**Suggested scenario**

Anneal is suggested when each trial does not take too long, and you have enough computation resource(almost same with Random Search). Or the variables in search space could be sample from some prior distribution. [Detailed Description](./HyperoptTuner.md)

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: Anneal
  classArgs:
    optimize_mode: maximize
```

<br />

<a name="Evolution"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Naive Evolution`

> Built-in Tuner Name: **Evolution**

**Suggested scenario**

Its requirement of computation resource is relatively high. Specifically, it requires large initial population to avoid falling into local optimum. If your trial is short or leverages assessor, this tuner is a good choice. And, it is more suggested when your trial code supports weight transfer, that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training progress. [Detailed Description](./EvolutionTuner.md)

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: Evolution
  classArgs:
    optimize_mode: maximize
```

<br />

<a name="SMAC"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `SMAC`

> Built-in Tuner Name: **SMAC**

**Please note that SMAC doesn't support running on windows currently. The specific reason can be referred to this [GitHub issue](https://github.com/automl/SMAC3/issues/483).**

**Installation**

SMAC need to be installed by following command before first use.

```bash
nnictl package install --name=SMAC
```

**Suggested scenario**

Similar to TPE, SMAC is also a black-box tuner which can be tried in various scenarios, and is suggested when computation resource is limited. It is optimized for discrete hyperparameters, thus, suggested when most of your hyperparameters are discrete. [Detailed Description](./SmacTuner.md)

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**Usage example**

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

> Built-in Tuner Name: BatchTuner

**Suggested scenario**

If the configurations you want to try have been decided, you can list them in searchspace file (using `choice`) and run them using batch tuner. [Detailed Description](./BatchTuner.md)

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: BatchTuner
```

<br />

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

> Built-in Tuner Name: **Grid Search**

**Suggested scenario**

Note that the only acceptable types of search space are `choice`, `quniform`, `qloguniform`. **The number `q` in `quniform` and `qloguniform` has special meaning (different from the spec in [search space spec](./SearchSpaceSpec.md)). It means the number of values that will be sampled evenly from the range `low` and `high`.**

It is suggested when search space is small, it is feasible to exhaustively sweeping the whole search space. [Detailed Description](./GridsearchTuner.md)

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: GridSearch
```

<br />

<a name="Hyperband"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Hyperband`

> Built-in Advisor Name: **Hyperband**

**Suggested scenario**

It is suggested when you have limited computation resource but have relatively large search space. It performs well in the scenario that intermediate result (e.g., accuracy) can reflect good or bad of final result (e.g., accuracy) to some extent. [Detailed Description](./HyperbandAdvisor.md)

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **R** (*int, 可选, 默认为 60*) - 分配给 Trial 的最大资源（可以是 mini-batches 或 epochs 的数值）。 每个 Trial 都需要用 TRIAL_BUDGET 来控制运行的步数。
* **eta** (*int, 可选, 默认为 3*) - `(eta-1)/eta` 是丢弃 Trial 的比例。

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

<br />

<a name="NetworkMorphism"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Network Morphism`

> Built-in Tuner Name: **NetworkMorphism**

**Installation**

NetworkMorphism requires [PyTorch](https://pytorch.org/get-started/locally), so users should install it first.

**Suggested scenario**

It is suggested that you want to apply deep learning methods to your task (your own dataset) but you have no idea of how to choose or design a network. You modify the [example](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism/cifar10/cifar10_keras.py) to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate or optimizer. It is feasible for different tasks to find a good network architecture. Now this tuner only supports the computer vision domain. [Detailed Description](./NetworkmorphismTuner.md)

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **task** (*('cv'), 可选, 默认为 'cv'*) - 实验的领域，当前仅支持视觉（cv）。
* **input_width** (*int, 可选, 默认为 = 32*) - 输入图像的宽度
* **input_channel** (*int, 可选, 默认为 3*) - 输入图像的通道数
* **n_output_node** (*int, 可选, 默认为 10*) - 输出分类的数量

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

<br />

<a name="MetisTuner"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Metis Tuner`

> Built-in Tuner Name: **MetisTuner**

Note that the only acceptable types of search space are `choice`, `quniform`, `uniform` and `randint`.

**Suggested scenario**

Similar to TPE and SMAC, Metis is a black-box tuner. If your system takes a long time to finish each trial, Metis is more favorable than other approaches such as random search. Furthermore, Metis provides guidance on the subsequent trial. Here is an [example](https://github.com/Microsoft/nni/tree/master/examples/trials/auto-gbdt/search_space_metis.json) about the use of Metis. User only need to send the final result like `accuracy` to tuner, by calling the NNI SDK. [Detailed Description](./MetisTuner.md)

**Requirement of classArg**

* **optimize_mode** (*'maximize' 或 'minimize', 可选项, 默认值为 'maximize'*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**Usage example**

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

> Built-in Tuner Name: **BOHB**

**Installation**

BOHB advisor requires [ConfigSpace](https://github.com/automl/ConfigSpace) package, ConfigSpace need to be installed by following command before first use.

```bash
nnictl package install --name=BOHB
```

**Suggested scenario**

Similar to Hyperband, it is suggested when you have limited computation resource but have relatively large search space. It performs well in the scenario that intermediate result (e.g., accuracy) can reflect good or bad of final result (e.g., accuracy) to some extent. In this case, it may converges to a better configuration due to Bayesian optimization usage. [Detailed Description](./BohbAdvisor.md)

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **min_budget** (*整数, 可选项, 默认值为 1*) - 运行一个试验给予的最低计算资源（budget），这里的计算资源通常使用mini-batches 或者 epochs。 该参数必须为正数。
* **max_budget** (*整数, 可选项, 默认值为 3*) - 运行一个试验给予的最大计算资源（budget），这里的计算资源通常使用 mini-batches 或者 epochs。 该参数必须大于“min_budget”。
* **eta** (*整数, 可选项, 默认值为3*) - 在每次迭代中，执行完整的“连续减半”算法。 在这里，当一个使用相同计算资源的子集结束后，选择表现前 1/eta 好的参数，给予更高的优先级，进入下一轮比较（会获得更多计算资源）。 该参数必须大于等于 2。
* **min_points_in_model**(*整数, 可选项, 默认值为None*): 建立核密度估计（KDE）要求的最小观察到的点。 默认值 None 表示 dim+1，当在该计算资源（budget）下试验过的参数已经大于等于`max{dim+1, min_points_in_model}` 时，BOHB 将会开始建立这个计算资源（budget）下对应的核密度估计（KDE）模型，然后用这个模型来指导参数的选取。 该参数必须为正数。（dim 指的是搜索空间中超参数的维度）
* **top_n_percent**(*整数, 可选项, 默认值为15*): 认为观察点为好点的百分数(在 1 到 99 之间，默认值为 15)。 区分表现好的点与坏的点是为了建立树形核密度估计模型。 比如，如果观察到了100个点的表现情况，同时把 top_n_percent 设置为 15，那么表现最好的 15个点将会用于创建表现好的点的分布 "l(x)"，剩下的85个点将用于创建表现坏的点的分布 “g(x)”。
* **num_samples** (*整数, 可选项, 默认值为64*): 用于优化 EI 值的采样个数（默认值为64）。 在这个例子中，将根据 l(x) 的分布采样“num_samples”（默认值为64）个点。若优化的目标为最大化指标，则会返回其中 l(x)/g(x) 的值最大的点作为下一个试验的参数。 否则，使用值最小的点。
* **random_fraction**(*浮点数, 可选项, 默认值为0.33*): 使用模型的先验（通常是均匀）来随机采样的比例。
* **bandwidth_factor**(< 1>浮点数, 可选, 默认值为3.0 </em>): 为了鼓励多样性，把优化EI的点加宽，即把KDE中采样的点乘以这个因子，从而增加KDE中的带宽。 如果不熟悉 KDE，建议保留默认值。
* **min_bandwidth**(< 1>float, 可选, 默认值 = 0.001 </em>): 为了保持多样性, 即使所有好的样本对其中一个参数具有相同的值，使用最小带宽 (默认值: 1e-3) 而不是零。 如果不熟悉 KDE，建议保留默认值。

*Please note that currently float type only support decimal representation, you have to use 0.333 instead of 1/3 and 0.001 instead of 1e-3.*

**Usage example**

```yml
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

> Built-in Tuner Name: **GPTuner**

Note that the only acceptable types of search space are `choice`, `randint`, `uniform`, `quniform`, `loguniform`, `qloguniform`.

**Suggested scenario**

As a strategy in Sequential Model-based Global Optimization(SMBO) algorithm, GP Tuner uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) and common tools can be employed. Therefore GP Tuner is most adequate for situations where the function to be optimized is a very expensive endeavor. GP can be used when the computation resource is limited. While GP Tuner has a computational cost that grows at *O(N^3)* due to the requirement of inverting the Gram matrix, so it's not suitable when lots of trials are needed. [Detailed Description](./GPTuner.md)

**Requirement of classArg**

* **optimize_mode** (*'maximize' or 'minimize', optional, default = 'maximize'*) - If 'maximize', the tuner will target to maximize metrics. If 'minimize', the tuner will target to minimize metrics.
* **utility** (*'ei', 'ucb' or 'poi', optional, default = 'ei'*) - The kind of utility function(acquisition function). 'ei', 'ucb' and 'poi' corresponds to 'Expected Improvement', 'Upper Confidence Bound' and 'Probability of Improvement' respectively. 
* **kappa** (*float, optional, default = 5*) - Used by utility function 'ucb'. The bigger `kappa` is, the more the tuner will be exploratory.
* **xi** (*float, optional, default = 0*) - Used by utility function 'ei' and 'poi'. The bigger `xi` is, the more the tuner will be exploratory.
* **nu** (*float, optional, default = 2.5*) - Used to specify Matern kernel. The smaller nu, the less smooth the approximated function is.
* **alpha** (*float, optional, default = 1e-6*) - Used to specify Gaussian Process Regressor. Larger values correspond to increased noise level in the observations.
* **cold_start_num** (*int, optional, default = 10*) - Number of random exploration to perform before Gaussian Process. Random exploration can help by diversifying the exploration space.
* **selection_num_warm_up** (*int, optional, default = 1e5*) - Number of random points to evaluate for getting the point which maximizes the acquisition function.
* **selection_num_starting_points** (*int, optional, default = 250*) - Number of times to run L-BFGS-B from a random starting point after the warmup.

**Usage example**

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