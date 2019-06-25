# 内置 Tuner

NNI 提供了先进的调优算法，使用上也很简单。 下面是内置 Tuner 的简单介绍：

注意：点击 **Tuner 的名称**可看到 Tuner 的安装需求，建议的场景以及示例。 算法的详细说明在每个 Tuner 建议场景的最后。 [本文](./CommunitySharings/HpoComparision.md)对比了不同 Tuner 在几个问题下的不同效果。

当前支持的 Tuner：

|Tuner|Brief Introduction of Algorithm| |\---|\---| |[**TPE**](#TPE)|The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. [Reference Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)| |[**Random Search**](#Random)|In Random Search for Hyper-Parameter Optimization show that Random Search might be surprisingly simple and effective. We suggest that we could use Random Search as the baseline when we have no knowledge about the prior distribution of hyper-parameters. [Reference Paper](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)| |[**Anneal**](#Anneal)|This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on the random search that leverages smoothness in the response surface. The annealing rate is not adaptive.| |[**Naive Evolution**](#Evolution)|Naive Evolution comes from Large-Scale Evolution of Image Classifiers. It randomly initializes a population-based on search space. For each generation, it chooses better ones and does some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naive Evolution requires many trials to works, but it's very simple and easy to expand new features. [Reference paper](https://arxiv.org/pdf/1703.01041.pdf)| |[**SMAC**](#SMAC)|SMAC is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by nni is a wrapper on the SMAC3 Github repo. Notice, SMAC need to be installed by `nnictl package` command. [Reference Paper,](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) [Github Repo](https://github.com/automl/SMAC3)| |[**Batch tuner**](#Batch)|Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type choice in search space spec.| |[**Grid Search**](#GridSearch)|Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file. Note that the only acceptable types of search space are choice, quniform, qloguniform. The number q in quniform and qloguniform has special meaning (different from the spec in search space spec). It means the number of values that will be sampled evenly from the range low and high.| |[**Hyperband**](#Hyperband)|Hyperband tries to use the limited resource to explore as many configurations as possible, and finds out the promising ones to get the final result. The basic idea is generating many configurations and to run them for the small number of trial budget to find out promising one, then further training those promising ones to select several more promising one.[Reference Paper](https://arxiv.org/pdf/1603.06560.pdf)| |[**Network Morphism**](#NetworkMorphism)|Network Morphism provides functions to automatically search for architecture of deep learning models. Every child network inherits the knowledge from its parent network and morphs into diverse types of networks, including changes of depth, width, and skip-connection. Next, it estimates the value of a child network using the historic architecture and metric pairs. Then it selects the most promising one to train. [Reference Paper](https://arxiv.org/abs/1806.10282)| |[**Metis Tuner**](#MetisTuner)|Metis offers the following benefits when it comes to tuning parameters: While most tools only predict the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guesswork. While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter. [Reference Paper](https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/)| |[**BOHB**](#BOHB)|BOHB is a follow-up work of Hyperband. It targets the weakness of Hyperband that new configurations are generated randomly without leveraging finished trials. For the name BOHB, HB means Hyperband, BO means Byesian Optimization. BOHB leverages finished trials by building multiple TPE models, a proportion of new configurations are generated through these models. [Reference Paper](https://arxiv.org/abs/1807.01774)| |[**GP Tuner**](#GPTuner)|Gaussian Process Tuner is a sequential model-based optimization (SMBO) approach with Gaussian Process as the surrogate. [Reference Paper, ](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)[Github Repo](https://github.com/fmfn/BayesianOptimization)| <br />

## 用法

要使用 NNI 内置的 Tuner，需要在 `config.yml` 文件中添加 **builtinTunerName** 和 **classArgs**。 这一节会介绍推荐的场景、参数等详细用法以及样例。

注意：参考样例中的格式来创建新的 `config.yml` 文件。 一些内置的 Tuner 还需要通过 `nnictl package` 命令先安装，如 SMAC。

<a name="TPE"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `TPE`

> 名称：**TPE**

**建议场景**

TPE 是一种黑盒优化方法，可以使用在各种场景中，通常情况下都能得到较好的结果。 特别是在计算资源有限，只能运行少量 Trial 的情况。 大量的实验表明，TPE 的性能远远优于随机搜索。 [详细说明](./HyperoptTuner.md)

**参数**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**示例**

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

在每个 Trial 运行时间不长（例如，能够非常快的完成，或者很快的被 Assessor 终止），并有充足计算资源的情况下。 或者需要均匀的探索搜索空间。 随机搜索可作为搜索算法的基准线。 [详细说明](./HyperoptTuner.md)

**参数**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**示例**

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

当每个 Trial 的时间不长，并且有足够的计算资源时使用（与随机搜索基本相同）。 或者搜索空间的变量能从一些先验分布中采样。 [详细说明](./HyperoptTuner.md)

**参数**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**示例**

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

> 名称：**Evolution**

**建议场景**

此算法对计算资源的需求相对较高。 需要非常大的初始种群，以免落入局部最优中。 如果 Trial 时间很短，或者使用了 Assessor，就非常适合此算法。 如果 Trial 代码支持权重迁移，即每次 Trial 会从上一轮继承已经收敛的权重，建议使用此算法。 这会大大提高训练速度。 [详细说明](./EvolutionTuner.md)

**示例**

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

> 名称：**SMAC**

**当前 SMAC 不支持在 WIndows 下运行。 原因参考：[github issue](https://github.com/automl/SMAC3/issues/483).**

**安装**

SMAC 在第一次使用前，必须用下面的命令先安装。

```bash
nnictl package install --name=SMAC
```

**建议场景**

与 TPE 类似，SMAC 也是一个可以被用在各种场景中的黑盒 Tuner。在计算资源有限时，也可以使用。 此算法为离散超参而优化，因此，如果大部分超参是离散值时，建议使用此算法。 [详细说明](./SmacTuner.md)

**参数**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**示例**

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

如果 Experiment 配置已确定，可通过 `choice` 将它们罗列到搜索空间文件中运行即可。 [详细说明](./BatchTuner.md)

**示例**

```yaml
# config.yml
tuner:
  builtinTunerName: BatchTuner
```

<br />

注意 Batch Tuner 支持的搜索空间文件如下例：

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

搜索空间文件使用了键 `combine_params`。 参数类型必须是 `choice` ，并且 `values` 要包含所有需要 Experiment 的参数组合。

<a name="GridSearch"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Grid Search`

> 名称：**Grid Search**

**建议场景**

注意，搜索空间仅支持 `choice`, `quniform`, `qloguniform`。 `quniform` 和 `qloguniform` 中的 **数字 `q` 有不同的含义（与[搜索空间](./SearchSpaceSpec.md)说明不同）。 这里的意义是在 `low` 和 `high` 之间均匀取值的数量。</p> 

当搜索空间比较小，能够遍历整个搜索空间。 [详细说明](./GridsearchTuner.md)

**示例**

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

当搜索空间很大，但计算资源有限时建议使用。 中间结果能够很好的反映最终结果的情况下，此算法会非常有效。 [详细说明](./HyperbandAdvisor.md)

**参数**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **R** (*int, 可选, 默认为 60*) - 分配给 Trial 的最大资源（可以是 mini-batches 或 epochs 的数值）。 每个 Trial 都需要用 TRIAL_BUDGET 来控制运行的步数。
* **eta** (*int, 可选, 默认为 3*) - `(eta-1)/eta` 是丢弃 Trial 的比例。

**示例**

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

必须先安装 [pyTorch](https://pytorch.org/get-started/locally)。

**建议场景**

需要将深度学习方法应用到自己的任务（自己的数据集）上，但不清楚该如何选择或设计网络。 可修改[样例](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism/cifar10/cifar10_keras.py)来适配自己的数据集和数据增强方法。 也可以修改批处理大小，学习率或优化器。 它可以为不同的任务找到好的网络架构。 当前，此 Tuner 仅支持视觉领域。 [详细说明](./NetworkmorphismTuner.md)

**参数**

* **optimize_mode** (*maximize 或 minimize, 可选项, 默认值为 maximize*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。
* **task** (*('cv'), 可选, 默认为 'cv'*) - 实验的领域，当前仅支持视觉（cv）。
* **input_width** (*int, 可选, 默认为 = 32*) - 输入图像的宽度
* **input_channel** (*int, 可选, 默认为 3*) - 输入图像的通道数
* **n_output_node** (*int, 可选, 默认为 10*) - 输出分类的数量

**示例**

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

注意，搜索空间仅支持 `choice`, `quniform`, `uniform` 和 `randint`。

**建议场景**

与 TPE 和 SMAC 类似，Metis 是黑盒 Tuner。 如果系统需要很长时间才能完成一次 Trial，Metis 就比随机搜索等其它方法要更合适。 此外，Metis 还为接下来的 Trial 提供了候选。 如何使用 Metis 的[样例](https://github.com/Microsoft/nni/tree/master/examples/trials/auto-gbdt/search_space_metis.json)。 通过调用 NNI 的 SDK，用户只需要发送`精度`这样的最终结果给 Tuner。 [详细说明](./MetisTuner.md)

**参数**

* **optimize_mode** (*'maximize' 或 'minimize', 可选项, 默认值为 'maximize'*) - 如果为 'maximize'，表示 Tuner 的目标是将指标最大化。 如果为 'minimize'，表示 Tuner 的目标是将指标最小化。

**示例**

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

BOHB Advisor 的使用依赖 [ConfigSpace](https://github.com/automl/ConfigSpace) 包，在第一次使用 BOHB 的时候，在命令行运行以下的指令来安装 ConfigSpace。

```bash
nnictl package install --name=BOHB
```

**建议场景**

与 Hyperband 类似, 当计算资源有限但搜索空间相对较大时, 建议使用此方法。 中间结果能够很好的反映最终结果的情况下，此算法会非常有效。 在这种情况下, 由于贝叶斯优化使用, 它可能会收敛到更好的配置。 [详细说明](./BohbAdvisor.md)

**参数**

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

*目前 NNI 的浮点类型仅支持十进制表示，必须使用 0.333 来代替 1/3，0.001代替 1e-3。*

**示例**

```yml
advisor:
  builtinAdvisorName: BOHB
  classArgs:
    optimize_mode: maximize
    min_budget: 1
    max_budget: 27
    eta: 3
```

<br />

<a name="GPTuner"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `GP Tuner`

> Builtin Tuner Name: **GPTuner**

Note that the only acceptable types of search space are `choice`, `randint`, `uniform`, `quniform`, `loguniform`, `qloguniform`.

**Suggested scenario**

As a strategy in Sequential Model-based Global Optimization(SMBO) algorithm, GP Tuner uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) and common tools can be employed. Therefore GP Tuner is most adequate for situations where the function to be optimized is a very expensive endeavor. GP can be used when the computation resource is limited. While GP Tuner has a computationoal cost that grows at *O(N^3)* due to the requirement of inverting the Gram matrix, so it's not suitable when lots of trials are needed. [Detailed Description](./GPTuner.md)

**Requirement of classArg**

* **optimize_mode** (*'maximize' or 'minimize', optional, default = 'maximize'*) - If 'maximize', the tuner will target to maximize metrics. If 'minimize', the tuner will target to minimize metrics.
* **utility** (*'ei', 'ucb' or 'poi', optional, default = 'ei'*) - The kind of utility function(acquisition function). 'ei', 'ucb' and 'poi' corresponds to 'Expected Improvement', 'Upper Confidence Bound' and 'Probability of Improvement' respectively. 
* **kappa** (*float, optional, default = 5*) - Used by utility function 'ucb'. The bigger `kappa` is, the more the tuner will be exploratory.
* **xi** (*float, optional, default = 0*) - Used by utility function 'ei' and 'poi'. The bigger `xi` is, the more the tuner will be exploratory.
* **nu** (*float, optional, default = 2.5*) - Used to specify Matern kernel. The smaller nu, the less smooth the approximated function is.
* **alpha** (*float, optional, default = 1e-6*) - Used to specify Gaussian Process Regressor. Larger values correspond to increased noise level in the observations.
* **cold_start_num** (*int, optional, default = 10*) - Number of random exploration to perform before Gaussian Process. Random exploration can help by diversifying the exploration space.
* **selection_num_warm_up** (*int, optional, default = 1e5*) - Number of random points to evaluate for getting the point which maximizes the acquisition function.
* **selection_num_starting_points** (*int, optional, default = 250*) - Nnumber of times to run L-BFGS-B from a random starting point after the warmup.

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: GPTuner
  classArgs:
    optimize_mode: maximize
    kappa: 5
    xi: 0
    nu: 2.5
    alpha: 1e-6
    cold_start_num: 10
    selection_num_warm_up: 100000
    selection_num_starting_points: 250
```