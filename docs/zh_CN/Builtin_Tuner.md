# 内置 Tuner

NNI 提供了先进的调优算法，使用上也很简单。 下面是内置 Tuner 的简单介绍：

注意：点击 **Tuner 的名称**可跳转到算法的详细描述，点击**用法**可看到 Tuner 的安装要求、建议场景和使用样例等等。

当前支持的 Tuner：

* [TPE](hyperoptTuner.md)
* [Random Search（随机搜索）](hyperoptTuner.md)
* [Anneal（退火算法）](hyperoptTuner.md)
* [Naive Evolution（进化算法）](evolutionTuner.md)
* [SMAC](smacTuner.md)
* [Batch Tuner（批量调参器）](batchTuner.md)
* [Grid Search（网格搜索）](gridsearchTuner.md)
* [Hyperband](hyperbandAdvisor.md)
* [Network Morphism](networkmorphismTuner.md)
* [Metis Tuner](metisTuner.md)

| Tuner                                         | 算法简介                                                                                                                                                                                                                                                                                           |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TPE** [(用法)](#TPE)                          | Tree-structured Parzen Estimator (TPE) 是一种 sequential model-based optimization（SMBO，即基于序列模型优化）的方法。 SMBO 方法根据历史指标数据来按顺序构造模型，来估算超参的性能，随后基于此模型来选择新的超参。 [参考论文](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)                                                  |
| **Random Search** [(用法)](#Random)             | 在超参优化时，随机搜索算法展示了其惊人的简单和效果。 建议当不清楚超参的先验分布时，采用随机搜索作为基准。 [参考论文](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)                                                                                                                                                                  |
| **Anneal** [(用法)](#Anneal)                    | 这种简单的退火算法从先前的采样开始，会越来越靠近发现的最佳点取样。 此算法是随机搜索的简单变体，利用了反应曲面的平滑性。 退火率不是自适应的。                                                                                                                                                                                                                        |
| **Naive Evolution** [(用法)](#Evolution)        | 朴素进化算法来自于大规模图像分类进化。 它会基于搜索空间随机生成一个种群。 在每一代中，会选择较好的结果，并对其下一代进行一些变异（例如，改动一个超参，增加或减少一层）。 进化算法需要很多次 Trial 才能有效，但它也非常简单，也很容易扩展新功能。 [参考论文](https://arxiv.org/pdf/1703.01041.pdf)                                                                                                                     |
| **SMAC** [(用法)](#SMAC)                        | SMAC 基于 Sequential Model-Based Optimization (SMBO，即序列的基于模型优化方法)。 它会利用使用过的结果好的模型（高斯随机过程模型），并将随机森林引入到 SMBO 中，来处理分类参数。 SMAC 算法包装了 Github 的 SMAC3。 注意：SMAC 需要通过 `nnictl package` 命令来安装。 [参考论文，](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) [Github 代码库](https://github.com/automl/SMAC3) |
| **Batch tuner** [(用法)](#Batch)                | Batch Tuner 能让用户简单的提供几组配置（如，超参选项的组合）。 当所有配置都执行完后，Experiment 即结束。 Batch Tuner 仅支持 choice 类型。                                                                                                                                                                                                    |
| **Grid Search** [(用法)](#GridSearch)           | Grid Search 会穷举定义在搜索空间文件中的所有超参组合。 网格搜索可以使用的类型有 choice, quniform, qloguniform。 quniform 和 qloguniform 中的数值 q 具有特别的含义（不同于搜索空间文档中的说明）。 它表示了在最高值与最低值之间采样的值的数量。                                                                                                                                     |
| **Hyperband** [(用法)](#Hyperband)              | Hyperband 试图用有限的资源来探索尽可能多的组合，并发现最好的结果。 它的基本思路是生成大量的配置，并运行少量的步骤来找到有可能好的配置，然后继续训练找到其中更好的配置。 [参考论文](https://arxiv.org/pdf/1603.06560.pdf)                                                                                                                                                         |
| **Network Morphism** [(用法)](#NetworkMorphism) | Network Morphism 提供了深度学习模型的自动架构搜索功能。 每个子网络都继承于父网络的知识和形态，并变换网络的不同形态，包括深度，宽度，跨层连接（skip-connection）。 然后使用历史的架构和指标，来估计子网络的值。 最后会选择最有希望的模型进行训练。 [参考论文](https://arxiv.org/abs/1806.10282)                                                                                                            |
| **Metis Tuner** [(用法)](#MetisTuner)           | 大多数调参工具仅仅预测最优配置，而 Metis 的优势在于有两个输出：(a) 最优配置的当前预测结果， 以及 (b) 下一次 Trial 的建议。 它不进行随机取样。 大多数工具假设训练集没有噪声数据，但 Metis 会知道是否需要对某个超参重新采样。 [参考论文](https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/)                                                |

<br />

## 用法

要使用 NNI 内置的 Tuner，需要在 `config.yml` 文件中添加 **builtinTunerName** 和 **classArgs**。 这一节会介绍推荐的场景、参数等详细用法以及样例。

注意：参考样例中的格式来创建新的 `config.yml` 文件。 一些内置的 Tuner 还需要通过 `nnictl package` 命令先安装，如 SMAC。

<a name="TPE"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `TPE`

> 名称：**TPE**

**建议场景**

TPE 是一种黑盒优化方法，可以使用在各种场景中，通常情况下都能得到较好的结果。 特别是在计算资源有限，只能运行少量 Trial 的情况。 大量的实验表明，TPE 的性能远远优于随机搜索。

**参数**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。

**使用样例：**

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

在每个 Trial 运行时间不长（例如，能够非常快的完成，或者很快的被 Assessor 终止），并有充足计算资源的情况下。 或者需要均匀的探索搜索空间。 随机搜索可作为搜索算法的基准线。

**参数**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。

**Usage example**

```yaml
# config.yml
tuner:
  builtinTunerName: Random
  classArgs:
    optimize_mode: maximize
```

<br />

<a name="Anneal"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Anneal`

> 名称：**Anneal**

**Suggested scenario**

Anneal is suggested when each trial does not take too long, and you have enough computation resource(almost same with Random Search). Or the variables in search space could be sample from some prior distribution.

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。

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

> 名称：**Evolution**

**Suggested scenario**

Its requirement of computation resource is relatively high. Specifically, it requires large initial population to avoid falling into local optimum. If your trial is short or leverages assessor, this tuner is a good choice. And, it is more suggested when your trial code supports weight transfer, that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training progress.

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。

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

> 名称：**SMAC**

**Installation**

SMAC need to be installed by following command before first use.

```bash
nnictl package install --name=SMAC
```

**Suggested scenario**

Similar to TPE, SMAC is also a black-box tuner which can be tried in various scenarios, and is suggested when computation resource is limited. It is optimized for discrete hyperparameters, thus, suggested when most of your hyperparameters are discrete.

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。

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

> 名称：BatchTuner

**Suggested scenario**

If the configurations you want to try have been decided, you can list them in searchspace file (using `choice`) and run them using batch tuner.

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

> 名称：**Grid Search**

**Suggested scenario**

Note that the only acceptable types of search space are `choice`, `quniform`, `qloguniform`. **The number `q` in `quniform` and `qloguniform` has special meaning (different from the spec in [search space spec](./SearchSpaceSpec.md)). It means the number of values that will be sampled evenly from the range `low` and `high`.**

It is suggested when search space is small, it is feasible to exhaustively sweeping the whole search space.

**Usage example**

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

It is suggested when you have limited computation resource but have relatively large search space. It performs well in the scenario that intermediate result (e.g., accuracy) can reflect good or bad of final result (e.g., accuracy) to some extent.

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。
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

<br />

<a name="NetworkMorphism"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Network Morphism`

> 名称：**NetworkMorphism**

**Installation**

NetworkMorphism requires [pyTorch](https://pytorch.org/get-started/locally), so users should install it first.

**Suggested scenario**

It is suggested that you want to apply deep learning methods to your task (your own dataset) but you have no idea of how to choose or design a network. You modify the [example](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism/cifar10/cifar10_keras.py) to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate or optimizer. It is feasible for different tasks to find a good network architecture. Now this tuner only supports the computer vision domain.

**Requirement of classArg**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will return the hyperparameter set with larger expectation. 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。
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

<br />

<a name="MetisTuner"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Metis Tuner`

> 名称：**MetisTuner**

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