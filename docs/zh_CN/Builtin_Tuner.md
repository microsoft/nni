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

**使用样例：**

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

**建议场景**

当每个 Trial 的时间不长，并且有足够的计算资源时使用（与随机搜索基本相同）。 或者搜索空间的变量能从一些先验分布中采样。

**参数**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。

**使用样例：**

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

此算法对计算资源的需求相对较高。 需要非常大的初始种群，以免落入局部最优中。 如果 Trial 时间很短，或者使用了 Assessor，就非常适合此算法。 如果 Trial 代码支持权重迁移，即每次 Trial 会从上一轮继承已经收敛的权重，建议使用此算法。 这会大大提高训练速度。

**参数**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。

**使用样例：**

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

**安装**

SMAC 在第一次使用前，必须用下面的命令先安装。

```bash
nnictl package install --name=SMAC
```

**建议场景**

与 TPE 类似，SMAC 也是一个可以被用在各种场景中的黑盒 Tuner。在计算资源有限时，也可以使用。 此算法为离散超参而优化，因此，如果大部分超参是离散值时，建议使用此算法。

**参数**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。

**使用样例：**

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

如果 Experiment 配置已确定，可通过 `choice` 将它们罗列到搜索空间文件中运行即可。

**使用样例：**

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

当搜索空间比较小，能够遍历整个搜索空间。

**使用样例：**

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

当搜索空间很大，但计算资源有限时建议使用。 中间结果能够很好的反映最终结果的情况下，此算法会非常有效。

**参数**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。
* **R** (*int, 可选, 默认为 60*) - 能分配给 Trial 的最大 STEPS (可以是 mini-batches 或 epochs 的数值)。 Trial 需要用 STEPS 来控制运行时间。
* **eta** (*int, 可选, 默认为 3*) - `(eta-1)/eta` 是丢弃 Trial 的比例。

**使用样例：**

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

需要将深度学习方法应用到自己的任务（自己的数据集）上，但不清楚该如何选择或设计网络。 可修改[样例](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism/cifar10/cifar10_keras.py)来适配自己的数据集和数据增强方法。 也可以修改批处理大小，学习率或优化器。 它可以为不同的任务找到好的网络架构。 当前，此 Tuner 仅支持视觉领域。

**参数**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。
* **task** (*('cv'), 可选, 默认为 'cv'*) - 实验的领域，当前仅支持视觉（cv）。
* **input_width** (*int, 可选, 默认为 = 32*) - 输入图像的宽度
* **input_channel** (*int, 可选, 默认为 3*) - 输入图像的通道数
* **n_output_node** (*int, 可选, 默认为 10*) - 输出分类的数量

**使用样例：**

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

**安装**

Metis Tuner 需要先安装 [sklearn](https://scikit-learn.org/)。 可通过 `pip3 install sklearn` 命令来安装。

**建议场景**

与 TPE 和 SMAC 类似，Metis 是黑盒 Tuner。 如果系统需要很长时间才能完成一次 Trial，Metis 就比随机搜索等其它方法要更合适。 此外，Metis 还为接下来的 Trial 提供了候选。 如何使用 Metis 的[样例](https://github.com/Microsoft/nni/tree/master/examples/trials/auto-gbdt/search_space_metis.json)。 通过调用 NNI 的 SDK，用户只需要发送 `精度` 这样的最终结果给 Tuner。

**参数**

* **optimize_mode** (*maximize 或 minimize，可选，默认值为 maximize*) - 如果为 'maximize'，Tuner 会给出有可能产生较大值的参数组合。 如果为 'minimize'，Tuner 会给出有可能产生较小值的参数组合。

**使用样例：**

```yaml
# config.yml
tuner:
  builtinTunerName: MetisTuner
  classArgs:
    optimize_mode: maximize
```