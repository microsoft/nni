# 如何使用 NNI 支持的调参器？

目前，NNI 已支持下列调参器算法。 注意，NNI 只安装了部分下列算法，其它算法在使用前需要通过 `nnictl package install` 命令安装。 例如，安装 SMAC 算法的命令为 `nnictl package install --name=SMAC`。

* [TPE](#TPE)
* [Random Search（随机搜索）](#Random)
* [Anneal（退火算法）](#Anneal)
* [Naive Evolution（进化算法）](#Evolution)
* [SMAC](#SMAC) (需要通过 `nnictl` 命令安装)
* [Batch Tuner（批量调参器）](#Batch)
* [Grid Search（网格搜索）](#Grid)
* [Hyperband](#Hyperband)
* [Network Morphism](#NetworkMorphism) (需要安装 pyTorch)
    
    ## 支持的调参器算法

这里将介绍这些调参器算法的基本知识，各个调参器建议使用的场景，以及使用样例（完整的使用样例参考 [这里]()）。

<a name="TPE"></a>
**TPE**

Tree-structured Parzen Estimator (TPE) 是一种 sequential model-based optimization（SMBO，即基于序列模型优化）的方法。 SMBO 方法根据历史指标数据来按顺序构造模型，来估算超参的性能，随后基于此模型来选择新的超参。 TPE 方法对 P(x|y) 和 P(y) 建模，其中 x 表示超参，y 表示相关的评估指标。 P(x|y) 通过变换超参的生成过程来建模，用非参数密度（non-parametric densities）代替配置的先验分布。 细节可参考 [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)。 ​  
*建议场景*： TPE 作为黑盒的优化方法，能在广泛的场景中使用，通常都能得到较好的结果。 特别是在计算资源有限，只能进行少量尝试时。 从大量的实验中，我们发现 TPE 的性能远远优于随机搜索。

*用法*：

```yaml
  # config.yaml
  tuner:
    builtinTunerName: TPE
    classArgs:
      # 可选项: maximize, minimize
      optimize_mode: maximize
```

<a name="Random"></a>
**Random Search（随机搜索）**

[Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) 中介绍了随机搜索惊人的简单和效果。 建议当不清楚超参的先验分布时，采用随机搜索作为基准。

*建议场景*：在每个尝试运行时间不长（例如，能够非常快的完成，或者很快的被评估器终止），并有充足计算资源的情况下。 或者需要均匀的探索搜索空间。 随机搜索可作为搜索算法的基准线。

*用法*：

```yaml
  # config.yaml
  tuner:
    builtinTunerName: Random
```

<a name="Anneal"></a>
**Anneal（退火算法）**

这种简单的退火算法从先前的采样开始，会越来越靠近发现的最佳点取样。 此算法是随机搜索的简单变体，利用了反应曲面的平滑性。 退火率不是自适应的。

*建议场景*：当每个尝试的时间不长，并且有足够的计算资源时使用（与随机搜索基本相同）。 或者搜索空间的变量能从一些先验分布中采样。

*用法*：

```yaml
  # config.yaml
  tuner:
    builtinTunerName: Anneal
    classArgs:
      # 可选项: maximize, minimize
      optimize_mode: maximize
```

<a name="Evolution"></a>
**Naive Evolution（进化算法）**

进化算法来自于 [Large-Scale Evolution of Image Classifiers](https://arxiv.org/pdf/1703.01041.pdf)。 它会基于搜索空间随机生成一个种群。 在每一代中，会选择较好的结果，并对其下一代进行一些变异（例如，改动一个超参，增加或减少一层）。 进化算法需要很多次尝试才能有效，但它也非常简单，也很容易扩展新功能。

*建议场景*：它需要相对较多的计算资源。 需要非常大的初始种群，以免落入局部最优中。 如果尝试时间很短，或者利用了评估器，这个调参器就非常合适。 如果尝试代码支持权重迁移，即每次尝试会从上一轮继承已经收敛的权重，建议使用此算法。 这会大大提高训练速度。

*用法*：

```yaml
  # config.yaml
  tuner:
    builtinTunerName: Evolution
    classArgs:
      # 可选项: maximize, minimize
      optimize_mode: maximize
```

<a name="SMAC"></a>
**SMAC**

[SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) 基于 Sequential Model-Based Optimization (SMBO). 它会利用使用过的突出的模型（高斯随机过程模型），并将随机森林引入到SMBO中，来处理分类参数。 NNI 的 SMAC 通过包装 [SMAC3](https://github.com/automl/SMAC3) 来支持。

NNI 中的 SMAC 只支持部分类型的[搜索空间](./SearchSpaceSpec.md)，包括`choice`, `randint`, `uniform`, `loguniform`, `quniform(q=1)`。

*安装*：

* 安装 swig。 (Ubuntu 下使用 `sudo apt-get install swig`)
* 运行 `nnictl package install --name=SMAC`

*建议场景*：与 TPE 类似，SMAC 也是一个可以被用在各种场景中的黑盒调参器。在计算资源有限时，也可以使用。 此算法为离散超参而优化，因此，如果大部分超参是离散值时，建议使用此算法。

*用法*：

```yaml
  # config.yaml
  tuner:
    builtinTunerName: SMAC
    classArgs:
      # 可选项: maximize, minimize
      optimize_mode: maximize
```

<a name="Batch"></a>
**Batch tuner（批量调参器）**

批量调参器能让用户简单的提供几组配置（如，超参选项的组合）。 当所有配置都完成后，实验即结束。 批量调参器 的[搜索空间](./SearchSpaceSpec.md)只支持 `choice`。

*建议场景*：如果需要实验的配置已经决定好了，可通过批量调参器将它们列到搜索空间中运行即可。

*用法*：

```yaml
  # config.yaml
  tuner:
    builtinTunerName: BatchTuner
```

注意批量调参器支持的搜索空间文件如下例：

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

搜索空间文件使用了键 `combine_params`。 参数类型必须是 `choice` ，并且 `values` 要包含所有需要实验的参数组合。

<a name="Grid"></a>
**Grid Search（网格搜索）**

网格搜索会穷举定义在搜索空间文件中的所有超参组合。 注意，搜索空间仅支持 `choice`, `quniform`, `qloguniform`。 `quniform` 和 `qloguniform` 中的 **数字 `q` 有不同的含义（与[搜索空间](./SearchSpaceSpec.md)说明不同）。 在这里意味着会在 `low` 和 `high` 之间均匀取值的数量。</p> 

*建议场景*：当搜索空间比较小，能够遍历整个搜索空间。

*用法*：

```yaml
  # config.yaml
  tuner:
    builtinTunerName: GridSearch
```

<a name="Hyperband"></a>
**Hyperband**

[Hyperband](https://arxiv.org/pdf/1603.06560.pdf) 尝试用有限的资源来探索尽量多的组合，从最有可能的组合中找到最好结果。 它的基本思路是生成大量的配置，并运行少量的步骤来找到有可能好的配置，然后继续训练找到其中更好的配置。 参考[这里](../src/sdk/pynni/nni/hyperband_advisor/README.md)，了解更多信息。

*建议场景*：当搜索空间很大，但计算资源有限时建议使用。 中间结果能够很好的反映最终结果的情况下，此算法会非常有效。

*用法*：

```yaml
  # config.yaml
  advisor:
    builtinAdvisorName: Hyperband
    classArgs:
      # 可选项: maximize, minimize
      optimize_mode: maximize
      # R: 可分配给尝试的最大的 STEPS（可以是小批量或 epoch 的数量）。 每个尝试都需要用 STEPS 来控制运行的时间。
      R: 60
      # eta: 丢弃的尝试的比例
      eta: 3
```

<a name="NetworkMorphism"></a>
**Network Morphism**

[Network Morphism](7) 提供了深度学习模型的自动架构搜索功能。 每个子网络都继承于父网络的知识和形态，并变换网络的不同形态，包括深度，宽度，跨层连接（skip-connection）。 接着，使用历史的架构和指标，来估计子网络的值。 然后会选择最有希望的模型进行训练。 参考[这里](../src/sdk/pynni/nni/networkmorphism_tuner/README.md)，了解更多信息。

*安装*： NetworkMorphism 需要 [pyTorch](https://pytorch.org/get-started/locally)，必须提前安装它。

*建议场景*：需要将深度学习方法应用到自己的任务（自己的数据集）上，但不清楚该如何选择或设计网络。 可修改[样例](../examples/trials/network_morphism/cifar10/cifar10_keras.py)来适配自己的数据集和数据增强方法。 也可以修改批处理大小，学习率或优化器。 它可以为不同的任务找到好的网络架构。 当前，此调参器仅支持视觉领域。

*用法*：

```yaml
  # config.yaml
  tuner:
    builtinTunerName: NetworkMorphism
    classArgs:
      #可选项: maximize, minimize
      optimize_mode: maximize
      #当前，仅支持 cv（视觉）领域
      task: cv
      #输入图像宽度
      input_width: 32
      #输入图像通道数量
      input_channel: 3
      #分类的数量
      n_output_node: 10
```

# 如何使用 NNI 支持的评估器？

目前，NNI 已支持下列评估器算法。

* [Medianstop（中位数终止）](#Medianstop)
* [Curvefitting（曲线拟合）](#Curvefitting)

## 支持的评估器算法

<a name="Medianstop"></a>
**Medianstop（中位数终止）**

Medianstop 是一种简单的提前停止规则，可参考[论文](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf)。 如果尝试 X 的在步骤 S 的最好目标值比所有已完成尝试的步骤 S 的中位数值明显低，就会停止运行尝试 X。

*建议场景*：它适用于各种性能曲线，因而能被用到各种场景中来加速优化过程。

*用法*：

```yaml
  assessor:
    builtinAssessorName: Medianstop
    classArgs:
      #可选项: maximize, minimize
      optimize_mode: maximize
      # (可选) 尽在接收到 start_step 数量个中间结果后，才判断尝试是否需要停止。
      # start_step 的默认值是 0。
      start_step: 5
```

<a name="Curvefitting"></a>
**Curvefitting（曲线拟合）**

Curve Fitting 评估器是一个 LPA (learning, predicting, assessing，即学习、预测、评估) 的算法。 如果预测的尝试 X 在 step S 比性能最好的尝试要差，就会提前终止它。 此算法中，使用了 12 条曲线来拟合精度曲线，从[参考论文](http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf)中选择了大量的参数曲线模型。 学习曲线的形状与先验知识是一致的：都是典型的递增的、饱和的函数。

*建议场景*：它适用于各种性能曲线，因而能被用到各种场景中来加速优化过程。 更好的是，它能够处理并评估性能类似的曲线。

*用法*：

```yaml
  assessor:
    builtinAssessorName: Curvefitting
    classArgs:
      # (必须) epoch 的总数。
      # 需要此数据来决定需要预测的点。
      epoch_num: 20
      # (可选项) choice: maximize, minimize
      # 如果选择了 minimize 模式，需要调整阈值为 >= 1.0 (例如：threshold=1.1)

      * optimize_mode 的默认值是 maximize
      optimize_mode: maximize
      # (可选项) 决定尝试是否应被停止的次数
      # 为了节省资源，只有在大于 start_step(default=6) 的精度点才开始预测。
      # 只有在收到 start_step 个中间结果之后。
      # start_step 的默认值是 6。
      start_step: 6
      # (可选) 决定是否提前终止的阈值。
      # 例如，如果 threshold = 0.95, optimize_mode = maximize，最好的历史结果是 0.9，那么会在尝试的预测值低于 0.95 * 0.9 = 0.855 时停止。
      * 阈值的默认值是 0.95。
      threshold: 0.95
```