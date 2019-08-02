# 搜索空间

## 概述

在 NNI 中，Tuner 会根据搜索空间来取样生成参数和网络架构。搜索空间通过 JSON 文件来定义。

要定义搜索空间，需要定义变量名称、采样策略的类型及其参数。

* 搜索空间样例如下：

```yaml
{
    "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "batch_size":{"_type":"choice","_value":[50, 250, 500]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
}

```

将第一行作为样例。 `dropout_rate` 定义了一个变量，先验分布为均匀分布，范围从 `0.1` 到 `0.5`。

## 类型

所有采样策略和参数如下：

* {"_type":"choice","_value":options}
  
  * 表示变量的值是选项之一。 这里的 'options' 是一个数组。 选项的每个元素都是字符串。 也可以是嵌套的子搜索空间。此子搜索空间仅在相应的元素选中后才起作用。 该子搜索空间中的变量可看作是条件变量。
  
  * [nested] 搜索空间定义的简单[示例](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nested-search-space/search_space.json)。 如果选项列表中的元素是 dict，则它是一个子搜索空间，对于内置的 Tuner，必须在此 dict 中添加键 “_name”，这有助于标识选中的元素。 相应的，这是使用从 NNI 获得的嵌套搜索空间的[示例](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nested-search-space/sample.json)。 以下 Tuner 支持嵌套搜索空间：
    
    * Random Search（随机搜索） 
    * TPE
    * Anneal（退火算法）
    * Evolution

* {"_type":"randint","_value":[lower, upper]}
  
  * 当前实现的是 "quniform" 的 "randint" 分布，随机变量的分布函数是 round(uniform(lower, upper))。 所选择值的类型是 float。 如果要使用整数，需要显式转换。

* {"_type":"uniform","_value":[low, high]}
  
  * 变量是 low 和 high 之间均匀分布的值。
  * 当优化时，此变量值会在两侧区间内。

* {"_type":"quniform","_value":[low, high, q]}
  
  * 这表示变量值会类似于 round(uniform(low, high) / q) * q
  * 适用于离散，同时反映了某种"平滑"的数值，但上下限都有限制。 如果需要从范围 [low, high] 中均匀选择整数，可以如下定义 `_value`：`[low, high, 1]`。

* {"_type":"loguniform","_value":[low, high]}
  
  * 变量值在范围 [low, high] 中是 loguniform 分布，如 exp(uniform(log(low), log(high)))，因此返回值是对数均匀分布的。
  * 当优化时，此变量必须是正数。

* {"_type":"qloguniform","_value":[low, high, q]}
  
  * 这表示变量值会类似于 round(loguniform(low, high)) / q) * q
  * 适用于值是“平滑”的离散变量，但上下限均有限制。

* {"_type":"normal","_value":[mu, sigma]}
  
  * 变量值为实数，且为正态分布，均值为 mu，标准方差为 sigma。 优化时，此变量不受约束。

* {"_type":"qnormal","_value":[mu, sigma, q]}
  
  * 这表示变量值会类似于 round(normal(mu, sigma) / q) * q
  * 适用于在 mu 周围的离散变量，且没有上下限限制。

* {"_type":"lognormal","_value":[mu, sigma]}
  
  * 变量值为 exp(normal(mu, sigma)) 分布，范围值是对数的正态分布。 当优化时，此变量必须是正数。

* {"_type":"qlognormal","_value":[mu, sigma, q]}
  
  * 这表示变量值会类似于 round(exp(normal(mu, sigma)) / q) * q
  * 适用于值是“平滑”的离散变量，但某一边有界。

* {"_type":"mutable_layer","_value":{mutable_layer_infomation}}
  
  * [神经网络架构搜索空间](../AdvancedFeature/GeneralNasInterfaces.md)的类型。 值是字典类型，键值对表示每个 mutable_layer 的名称和搜索空间。
  * 当前，只能通过 Annotation 来使用这种类型的搜索空间。因此不需要为搜索空间定义 JSON 文件，它会通过 Trial 中的 Annotation 自动生成。
  * 具体用法参考[通用 NAS 接口](../AdvancedFeature/GeneralNasInterfaces.md)。

## 每种 Tuner 支持的搜索空间类型

|                     |  choice  | randint  | uniform  | quniform | loguniform | qloguniform |  normal  | qnormal  | lognormal | qlognormal |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|:----------:|:-----------:|:--------:|:--------:|:---------:|:----------:|
|      TPE Tuner      | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
| Random Search Tuner | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|    Anneal Tuner     | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|   Evolution Tuner   | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|     SMAC Tuner      | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |             |          |          |           |            |
|     Batch Tuner     | &#10003; |          |          |          |            |             |          |          |           |            |
|  Grid Search Tuner  | &#10003; | &#10003; |          | &#10003; |            |             |          |          |           |            |
|  Hyperband Advisor  | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|     Metis Tuner     | &#10003; | &#10003; | &#10003; | &#10003; |            |             |          |          |           |            |
|      GP Tuner       | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   |          |          |           |            |

已知的局限：

* 注意 Metis Tuner 当前仅支持在 `choice` 中使用数值。

* 请注意，对于嵌套搜索空间：
  
      * 只有 随机搜索/TPE/Anneal/Evolution Tuner 支持嵌套搜索空间
      * 不支持嵌套搜索空间 "超参" 的可视化，对其的改进通过 #1110(https://github.com/microsoft/nni/issues/1110) 来跟踪 。欢迎任何建议和贡献。