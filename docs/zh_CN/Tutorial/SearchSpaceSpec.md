# 搜索空间

## 概述

在 NNI 中，Tuner 会根据搜索空间来取样生成参数和网络架构。搜索空间通过 JSON 文件来定义。

要定义搜索空间，需要定义变量名称、采样策略的类型及其参数。

* 搜索空间样例如下：

```yaml
{
    "dropout_rate": {"_type": "uniform", "_value": [0.1, 0.5]},
    "conv_size": {"_type": "choice", "_value": [2, 3, 5, 7]},
    "hidden_size": {"_type": "choice", "_value": [124, 512, 1024]},
    "batch_size": {"_type": "choice", "_value": [50, 250, 500]},
    "learning_rate": {"_type": "uniform", "_value": [0.0001, 0.1]}
}

```

将第一行作为样例。 `dropout_rate` 定义了一个变量，先验分布为均匀分布，范围从 `0.1` 到 `0.5`。

## 类型

所有采样策略和参数如下：

* `{"_type": "choice", "_value": options}`
  
  * Which means the variable's value is one of the options. Here `options` should be a list of numbers or a list of strings. Using arbitrary objects as members of this list (like sublists, a mixture of numbers and strings, or null values) should work in most cases, but may trigger undefined behaviors.
  * `options` could also be a nested sub-search-space, this sub-search-space takes effect only when the corresponding element is chosen. The variables in this sub-search-space could be seen as conditional variables. Here is an simple [example of nested search space definition](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nested-search-space/search_space.json). If an element in the options list is a dict, it is a sub-search-space, and for our built-in tuners you have to add a key `_name` in this dict, which helps you to identify which element is chosen. Accordingly, here is a [sample](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nested-search-space/sample.json) which users can get from nni with nested search space definition. Tuners which support nested search space are as follows:
    
    * Random Search（随机搜索） 
    * TPE
    * Anneal（退火算法）
    * Evolution

* `{"_type": "randint", "_value": [lower, upper]}`
  
  * 从 `lower` (包含) 到 `upper` (不包含) 中选择一个随机整数。
  * 注意：不同 Tuner 可能对 `randint` 有不同的实现。 一些 Tuner（例如，TPE，GridSearch）将从低到高无序选择，而其它一些（例如，SMAC）则有顺序。 如果希望所有 Tuner 都有序，可使用 `quniform` 并设置 `q=1`。

* `{"_type": "uniform", "_value": [low, high]}`
  
  * 变量是 low 和 high 之间均匀分布的值。
  * 当优化时，此变量值会在两侧区间内。

* `{"_type": "quniform", "_value": [low, high, q]}`
  
  * 变量值为 `clip(round(uniform(low, high) / q) * q, low, high)`，clip 操作用于约束生成值的边界。 例如，`_value` 为 [0, 10, 2.5]，可取的值为 [0, 2.5, 5.0, 7.5, 10.0]; `_value` 为 [2, 10, 5]，可取的值为 [2, 5, 10]。
  * 适用于离散，同时反映了某种"平滑"的数值，但上下限都有限制。 如果需要从范围 [low, high] 中均匀选择整数，可以如下定义 `_value`：`[low, high, 1]`。

* `{"_type": "loguniform", "_value": [low, high]}`
  
  * 变量值在范围 [low, high] 中是 loguniform 分布，如 exp(uniform(log(low), log(high)))，因此返回值是对数均匀分布的。
  * 当优化时，此变量必须是正数。

* `{"_type": "qloguniform", "_value": [low, high, q]}`
  
  * 变量值为 `clip(round(loguniform(low, high) / q) * q, low, high)`，clip 操作用于约束生成值的边界。
  * 适用于值是“平滑”的离散变量，但上下限均有限制。

* `{"_type": "normal", "_value": [mu, sigma]}`
  
  * 变量值为实数，且为正态分布，均值为 mu，标准方差为 sigma。 优化时，此变量不受约束。

* `{"_type": "qnormal", "_value": [mu, sigma, q]}`
  
  * 这表示变量值会类似于 `round(normal(mu, sigma) / q) * q`
  * 适用于在 mu 周围的离散变量，且没有上下限限制。

* `{"_type": "lognormal", "_value": [mu, sigma]}`
  
  * 变量值为 `exp(normal(mu, sigma))` 分布，范围值是对数的正态分布。 当优化时，此变量必须是正数。

* `{"_type": "qlognormal", "_value": [mu, sigma, q]}`
  
  * 这表示变量值会类似于 `round(exp(normal(mu, sigma)) / q) * q`
  * 适用于值是“平滑”的离散变量，但某一边有界。

* `{"_type": "mutable_layer", "_value": {mutable_layer_infomation}}`
  
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