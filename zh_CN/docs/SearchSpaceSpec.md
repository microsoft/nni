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
  
  * 这表示变量值应该是列表中的选项之一。 选项的元素也可以是 [nested]（嵌套的）随机表达式。 在这种情况下，随机选项仅会在条件满足时出现。

* {"_type":"randint","_value":[upper]}
  
  * 此变量为范围 [0, upper) 之间的随机整数。 这种分布的语义，在较远整数与附近整数之间的损失函数无太大关系， 这是用来描述随机种子的较好分布。 如果损失函数与较近的整数更相关，则应该使用某个"quantized"的连续分布，如quniform, qloguniform, qnormal 或 qlognormal。 注意，如果需要改动数字下限，可以使用 `quniform`。

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

* {"_type":"normal","_value":[label, mu, sigma]}
  
  * 变量值为实数，且为正态分布，均值为 mu，标准方差为 sigma。 优化时，此变量不受约束。

* {"_type":"qnormal","_value":[label, mu, sigma, q]}
  
  * 这表示变量值会类似于 round(normal(mu, sigma) / q) * q
  * 适用于在 mu 周围的离散变量，且没有上下限限制。

* {"_type":"lognormal","_value":[label, mu, sigma]}
  
  * 变量值为 exp(normal(mu, sigma)) 分布，范围值是对数的正态分布。 当优化时，此变量必须是正数。

* {"_type":"qlognormal","_value":[label, mu, sigma, q]}
  
  * 这表示变量值会类似于 round(exp(normal(mu, sigma)) / q) * q
  * 适用于值是“平滑”的离散变量，但某一边有界。

## 每种 Tuner 支持的搜索空间类型

|                     |  choice  | randint  | uniform  | quniform | loguniform | qloguniform |  normal  | qnormal  | lognormal | qlognormal |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|:----------:|:-----------:|:--------:|:--------:|:---------:|:----------:|
|      TPE Tuner      | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
| Random Search Tuner | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|    Anneal Tuner     | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|   Evolution Tuner   | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|     SMAC Tuner      | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |             |          |          |           |            |
|     Batch Tuner     | &#10003; |          |          |          |            |             |          |          |           |            |
|  Grid Search Tuner  | &#10003; |          |          | &#10003; |            |  &#10003;   |          |          |           |            |
|  Hyperband Advisor  | &#10003; | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|     Metis Tuner     | &#10003; | &#10003; | &#10003; | &#10003; |            |             |          |          |           |            |

注意，在 Grid Search Tuner 中，为了使用方便 `quniform` 和 `qloguniform` 的定义也有所改变，其中的 q 表示采样值的数量。 详情如下：

* 类型 'quniform' 接收三个值 [low, high, q]， 其中 [low, high] 指定了范围，而 'q' 指定了会被均匀采样的值的数量。 注意 q 至少为 2。 它的第一个采样值为 'low'，每个采样值都会比前一个大 (high-low)/q 。
* 类型 'qloguniform' 的行为与 'quniform' 类似，不同处在于首先将范围改为 [log(low), log(high)] 采样后，再将数值还原。

注意 Metis Tuner 当前仅支持在 `choice` 中使用数值。