## 如何定义搜索空间？

### 超参搜索空间

* 超参搜索空间配置样例：

```python
{
    "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "batch_size":{"_type":"choice","_value":[50, 250, 500]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
}

```

The example define ```dropout_rate``` as variable which priori distribution is uniform distribution, and its value from ```0.1``` and ```0.5```. The tuner will sample parameters/architecture by understanding the search space first.

用户需要定义变量名、类型和取值范围。 变量类型和取值范围包括：

* {"_type":"choice","_value":options}
   
   * 变量值会是列表中的 options 之一。options 的元素可以是 [nested] 嵌套的随机表达式。 在这种情况下，随机选项仅会出现在某些选项满足条件时。   
      

* {"_type":"randint","_value":[upper]}
   
   * 此变量为范围 [0, upper) 之间的随机整数。 这种分布的语义，在较远整数与附近整数之间的损失函数无太大关系， 这是用来描述随机种子的较好分布。 如果损失函数与较近的整数更相关，则应该使用某个"quantized"的连续分布，如quniform, qloguniform, qnormal 或 qlognormal。 注意，如果需要改动数字下限，可以使用 `quniform`。   
      

* {"_type":"uniform","_value":[low, high]}
   
   * 变量是 low 和 high 之间均匀分布的值。
   * When optimizing, this variable is constrained to a two-sided interval.   
      

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
      

注意：SMAC 仅支持部分类型，包括 `choice`, `randint`, `uniform`, `loguniform`, `quniform(q=1)`。 当前版本中，SMAC 不支持级联搜索空间（即，SMAC中的条件变量）。

Note that GridSearch Tuner only supports a subset of the types above, including `choic`, `quniform` and `qloguniform`, where q here specifies the number of values that will be sampled. 最后两种类型的细节如下：

* 类型 'quniform' 接收三个值 [low, high, q]， 其中 [low, high] 指定了范围，而 'q' 指定了会被均匀采样的值的数量。 注意 q 至少为 2。 它的第一个采样值为 'low'，每个采样值都会比前一个大 (high-low)/q 。
* 类型 'qloguniform' 的行为与 'quniform' 类似，不同处在于首先将范围改为 [log(low), log(high)] 采样后，再将数值还原。