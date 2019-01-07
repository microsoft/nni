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

此样例定义了 `dropout_rate` 变量，它的先验分布为均匀分布，值在 `0.1` 与 `0.5` 之间。 调参器会首先了解搜索空间，然后从中采样。

用户需要定义变量名、类型和取值范围。 变量类型和取值范围包括：

* {"_type":"choice","_value":options}
   
   * 变量值会是列表中的 options 之一。options 的元素可以是 [nested] 嵌套的随机表达式。 在这种情况下，随机选项仅会出现在某些选项满足条件时。   
      

* {"_type":"randint","_value":[upper]}
   
   * 此变量为范围 [0, upper) 之间的随机整数。 这种分布的语义，在较远整数与附近整数之间的损失函数无太大关系， 这是用来描述随机种子的较好分布。 如果损失函数与较近的整数更相关，则应该使用某个"quantized"的连续分布，如quniform, qloguniform, qnormal 或 qlognormal。 Note that if you want to change lower bound, you can use `quniform` for now.   
      

* {"_type":"uniform","_value":[low, high]}
   
   * Which means the variable value is a value uniformly between low and high.
   * When optimizing, this variable is constrained to a two-sided interval.   
      

* {"_type":"quniform","_value":[low, high, q]}
   
   * Which means the variable value is a value like round(uniform(low, high) / q) * q
   * Suitable for a discrete value with respect to which the objective is still somewhat "smooth", but which should be bounded both above and below. If you want to uniformly choose integer from a range [low, high], you can write `_value` like this: `[low, high, 1]`.   
      

* {"_type":"loguniform","_value":[low, high]}
   
   * Which means the variable value is a value drawn from a range [low, high] according to a loguniform distribution like exp(uniform(log(low), log(high))), so that the logarithm of the return value is uniformly distributed.
   * When optimizing, this variable is constrained to be positive.   
      

* {"_type":"qloguniform","_value":[low, high, q]}
   
   * Which means the variable value is a value like round(loguniform(low, high)) / q) * q
   * Suitable for a discrete variable with respect to which the objective is "smooth" and gets smoother with the size of the value, but which should be bounded both above and below.   
      

* {"_type":"normal","_value":[label, mu, sigma]}
   
   * Which means the variable value is a real value that's normally-distributed with mean mu and standard deviation sigma. When optimizing, this is an unconstrained variable.   
      

* {"_type":"qnormal","_value":[label, mu, sigma, q]}
   
   * Which means the variable value is a value like round(normal(mu, sigma) / q) * q
   * Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.   
      

* {"_type":"lognormal","_value":[label, mu, sigma]}
   
   * Which means the variable value is a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed. When optimizing, this variable is constrained to be positive.   
      

* {"_type":"qlognormal","_value":[label, mu, sigma, q]}
   
   * Which means the variable value is a value like round(exp(normal(mu, sigma)) / q) * q
   * Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is bounded from one side.   
      

Note that SMAC only supports a subset of the types above, including `choice`, `randint`, `uniform`, `loguniform`, `quniform(q=1)`. In the current version, SMAC does not support cascaded search space (i.e., conditional variable in SMAC).

Note that GridSearch Tuner only supports a subset of the types above, including `choic`, `quniform` and `qloguniform`, where q here specifies the number of values that will be sampled. Details about the last two type as follows

* Type 'quniform' will receive three values [low, high, q], where [low, high] specifies a range and 'q' specifies the number of values that will be sampled evenly. Note that q should be at least 2. It will be sampled in a way that the first sampled value is 'low', and each of the following values is (high-low)/q larger that the value in front of it.
* Type 'qloguniform' behaves like 'quniform' except that it will first change the range to [log(low), log(high)] and sample and then change the sampled value back.