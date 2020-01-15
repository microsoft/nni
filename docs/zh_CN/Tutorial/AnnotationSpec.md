# NNI Annotation

## 概述

为了获得良好的用户体验并减少对以后代码的影响，NNI 设计了通过 Annotation（标记）来使用的语法。 通过 Annotation，只需要在代码中加入一些注释字符串，就能启用 NNI，完全不影响代码原先的执行逻辑。

示例如下：

```python
'''@nni.variable(nni.choice(0.1, 0.01, 0.001), name=learning_rate)'''
learning_rate = 0.1

```

此示例中，NNI 会从 (0.1, 0.01, 0.001) 中选择一个值赋给 learning_rate 变量。 第一行就是 NNI 的 Annotation，是 Python 中的一个字符串。 接下来的一行需要是赋值语句。 NNI 会根据 Annotation 行的信息，来给这一行的变量赋上相应的值。

通过这种方式，不需要修改任何代码，代码既可以直接运行，又可以使用 NNI 来调参。

## Annotation 的类型：

NNI 中，有 4 种类型的 Annotation；

### 1. 变量

`'''@nni.variable(sampling_algo, name)'''`

`@nni.variable` 用来标记变量。

**参数**

- **sampling_algo**: 指定搜索空间的采样算法。 可将其换成 NNI 支持的其它采样函数，函数要以 `nni.` 开头。例如，`choice` 或 `uniform`，详见 [SearchSpaceSpec](SearchSpaceSpec.md)。
- **name**: 将被赋值的变量名称。 注意，此参数应该与下面一行等号左边的值相同。

NNI 支持如下 10 种类型来表示搜索空间：

- `@nni.variable(nni.choice(option1,option2,...,optionN),name=variable)` 变量值是选项中的一种，这些变量可以是任意的表达式。
- `@nni.variable(nni.randint(lower, upper),name=variable)` 变量值的公式为：round(uniform(low, high))。 目前，值的类型为 float。 如果要使用整数，需要显式转换。
- `@nni.variable(nni.uniform(low, high),name=variable)` 变量值会是 low 和 high 之间均匀分布的某个值。
- `@nni.variable(nni.quniform(low, high, q),name=variable)` 变量值为 clip(round(uniform(low, high) / q) * q, low, high)，clip 操作用于约束生成值的边界。
- `@nni.variable(nni.loguniform(low, high),name=variable)` 变量值是 exp(uniform(low, high)) 的点，数值以对数均匀分布。
- `@nni.variable(nni.qloguniform(low, high, q),name=variable)` 变量值为 clip(round(loguniform(low, high) / q) * q, low, high)，clip 操作用于约束生成值的边界。
- `@nni.variable(nni.normal(mu, sigma),name=variable)` 变量值为正态分布的实数值，平均值为 mu，标准方差为 sigma。
- `@nni.variable(nni.qnormal(mu, sigma, q),name=variable)` 变量值分布的公式为： round(normal(mu, sigma) / q) * q
- `@nni.variable(nni.lognormal(mu, sigma),name=variable)` 变量值分布的公式为： exp(normal(mu, sigma))
- `@nni.variable(nni.qlognormal(mu, sigma, q),name=variable)` 变量值分布的公式为： round(exp(normal(mu, sigma)) / q) * q

示例如下：

```python
'''@nni.variable(nni.choice(0.1, 0.01, 0.001), name=learning_rate)'''
learning_rate = 0.1
```

### 2. 函数

`'''@nni.function_choice(*functions, name)'''`

`@nni.function_choice` 可以从几个函数中选择一个来执行。

**参数**

- **functions**: 可选择的函数。 注意，必须是包括参数的完整函数调用。 例如 `max_pool(hidden_layer, pool_size)`。
- **name**: 将被替换的函数名称。

例如：

```python
"""@nni.function_choice(max_pool(hidden_layer, pool_size), avg_pool(hidden_layer, pool_size), name=max_pool)"""
h_pooling = max_pool(hidden_layer, pool_size)
```

### 3. 中间结果

`'''@nni.report_intermediate_result(metrics)'''`

`@nni.report_intermediate_result` 用来返回中间结果，这和[在 NNI 上实现 Trial](../TrialExample/Trials.md) 中 `nni.report_intermediate_result` 的用法一样。

### 4. 最终结果

`'''@nni.report_final_result(metrics)'''`

`@nni.report_final_result` 用来返回当前 Trial 的最终结果，这和[在 NNI 上实现 Trial](../TrialExample/Trials.md) 中的 `nni.report_final_result` 用法一样。