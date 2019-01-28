# NNI Annotation

## 概述

为了获得良好的用户体验并减少对以后代码的影响，NNI 设计了通过 Annotation（标记）来使用的语法。 通过 Annotation，只需要在代码中加入一些注释字符串，就能启用 NNI，完全不影响代码原先的执行逻辑。

样例如下：

```python
'''@nni.variable(nni.choice(0.1, 0.01, 0.001), name=learning_rate)'''
learning_rate = 0.1
```

此样例中，NNI 会从 (0.1, 0.01, 0.001) 中选择一个值赋给 learning_rate 变量。 第一行就是 NNI 的 Annotation，是 Python 中的一个字符串。 接下来的一行需要是赋值语句。 NNI 会根据 Annotation 行的信息，来给这一行的变量赋上相应的值。

通过这种方式，不需要修改任何代码，代码既可以直接运行，又可以使用 NNI 来调参。

## Annotation 的类型：

NNI 中，有 4 种类型的 Annotation；

### 1. Annotate variables

`'''@nni.variable(sampling_algo, name)'''`

`@nni.variable` is used in NNI to annotate a variable.

**Arguments**

- **sampling_algo**: Sampling algorithm that specifies a search space. User should replace it with a built-in NNI sampling function whose name consists of an `nni.` identification and a search space type specified in [SearchSpaceSpec](SearchSpaceSpec.md) such as `choice` or `uniform`. 
- **name**: The name of the variable that the selected value will be assigned to. Note that this argument should be the same as the left value of the following assignment statement.

An example here is:

```python
'''@nni.variable(nni.choice(0.1, 0.01, 0.001), name=learning_rate)'''
learning_rate = 0.1
```

### 2. Annotate functions

`'''@nni.function_choice(*functions, name)'''`

`@nni.function_choice` is used to choose one from several functions.

**Arguments**

- **\*functions**: Several functions that are waiting to be selected from. Note that it should be a complete function call with arguments. Such as `max_pool(hidden_layer, pool_size)`.
- **name**: The name of the function that will be replaced in the following assignment statement.

An example here is:

```python
"""@nni.function_choice(max_pool(hidden_layer, pool_size), avg_pool(hidden_layer, pool_size), name=max_pool)"""
h_pooling = max_pool(hidden_layer, pool_size)
```

### 3. Annotate intermediate result

`'''@nni.report_intermediate_result(metrics)'''`

`@nni.report_intermediate_result` is used to report intermediate result, whose usage is the same as `nni.report_intermediate_result` in <Trials.md>

### 4. Annotate final result

`'''@nni.report_final_result(metrics)'''`

`@nni.report_final_result` is used to report the final result of the current trial, whose usage is the same as `nni.report_final_result` in <Trials.md>