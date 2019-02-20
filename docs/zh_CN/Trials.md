# 实现 NNI 的 Trial（尝试）代码

**Trial（尝试）**是将一组参数组合（例如，超参）在模型上独立的一次尝试。

定义 NNI 的 Trial，需要首先定义参数组，并更新模型代码。 NNI 有两种方法来实现 Trial：[NNI API](#nni-api) 以及 [NNI Python annotation](#nni-annotation)。 参考[这里的](#more-examples)更多 Trial 样例。

<a name="nni-api"></a>

## NNI API

### 第一步：准备搜索空间参数文件。

样例如下：

```json
{
    "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
}
```

参考 [SearchSpaceSpec.md](./SearchSpaceSpec.md) 进一步了解搜索空间。 Tuner 会根据搜索空间来生成配置，即从每个超参的范围中选一个值。

### 第二步：更新模型代码

* Import NNI
    
    在 Trial 代码中加上 `import nni`。

* 从 Tuner 获得参数值

```python
RECEIVED_PARAMS = nni.get_next_parameter()
```

`RECEIVED_PARAMS` 是一个对象，如： `{"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}`.

* 定期返回指标数据（可选）

```python
nni.report_intermediate_result(metrics)
```

`指标`可以是任意的 Python 对象。 如果使用了 NNI 内置的 Tuner/Assessor，`指标`只可以是两种类型：1) 数值类型，如 float、int， 2) dict 对象，其中必须由键名为 `default`，值为数值的项目。 `指标`会发送给[Assessor](Builtin_Assessors.md)。 通常，`指标`是损失值或精度。

* 返回配置的最终性能

```python
nni.report_final_result(metrics)
```

`指标`也可以是任意的 Python 对象。 如果使用了内置的 Tuner/Assessor，`指标`格式和 `report_intermediate_result` 中一样，这个数值表示模型的性能，如精度、损失值等。 `指标`会发送给 [Tuner](Builtin_Tuner.md)。

### 第三步：启用 NNI API

要启用 NNI 的 API 模式，需要将 useAnnotation 设置为 *false*，并提供搜索空间文件的路径（即第一步中定义的文件）：

```yaml
useAnnotation: false
searchSpacePath: /path/to/your/search_space.json
```

参考 [这里](ExperimentConfig.md) 进一步了解如何配置实验。

* 参考[这里](https://nni.readthedocs.io/en/latest/sdk_reference.html)，了解更多 NNI API (例如 `nni.get_sequence_id()`)。

<a name="nni-annotation"></a>

## NNI Annotation

另一种实现 Trial 的方法是使用 Python 注释来标记 NNI。 就像其它 Python Annotation，NNI 的 Annotation 和代码中的注释一样。 不需要在代码中做大量改动。 只需要添加一些 NNI Annotation，就能够：

* 标记需要调整的参数变量 
* 指定变量的搜索空间范围
* 标记哪个变量需要作为中间结果范围给 `Assessor`
* 标记哪个变量需要作为最终结果（例如：模型精度）返回给 `Tuner`。 

同样以 MNIST 为例，只需要两步就能用 NNI Annotation 来实现 Trial 代码。

### 第一步：在代码中加入 Annotation

下面是加入了 Annotation 的 TensorFlow 代码片段，高亮的 4 行 Annotation 用于：

1. 调优 batch\_size 和 dropout\_rate
2. 每执行 100 步返回 test\_acc
3. 最后返回 test\_acc 作为最终结果。

新添加的代码都是注释，不会影响以前的执行逻辑。因此这些代码仍然能在没有安装 NNI 的环境中运行。

```diff
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
+   """@nni.variable(nni.choice(50, 250, 500), name=batch_size)"""
    batch_size = 128
    for i in range(10000):
        batch = mnist.train.next_batch(batch_size)
+       """@nni.variable(nni.choice(1, 5), name=dropout_rate)"""
        dropout_rate = 0.5
        mnist_network.train_step.run(feed_dict={mnist_network.images: batch[0],
                                                mnist_network.labels: batch[1],
                                                mnist_network.keep_prob: dropout_rate})
        if i % 100 == 0:
            test_acc = mnist_network.accuracy.eval(
                feed_dict={mnist_network.images: mnist.test.images,
                            mnist_network.labels: mnist.test.labels,
                            mnist_network.keep_prob: 1.0})
+           """@nni.report_intermediate_result(test_acc)"""

    test_acc = mnist_network.accuracy.eval(
        feed_dict={mnist_network.images: mnist.test.images,
                    mnist_network.labels: mnist.test.labels,
                    mnist_network.keep_prob: 1.0})

+   """@nni.report_final_result(test_acc)"""
```

**注意**：

* `@nni.variable` 会对它的下面一行进行修改，左边被赋值变量必须在 `@nni.variable` 的 `name` 参数中指定。
* `@nni.report_intermediate_result`/`@nni.report_final_result` 会将数据发送给 Assessor、Tuner。 

Annotation 的语法和用法等，参考 [Annotation](AnnotationSpec.md)。

### 第二步：启用 Annotation

在 YAML 配置文件中设置 *useAnnotation* 为 true 来启用 Annotation：

    useAnnotation: true
    

<a name="more-examples"></a>

## 更多 Trial 的样例

* [MNIST 样例](mnist_examples.md)
* [为 CIFAR 10 分类找到最佳的 optimizer](cifar10_examples.md)
* [如何在 NNI 调优 SciKit-learn 的参数](sklearn_examples.md)
* [在阅读理解上使用自动模型架构搜索。](SQuAD_evolution_examples.md)
* [如何在 NNI 上调优 GBDT](gbdt_example.md)