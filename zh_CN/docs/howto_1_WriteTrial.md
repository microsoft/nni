# Write a Trial Run on NNI

**Trial（尝试）**是将一组参数在模型上独立的一次尝试。

定义 NNI 的 Trial，需要首先定义参数组，并更新模型代码。 NNI 有两种方法来定义 Trial：`NNI API` 和 `NNI Annotation（标记）`.

## NNI API

> 第一步：准备搜索空间参数文件。

样例如下：

```json
{
    "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
}
```

参考 [SearchSpaceSpec.md](./SearchSpaceSpec.md) 进一步了解搜索空间。

> 第二步：更新模型代码

    2.1 声明 NNI API
        在 Trial 代码中通过 `import nni` 来导入 NNI API。 
    
    2.2 获取预定义的参数
        参考下列代码片段： 
    
            RECEIVED_PARAMS = nni.get_next_parameter()
    
        来获得 Tuner 分配的超参值。 `RECEIVED_PARAMS` 是一个对象，例如： 
    
            {"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}
    
    2.3 向 NNI 返回结果
        使用 API：
    
            `nni.report_intermediate_result(accuracy)` 
    
        返回 `accuracy` 的值给 Assessor。
    
        使用 API:
    
            `nni.report_final_result(accuracy)` 
    
        返回 `accuracy` 的值给 Tuner。
    

**注意**：

    accuracy - 如果使用内置的 Tuner/Assessor，那么 `accuracy` 必须是数值（如 float, int）。在定制 Tuner/Assessor 时 `accuracy` 可以是任何类型的 Python 对象。
    Assessor（评估器）- 会根据 Trial 的历史值（即其中间结果），来决定这次 Trial 是否应该提前终止。
    tuner（调参器） - 会根据探索的历史（所有 Trial 的最终结果）来生成下一组参数、架构。
    

> 第三步：启用 NNI API

要启用 NNI 的 API 模式，需要将 useAnnotation 设置为 *false*，并提供搜索空间文件的路径（即第一步中定义的文件）：

    useAnnotation: false
    searchSpacePath: /path/to/your/search_space.json
    

参考[这里](./ExperimentConfig.md)进一步了解如何配置 Experiment。

参考[这里](../examples/trials/README.md)了解如何使用 NNI API 来编写 Trial 代码。

## NNI Annotation

另一种实现 Trial 的方法是使用 Python 注释来标记 NNI。 就像其它 Python Annotation，NNI 的 Annotation 和代码中的注释一样。 不需要在代码中做大量改动。 只需要添加一些 NNI Annotation，就能够：

* 标记需要调整的参数变量 
* 指定变量的搜索空间范围
* 标记哪个变量需要作为中间结果范围给`Assessor`
* 标记哪个变量需要作为最终结果（例如：模型精度）返回给`Tuner`。

同样以 MNIST 为例，只需要两步就能用 NNI Annotation 来实现 Trial 代码。

> 第一步：在代码中加入 Annotation

参考下列 tensorflow 的 NNI Annotation 的代码片段，高亮的 4 行 Annotation 实现了： (1) 调整 batch\_size 和 (2) dropout\_rate, (3) 每 100 步返回一次 test\_acc ，并且 (4) 在最后返回 test\_acc 作为最终结果。

> 值得注意的是，新添加的代码都是注释，不会影响以前的执行逻辑。因此这些代码仍然能在没有安装 NNI 的环境中运行。

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

> 注意
> 
> > `@nni.variable` 会影响下面紧接的一行。
> > 
> > `@nni.report_intermediate_result`/`@nni.report_final_result` 会将数据发送给 Assessor、Tuner。
> > 
> > 参考 [Annotation](../tools/nni_annotation/README.md) 了解更多关于标记的语法和用法。
> 
> 第二步：启用 NNI Annotation 在 yml 配置文件中，将 *useAnnotation* 设置为 true 来启用 NNI Annotation。

```yaml
useAnnotation: true
```

## 更多 Trial 的样例

* [在阅读理解上使用自动模型架构搜索。](../examples/trials/ga_squad/README.md)