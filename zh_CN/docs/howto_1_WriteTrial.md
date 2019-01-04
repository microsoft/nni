# **在 NNI 上编写 Trial（尝试）代码**

**Trial（尝试）**是将一组参数在模型上独立的一次尝试。

定义 NNI 的尝试，需要首先定义参数组，并更新模型代码。 NNI 有两种方法来定义尝试：`NNI API` 和 `NNI 标记`.

## NNI API

> 第一步：准备搜索空间参数文件。

样例如下：

    {
        "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
        "conv_size":{"_type":"choice","_value":[2,3,5,7]},
        "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
        "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
    }
    

参考 <SearchSpaceSpec.md> 进一步了解搜索空间。

> 第二步：更新模型代码

    2.1 声明 NNI API
        在尝试代码中通过 `import nni` 来导入 NNI API。 
    
    2.2 获取预定义的参数
        参考下列代码片段： 
    
            RECEIVED_PARAMS = nni.get_next_parameter()
    
        来获得调参器分配的超参值。 `RECEIVED_PARAMS` 是一个对象，例如： 
    
            {"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}
    
    2.3 向 NNI 返回结果
        使用 API：
    
            `nni.report_intermediate_result(accuracy)` 
    
        返回 `accuracy` 的值给评估器。
    
        使用 API:
    
            `nni.report_final_result(accuracy)` 
    
        返回 `accuracy` 的值给调参器。 
    

**注意**：

    accuracy - 如果使用 NNI 内置的调参器/评估器，那么 `accuracy` 必须是数值（如 float, int）。在定制调参器/评估器时 `accuracy` 可以是任何类型的 Python 对象。
    评估器 - 会根据尝试的历史值（即其中间结果），来决定这次尝试是否应该提前终止。
    调参器 - 会根据探索的历史（所有尝试的最终结果）来生成下一组参数、架构。
    

> 第三步：启用 NNI API

要启用 NNI 的 API 模式，需要将 useAnnotation 设置为 *false*，并提供搜索空间文件的路径（即第一步中定义的文件）：

    useAnnotation: false
    searchSpacePath: /path/to/your/search_space.json
    

参考 [here](ExperimentConfig.md) 进一步了解如何配置实验。

参考 \[这里\](../examples/trials/README.md) 进一步了解如何使用 NNI 的 API 来编写尝试的代码。

## NNI 标记

另一种编写尝试的方法是使用 Python 注释来标记 NNI。 就像其它标记，NNI 的标记和代码中的注释一样。 You don't have to make structure or any other big changes to your existing codes. With a few lines of NNI annotation, you will be able to:

* annotate the variables you want to tune 
* specify in which range you want to tune the variables
* annotate which variable you want to report as intermediate result to `assessor`
* annotate which variable you want to report as the final result (e.g. model accuracy) to `tuner`. 

Again, take MNIST as an example, it only requires 2 steps to write a trial with NNI Annotation.

> 第一步：在代码中加入标记

Please refer the following tensorflow code snippet for NNI Annotation, the highlighted 4 lines are annotations that help you to: (1) tune batch\_size and (2) dropout\_rate, (3) report test\_acc every 100 steps, and (4) at last report test\_acc as final result.

> What noteworthy is: as these new added codes are annotations, it does not actually change your previous codes logic, you can still run your code as usual in environments without NNI installed.

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

> NOTE
> 
> > `@nni.variable` will take effect on its following line
> > 
> > `@nni.report_intermediate_result`/`@nni.report_final_result` will send the data to assessor/tuner at that line.
> > 
> > Please refer to [Annotation README](../tools/nni_annotation/README.md) for more information about annotation syntax and its usage.
> 
> 第二步：启用 NNI 标记 在 yaml 配置文件中，将 *useAnnotation* 设置为 true 来启用 NNI 标记。

    useAnnotation: true
    

## More Trial Example

* [Automatic Model Architecture Search for Reading Comprehension.](../examples/trials/ga_squad/README.md)