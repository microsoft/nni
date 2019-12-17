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

参考 [SearchSpaceSpec.md](../Tutorial/SearchSpaceSpec.md) 进一步了解搜索空间。 Tuner 会根据搜索空间来生成配置，即从每个超参的范围中选一个值。

### 第二步：更新模型代码

* Import NNI
    
    在 Trial 代码中加上 `import nni`。

* 从 Tuner 获得参数值

```python
RECEIVED_PARAMS = nni.get_next_parameter()
```

`RECEIVED_PARAMS` 是一个对象，如：

`{"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}`.

* 定期返回指标数据（可选）

```python
nni.report_intermediate_result(metrics)
```

`指标`可以是任意的 Python 对象。 如果使用了 NNI 内置的 Tuner/Assessor，`指标`只可以是两种类型：1) 数值类型，如 float、int， 2) dict 对象，其中必须由键名为 `default`，值为数值的项目。 `指标`会发送给 [Assessor](../Assessor/BuiltinAssessor.md)。 通常，`指标`是损失值或精度。

* 返回配置的最终性能

```python
nni.report_final_result(metrics)
```

`指标`也可以是任意的 Python 对象。 如果使用了内置的 Tuner/Assessor，`指标`格式和 `report_intermediate_result` 中一样，这个数值表示模型的性能，如精度、损失值等。 `指标`会发送给 [Tuner](../Tuner/BuiltinTuner.md)。

### 第三步：启用 NNI API

要启用 NNI 的 API 模式，需要将 useAnnotation 设置为 *false*，并提供搜索空间文件的路径（即第一步中定义的文件）：

```yaml
useAnnotation: false
searchSpacePath: /path/to/your/search_space.json
```

参考[这里](../Tutorial/ExperimentConfig.md)进一步了解如何配置 Experiment。

* 参考[这里](https://nni.readthedocs.io/zh/latest/sdk_reference.html)，了解更多 NNI API (例如 `nni.get_sequence_id()`)。

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
+       """@nni.variable(nni.choice(0.1, 0.5), name=dropout_rate)"""
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

Annotation 的语法和用法等，参考 [Annotation](../Tutorial/AnnotationSpec.md)。

### 第二步：启用 Annotation

在 YAML 配置文件中设置 *useAnnotation* 为 true 来启用 Annotation：

    useAnnotation: true
    

## 用于调试的独立模式

NNI 支持独立模式，使 Trial 代码无需启动 NNI 实验即可运行。 这样能更容易的找出 Trial 代码中的 Bug。 NNI Annotation 天然支持独立模式，因为添加的 NNI 相关的行都是注释的形式。 For NNI trial APIs, the APIs have changed behaviors in standalone mode, some APIs return dummy values, and some APIs do not really report values. Please refer to the following table for the full list of these APIs.

```python
# NOTE: please assign default values to the hyperparameters in your trial code
nni.get_next_parameter # return {}
nni.report_final_result # have log printed on stdout, but does not report
nni.report_intermediate_result # have log printed on stdout, but does not report
nni.get_experiment_id # return "STANDALONE"
nni.get_trial_id # return "STANDALONE"
nni.get_sequence_id # return 0
```

You can try standalone mode with the [mnist example](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-tfv1). Simply run `python3 mnist.py` under the code directory. The trial code successfully runs with default hyperparameter values.

For more debuggability, please refer to [How to Debug](../Tutorial/HowToDebug.md)

## Where are my trials?

### 本机模式

In NNI, every trial has a dedicated directory for them to output their own data. In each trial, an environment variable called `NNI_OUTPUT_DIR` is exported. Under this directory, you could find each trial's code, data and other possible log. In addition, each trial's log (including stdout) will be re-directed to a file named `trial.log` under that directory.

If NNI Annotation is used, trial's converted code is in another temporary directory. You can check that in a file named `run.sh` under the directory indicated by `NNI_OUTPUT_DIR`. The second line (i.e., the `cd` command) of this file will change directory to the actual directory where code is located. Below is an example of `run.sh`:

```bash
#!/bin/bash
cd /tmp/user_name/nni/annotation/tmpzj0h72x6 #This is the actual directory
export NNI_PLATFORM=local
export NNI_SYS_DIR=/home/user_name/nni/experiments/$experiment_id$/trials/$trial_id$
export NNI_TRIAL_JOB_ID=nrbb2
export NNI_OUTPUT_DIR=/home/user_name/nni/experiments/$eperiment_id$/trials/$trial_id$
export NNI_TRIAL_SEQ_ID=1
export MULTI_PHASE=false
export CUDA_VISIBLE_DEVICES=
eval python3 mnist.py 2>/home/user_name/nni/experiments/$experiment_id$/trials/$trial_id$/stderr
echo $? `date +%s%3N` >/home/user_name/nni/experiments/$experiment_id$/trials/$trial_id$/.nni/state
```

### 其它模式

When running trials on other platform like remote machine or PAI, the environment variable `NNI_OUTPUT_DIR` only refers to the output directory of the trial, while trial code and `run.sh` might not be there. However, the `trial.log` will be transmitted back to local machine in trial's directory, which defaults to `~/nni/experiments/$experiment_id$/trials/$trial_id$/`

For more information, please refer to [HowToDebug](../Tutorial/HowToDebug.md)

<a name="more-examples"></a>

## More Trial Examples

* [MNIST 样例](MnistExamples.md)
* [为 CIFAR 10 分类找到最佳的 optimizer](Cifar10Examples.md)
* [如何在 NNI 调优 SciKit-learn 的参数](SklearnExamples.md)
* [在阅读理解上使用自动模型架构搜索。](SquadEvolutionExamples.md)
* [如何在 NNI 上调优 GBDT](GbdtExample.md)
* [在 NNI 上调优 RocksDB](RocksdbExamples.md)