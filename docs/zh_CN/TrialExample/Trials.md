# 实现 NNI 的 Trial（尝试）代码

A **Trial** in NNI is an individual attempt at applying a configuration (e.g., a set of hyper-parameters) to a model.

To define an NNI trial, you need to first define the set of parameters (i.e., search space) and then update the model. NNI provides two approaches for you to define a trial: [NNI API](#nni-api) and [NNI Python annotation](#nni-annotation). 参考[这里的](#more-examples)更多 Trial 示例。

<a name="nni-api"></a>

## NNI API

### 第一步：准备搜索空间参数文件。

示例如下：

```json
{
    "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
}
```

Refer to [SearchSpaceSpec.md](../Tutorial/SearchSpaceSpec.md) to learn more about search spaces. Tuner 会根据搜索空间来生成配置，即从每个超参的范围中选一个值。

### Step 2 - Update model code

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

`metrics` can be any python object. If users use the NNI built-in tuner/assessor, `metrics` can only have two formats: 1) a number e.g., float, int, or 2) a dict object that has a key named `default` whose value is a number. These `metrics` are reported to [assessor](../Assessor/BuiltinAssessor.md). Often, `metrics` includes the periodically evaluated loss or accuracy.

* 返回配置的最终性能

```python
nni.report_final_result(metrics)
```

`metrics` can also be any python object. If users use the NNI built-in tuner/assessor, `metrics` follows the same format rule as that in `report_intermediate_result`, the number indicates the model's performance, for example, the model's accuracy, loss etc. These `metrics` are reported to [tuner](../Tuner/BuiltinTuner.md).

### 第三步：启用 NNI API

To enable NNI API mode, you need to set useAnnotation to *false* and provide the path of the SearchSpace file was defined in step 1:

```yaml
useAnnotation: false
searchSpacePath: /path/to/your/search_space.json
```

参考[这里](../Tutorial/ExperimentConfig.md)进一步了解如何配置 Experiment。

* 参考[这里](https://nni.readthedocs.io/zh/latest/sdk_reference.html)，了解更多 NNI API (例如 `nni.get_sequence_id()`)。

<a name="nni-annotation"></a>

## NNI Annotation

另一种实现 Trial 的方法是使用 Python 注释来标记 NNI。 NNI annotations are simple, similar to comments. You don't have to make structural changes to your existing code. 只需要添加一些 NNI Annotation，就能够：

* 标记需要调整的参数变量
* specify the range in which you want to tune the variables
* annotate which variable you want to report as an intermediate result to `assessor`
* 标记哪个变量需要作为最终结果（例如：模型精度）返回给 `Tuner`。

同样以 MNIST 为例，只需要两步就能用 NNI Annotation 来实现 Trial 代码。

### 第一步：在代码中加入 Annotation

The following is a TensorFlow code snippet for NNI Annotation where the highlighted four lines are annotations that:

1. 调优 batch\_size 和 dropout\_rate
2. 每执行 100 步返回 test\_acc
3. lastly report test\_acc as the final result.

It's worth noting that, as these newly added codes are merely annotations, you can still run your code as usual in environments without NNI installed.

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

* `@nni.variable` will affect its following line which should be an assignment statement whose left-hand side must be the same as the keyword `name` in the `@nni.variable` statement.
* `@nni.report_intermediate_result`/`@nni.report_final_result` 会将数据发送给 Assessor、Tuner。

Annotation 的语法和用法等，参考 [Annotation](../Tutorial/AnnotationSpec.md)。

### 第二步：启用 Annotation

在 YAML 配置文件中设置 *useAnnotation* 为 true 来启用 Annotation：

    useAnnotation: true
    

## Standalone mode for debugging

NNI supports a standalone mode for trial code to run without starting an NNI experiment. 这样能更容易的找出 Trial 代码中的 Bug。 NNI Annotation 天然支持独立模式，因为添加的 NNI 相关的行都是注释的形式。 NNI Trial API 在独立模式下的行为有所变化，某些 API 返回虚拟值，而某些 API 不报告值。 有关这些 API 的完整列表，请参阅下表。

```python
＃注意：请为 Trial 代码中的超参分配默认值
nni.get_next_parameter＃返回 {}
nni.report_final_result＃已在 stdout 上打印日志，但不报告
nni.report_intermediate_result＃已在 stdout 上打印日志，但不报告
nni.get_experiment_id＃返回 "STANDALONE"
nni.get_trial_id＃返回 "STANDALONE"
nni.get_sequence_id＃返回 0
```

可使用 [mnist 示例](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-tfv1) 来尝试独立模式。 只需在代码目录下运行 `python3 mnist.py`。 The trial code should successfully run with the default hyperparameter values.

For more information on debugging, please refer to [How to Debug](../Tutorial/HowToDebug.md)

## Trial 存放在什么地方？

### 本机模式

每个 Trial 都有单独的目录来输出自己的数据。 在每次 Trial 运行后，环境变量 `NNI_OUTPUT_DIR` 定义的目录都会被导出。 Under this directory, you can find each trial's code, data, and other logs. 此外，Trial 的日志（包括 stdout）还会被重定向到此目录中的 `trial.log` 文件。

If NNI Annotation is used, the trial's converted code is in another temporary directory. 可以在 `run.sh` 文件中的 `NNI_OUTPUT_DIR` 变量找到此目录。 文件中的第二行（即：`cd`）会切换到代码所在的实际路径。 参考 `run.sh` 文件示例：

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

When running trials on other platforms like remote machine or PAI, the environment variable `NNI_OUTPUT_DIR` only refers to the output directory of the trial, while the trial code and `run.sh` might not be there. However, the `trial.log` will be transmitted back to the local machine in the trial's directory, which defaults to `~/nni/experiments/$experiment_id$/trials/$trial_id$/`

For more information, please refer to [HowToDebug](../Tutorial/HowToDebug.md).

<a name="more-examples"></a>

## 更多 Trial 的示例

* [MNIST 示例](MnistExamples.md)
* [为 CIFAR 10 分类找到最佳的 optimizer](Cifar10Examples.md)
* [如何在 NNI 调优 SciKit-learn 的参数](SklearnExamples.md)
* [在阅读理解上使用自动模型架构搜索。](SquadEvolutionExamples.md)
* [如何在 NNI 上调优 GBDT](GbdtExample.md)
* [在 NNI 上调优 RocksDB](RocksdbExamples.md)