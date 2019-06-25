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

`RECEIVED_PARAMS` is an object, for example:

`{"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}`.

* 定期返回指标数据（可选）

```python
nni.report_intermediate_result(metrics)
```

`metrics` could be any python object. If users use NNI built-in tuner/assessor, `metrics` can only have two formats: 1) a number e.g., float, int, 2) a dict object that has a key named `default` whose value is a number. This `metrics` is reported to [assessor](BuiltinAssessors.md). Usually, `metrics` could be periodically evaluated loss or accuracy.

* 返回配置的最终性能

```python
nni.report_final_result(metrics)
```

`metrics` also could be any python object. If users use NNI built-in tuner/assessor, `metrics` follows the same format rule as that in `report_intermediate_result`, the number indicates the model's performance, for example, the model's accuracy, loss etc. This `metrics` is reported to [tuner](BuiltinTuner.md).

### 第三步：启用 NNI API

To enable NNI API mode, you need to set useAnnotation to *false* and provide the path of SearchSpace file (you just defined in step 1):

```yaml
useAnnotation: false
searchSpacePath: /path/to/your/search_space.json
```

You can refer to [here](ExperimentConfig.md) for more information about how to set up experiment configurations.

*Please refer to [here](https://nni.readthedocs.io/en/latest/sdk_reference.html) for more APIs (e.g., `nni.get_sequence_id()`) provided by NNI.

<a name="nni-annotation"></a>

## NNI Annotation

An alternative to writing a trial is to use NNI's syntax for python. Simple as any annotation, NNI annotation is working like comments in your codes. You don't have to make structure or any other big changes to your existing codes. With a few lines of NNI annotation, you will be able to:

* 标记需要调整的参数变量
* 指定变量的搜索空间范围
* 标记哪个变量需要作为中间结果范围给 `Assessor`
* 标记哪个变量需要作为最终结果（例如：模型精度）返回给 `Tuner`。

Again, take MNIST as an example, it only requires 2 steps to write a trial with NNI Annotation.

### 第一步：在代码中加入 Annotation

The following is a tensorflow code snippet for NNI Annotation, where the highlighted four lines are annotations that help you to:

1. 调优 batch\_size 和 dropout\_rate
2. 每执行 100 步返回 test\_acc
3. 最后返回 test\_acc 作为最终结果。

What noteworthy is: as these newly added codes are annotations, it does not actually change your previous codes logic, you can still run your code as usual in environments without NNI installed.

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

**NOTE**:

* `@nni.variable` 会对它的下面一行进行修改，左边被赋值变量必须在 `@nni.variable` 的 `name` 参数中指定。
* `@nni.report_intermediate_result`/`@nni.report_final_result` 会将数据发送给 Assessor、Tuner。

For more information about annotation syntax and its usage, please refer to [Annotation](AnnotationSpec.md).

### 第二步：启用 Annotation

In the YAML configure file, you need to set *useAnnotation* to true to enable NNI annotation:

    useAnnotation: true
    

## Trial 存放在什么地方？

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

For more information, please refer to [HowToDebug](HowToDebug.md)

<a name="more-examples"></a>

## 更多 Trial 的样例

* [MNIST 样例](MnistExamples.md)
* [为 CIFAR 10 分类找到最佳的 optimizer](Cifar10Examples.md)
* [如何在 NNI 调优 SciKit-learn 的参数](SklearnExamples.md)
* [在阅读理解上使用自动模型架构搜索。](SquadEvolutionExamples.md)
* [如何在 NNI 上调优 GBDT](GbdtExample.md)