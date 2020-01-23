# 快速入门

## 安装

We support Linux macOS and Windows in current stage, Ubuntu 16.04 or higher, macOS 10.14.1 and Windows 10.1809 are tested and supported. 在 `python >= 3.5` 的环境中，只需要运行 `pip install` 即可完成安装。

**Linux and macOS**

```bash
    python3 -m pip install --upgrade nni
```

**Windows**

```bash
    python -m pip install --upgrade nni
```

Note:

* For Linux and macOS `--user` can be added if you want to install NNI in your home directory, which does not require any special privileges.
* 如果遇到如`Segmentation fault` 这样的任何错误请参考[常见问题](FAQ.md)。
* 参考[安装 NNI](Installation.md)，来了解`系统需求`。

## MNIST 上的 "Hello World"

NNI is a toolkit to help users run automated machine learning experiments. It can automatically do the cyclic process of getting hyperparameters, running trials, testing results, tuning hyperparameters. Now, we show how to use NNI to help you find the optimal hyperparameters.

Here is an example script to train a CNN on MNIST dataset **without NNI**:

```python
def run_trial(params):
    # 输入数据
    mnist = input_data.read_data_sets(params['data_dir'], one_hot=True)
    # 构建网络
    mnist_network = MnistNetwork(channel_1_num=params['channel_1_num'], channel_2_num=params['channel_2_num'], conv_size=params['conv_size'], hidden_size=params['hidden_size'], pool_size=params['pool_size'], learning_rate=params['learning_rate'])
    mnist_network.build_network()

    test_acc = 0.0
    with tf.Session() as sess:
        # 训练网络
        mnist_network.train(sess, mnist)
        # 评估网络
        test_acc = mnist_network.evaluate(mnist)

if __name__ == '__main__':
    params = {'data_dir': '/tmp/tensorflow/mnist/input_data', 'dropout_rate': 0.5, 'channel_1_num': 32, 'channel_2_num': 64, 'conv_size': 5, 'pool_size': 2, 'hidden_size': 1024, 'learning_rate': 1e-4, 'batch_num': 2000, 'batch_size': 32}
    run_trial(params)
```

Note: If you want to see the full implementation, please refer to [examples/trials/mnist-tfv1/mnist_before.py](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/mnist_before.py)

The above code can only try one set of parameters at a time, if we want to tune learning rate, we need to manually modify the hyperparameter and start the trial again and again.

NNI is born for helping user do the tuning jobs, the NNI working process is presented below:

```text
输入: 搜索空间, Trial 代码, 配置文件
输出: 一组最佳的超参配置

1: For t = 0, 1, 2, ..., maxTrialNum,
2:      hyperparameter = 从搜索空间选择一组参数
3:      final result = run_trial_and_evaluate(hyperparameter)
4:      返回最终结果给 NNI
5:      If 时间达到上限,
6:          停止实验
7: return 最好的实验结果
```

If you want to use NNI to automatically train your model and find the optimal hyper-parameters, you need to do three changes base on your code:

**Three steps to start an experiment**

**Step 1**: Give a `Search Space` file in JSON, includes the `name` and the `distribution` (discrete valued or continuous valued) of all the hyperparameters you need to search.

```diff
-   params = {'data_dir': '/tmp/tensorflow/mnist/input_data', 'dropout_rate': 0.5, 'channel_1_num': 32, 'channel_2_num': 64,
-   'conv_size': 5, 'pool_size': 2, 'hidden_size': 1024, 'learning_rate': 1e-4, 'batch_num': 2000, 'batch_size': 32}
+ {
+     "dropout_rate":{"_type":"uniform","_value":[0.5, 0.9]},
+     "conv_size":{"_type":"choice","_value":[2,3,5,7]},
+     "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
+     "batch_size": {"_type":"choice", "_value": [1, 4, 8, 16, 32]},
+     "learning_rate":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]}
+ }
```

*Implemented code directory: [search_space.json](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/search_space.json)*

**Step 2**: Modified your `Trial` file to get the hyperparameter set from NNI and report the final result to NNI.

```diff
+ import nni

  def run_trial(params):
      mnist = input_data.read_data_sets(params['data_dir'], one_hot=True)

      mnist_network = MnistNetwork(channel_1_num=params['channel_1_num'], channel_2_num=params['channel_2_num'], conv_size=params['conv_size'], hidden_size=params['hidden_size'], pool_size=params['pool_size'], learning_rate=params['learning_rate'])
      mnist_network.build_network()

      with tf.Session() as sess:
          mnist_network.train(sess, mnist)
          test_acc = mnist_network.evaluate(mnist)

+         nni.report_final_result(test_acc)

  if __name__ == '__main__':

-     params = {'data_dir': '/tmp/tensorflow/mnist/input_data', 'dropout_rate': 0.5, 'channel_1_num': 32, 'channel_2_num': 64,
-     'conv_size': 5, 'pool_size': 2, 'hidden_size': 1024, 'learning_rate': 1e-4, 'batch_num': 2000, 'batch_size': 32}
+     params = nni.get_next_parameter()
      run_trial(params)
```

*Implemented code directory: [mnist.py](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/mnist.py)*

**Step 3**: Define a `config` file in YAML, which declare the `path` to search space and trial, also give `other information` such as tuning algorithm, max trial number and max duration arguments.

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
trainingServicePlatform: local
# 搜索空间文件
searchSpacePath: search_space.json
useAnnotation: false
tuner:
  builtinTunerName: TPE
# 运行的命令，以及 Trial 代码的路径
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
```

Note, **for Windows, you need to change trial command `python3` to `python`**

*Implemented code directory: [config.yml](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/config.yml)*

All the codes above are already prepared and stored in [examples/trials/mnist-tfv1/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1).

**Linux and macOS**

Run the **config.yml** file from your command line to start MNIST experiment.

```bash
    nnictl create --config nni/examples/trials/mnist-tfv1/config.yml
```

**Windows**

Run the **config_windows.yml** file from your command line to start MNIST experiment.

Note, if you're using NNI on Windows, it needs to change `python3` to `python` in the config.yml file, or use the config_windows.yml file to start the experiment.

```bash
    nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml
```

Note, `nnictl` is a command line tool, which can be used to control experiments, such as start/stop/resume an experiment, start/stop NNIBoard, etc. Click [here](Nnictl.md) for more usage of `nnictl`

Wait for the message `INFO: Successfully started experiment!` in the command line. This message indicates that your experiment has been successfully started. And this is what we expected to get:

```text
INFO: Starting restful server...
INFO: Successfully started Restful server!
INFO: Setting local config...
INFO: Successfully set local config!
INFO: Starting experiment...
INFO: Successfully started experiment!
-----------------------------------------------------------------------
The experiment id is egchD4qy
The Web UI urls are: [Your IP]:8080
-----------------------------------------------------------------------

You can use these commands to get more information about the experiment
-----------------------------------------------------------------------
         commands                       description

1. nnictl experiment show        show the information of experiments
2. nnictl trial ls               list all of trial jobs
3. nnictl top                    monitor the status of running experiments
4. nnictl log stderr             show stderr log content
5. nnictl log stdout             show stdout log content
6. nnictl stop                   stop an experiment
7. nnictl trial kill             kill a trial job by id
8. nnictl --help                 get help information about nnictl
-----------------------------------------------------------------------
```

If you prepare `trial`, `search space` and `config` according to the above steps and successfully create a NNI job, NNI will automatically tune the optimal hyper-parameters and run different hyper-parameters sets for each trial according to the requirements you set. You can clearly sees its progress by NNI WebUI.

## Web 界面

After you start your experiment in NNI successfully, you can find a message in the command-line interface to tell you `Web UI url` like this:

```text
Web 地址为：[IP 地址]:8080
```

Open the `Web UI url`(In this information is: `[Your IP]:8080`) in your browser, you can view detail information of the experiment and all the submitted trial jobs as shown below. If you can not open the WebUI link in your terminal, you can refer to [FAQ](FAQ.md).

### View summary page

Click the tab "Overview".

Information about this experiment will be shown in the WebUI, including the experiment trial profile and search space message. NNI also support `download these information and parameters` through the **Download** button. You can download the experiment result anytime in the middle for the running or at the end of the execution, etc.

![](../../img/QuickStart1.png)

Top 10 trials will be listed in the Overview page, you can browse all the trials in "Trials Detail" page.

![](../../img/QuickStart2.png)

### View trials detail page

Click the tab "Default Metric" to see the point graph of all trials. Hover to see its specific default metric and search space message.

![](../../img/QuickStart3.png)

Click the tab "Hyper Parameter" to see the parallel graph.

* 可选择百分比查看最好的 Trial。
* 选择两个轴来交换位置。

![](../../img/QuickStart4.png)

Click the tab "Trial Duration" to see the bar graph.

![](../../img/QuickStart5.png)

Below is the status of the all trials. Specifically:

* Trial 详情：Trial 的 id，持续时间，开始时间，结束时间，状态，精度和搜索空间。
* 如果在 OpenPAI 平台上运行，还可以看到 hdfsLog。
* Kill: 可终止正在运行的任务。
* 支持搜索某个特定的 Trial。

![](../../img/QuickStart6.png)

* 中间结果图

![](../../img/QuickStart7.png)

## 相关主题

* [尝试不同的 Tuner](../Tuner/BuiltinTuner.md)
* [尝试不同的 Assessor](../Assessor/BuiltinAssessor.md)
* [使用命令行工具 nnictl](Nnictl.md)
* [如何实现 Trial 代码](../TrialExample/Trials.md)
* [如何在本机运行 Experiment (支持多 GPU 卡)？](../TrainingService/LocalMode.md)
* [如何在多机上运行 Experiment？](../TrainingService/RemoteMachineMode.md)
* [如何在 OpenPAI 上运行 Experiment？](../TrainingService/PaiMode.md)
* [如何通过 Kubeflow 在 Kubernetes 上运行 Experiment？](../TrainingService/KubeflowMode.md)
* [如何通过 FrameworkController 在 Kubernetes 上运行 Experiment？](../TrainingService/FrameworkControllerMode.md)