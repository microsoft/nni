# 快速入门

## 安装

目前支持 Linux、macOS 和 Windows。 Ubuntu 16.04 或更高版本、macOS 10.14.1 和 Windows 10.1809 均经过测试并支持。 在 `python >= 3.5` 的环境中，只需要运行 `pip install` 即可完成安装。

**Linux 和 macOS**

```bash
    python3 -m pip install --upgrade nni
```

**Windows**

```bash
    python -m pip install --upgrade nni
```

注意：

* 在 Linux 和 macOS 上，如果要将 NNI 安装到当前用户的 home 目录中，可使用 `--user`，则不需要特殊权限。
* 如果遇到 `Segmentation fault` 这样的错误，参考[常见问答](FAQ.md)。
* 有关 NNI 的`系统要求`，参考[在 Linux 和 macOS 上安装](InstallationLinux.md) 或 [Windows](InstallationWin.md)。

## MNIST 上的 "Hello World"

NNI 是一个能进行自动机器学习实验的工具包。 它可以自动进行获取超参、运行 Trial，测试结果，调优超参的循环。 在这里，将演示如何使用 NNI 帮助找到 MNIST 模型的最佳超参数。

这是还**没有 NNI** 的示例代码，用 CNN 在 MNIST 数据集上训练：

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

注意：完整实现请参考 [examples/trials/mnist-tfv1/mnist_before.py](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/mnist_before.py)

上面的代码一次只能尝试一组参数，如果想要调优学习率，需要手工改动超参，并一次次尝试。

NNI 用来帮助超参调优。它的流程如下：

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

如果需要使用 NNI 来自动训练模型，找到最佳超参，需要根据代码，进行如下三步改动：

**启动 Experiment 的三个步骤**

**第一步**：定义 JSON 格式的`搜索空间`文件，包括所有需要搜索的超参的`名称`和`分布`（离散和连续值均可）。

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

*实现代码：[search_space.json](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/search_space.json)*

**Step 2**: Modify your `Trial` file to get the hyperparameter set from NNI and report the final result to NNI.

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

*实现代码：[mnist.py](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/mnist.py)*

**Step 3**: Define a `config` file in YAML which declares the `path` to the search space and trial files. It also gives other information such as the tuning algorithm, max trial number, and max duration arguments.

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

Note, **for Windows, you need to change the trial command from `python3` to `python`**.

*实现代码：[config.yml](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/config.yml)*

All the cod above is already prepared and stored in [examples/trials/mnist-tfv1/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1).

**Linux 和 macOS**

Run the **config.yml** file from your command line to start an MNIST experiment.

```bash
    nnictl create --config nni/examples/trials/mnist-tfv1/config.yml
```

**Windows**

Run the **config_windows.yml** file from your command line to start an MNIST experiment.

Note: if you're using NNI on Windows, you need to change `python3` to `python` in the config.yml file or use the config_windows.yml file to start the experiment.

```bash
    nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml
```

Note: `nnictl` is a command line tool that can be used to control experiments, such as start/stop/resume an experiment, start/stop NNIBoard, etc. 查看[这里](Nnictl.md)，了解 `nnictl` 更多用法。

在命令行中等待输出 `INFO: Successfully started experiment!`。 此消息表明 Experiment 已成功启动。 And this is what we expect to get:

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

If you prepared `trial`, `search space`, and `config` according to the above steps and successfully created an NNI job, NNI will automatically tune the optimal hyper-parameters and run different hyper-parameter sets for each trial according to the requirements you set. You can clearly see its progress through the NNI WebUI.

## Web 界面

After you start your experiment in NNI successfully, you can find a message in the command-line interface that tells you the `Web UI url` like this:

```text
Web 地址为：[IP 地址]:8080
```

Open the `Web UI url` (Here it's: `[Your IP]:8080`) in your browser; you can view detailed information about the experiment and all the submitted trial jobs as shown below. If you cannot open the WebUI link in your terminal, please refer to the [FAQ](FAQ.md).

### 查看概要页面

Click the "Overview" tab.

Experiment 相关信息会显示在界面上，配置和搜索空间等。 NNI also supports downloading this information and the parameters through the **Download** button. You can download the experiment results anytime while the experiment is running, or you can wait until the end of the execution, etc.

![](../../img/QuickStart1.png)

The top 10 trials will be listed on the Overview page. You can browse all the trials on the "Trials Detail" page.

![](../../img/QuickStart2.png)

### 查看 Trial 详情页面

Click the "Default Metric" tab to see the point graph of all trials. Hover to see specific default metrics and search space messages.

![](../../img/QuickStart3.png)

Click the "Hyper Parameter" tab to see the parallel graph.

* You can select the percentage to see the top trials.
* Choose two axis to swap their positions.

![](../../img/QuickStart4.png)

Click the "Trial Duration" tab to see the bar graph.

![](../../img/QuickStart5.png)

Below is the status of all trials. 包括：

* Trial detail: trial's id, duration, start time, end time, status, accuracy, and search space file.
* 如果在 OpenPAI 平台上运行，还可以看到 hdfsLog。
* Kill: you can kill a job that has the `Running` status.
* Support: Used to search for a specific trial.

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