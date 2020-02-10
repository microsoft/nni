# 在 Windows 上安装

## 安装

强烈建议使用 Anaconda 或 Miniconda 来管理多个 Python 环境。

### 通过 pip 命令安装 NNI

  先决条件：`python 64-bit >= 3.5`

  ```bash
  python -m pip install --upgrade nni
  ```

### 通过源代码安装 NNI

  如果对某个或最新版本的代码感兴趣，可通过源代码安装 NNI。

  先决条件：`python 64-bit >=3.5`, `git`, `PowerShell`

  ```bash
  git clone -b v1.3 https://github.com/Microsoft/nni.git
  cd nni
  powershell -ExecutionPolicy Bypass -file install.ps1
  ```

## 验证安装

以下示例基于 TensorFlow 1.x 。确保运行环境中使用的的是 **TensorFlow 1.x**。

* 通过克隆源代码下载示例。

  ```bash
  git clone -b v1.3 https://github.com/Microsoft/nni.git
  ```

* 运行 MNIST 示例。

  ```bash
  nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml
  ```

  注意：在其它示例中，如果 Python3 是通过 `python` 命令启动，需要将每个示例 YAML 文件的 Trial 命令中的 `python3` 改为 `python`。

* 在命令行中等待输出 `INFO: Successfully started experiment!`。 此消息表明 Experiment 已成功启动。 通过命令行输出的 `Web UI url` 来访问 Experiment 的界面。

```text
INFO: Starting restful server...
INFO: Successfully started Restful server!
INFO: Setting local config...
INFO: Successfully set local config!
INFO: Starting experiment...
INFO: Successfully started experiment!
-----------------------------------------------------------------------
The experiment id is egchD4qy
The Web UI urls are: http://223.255.255.1:8080   http://127.0.0.1:8080
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

* 在浏览器中打开 `Web UI url`，可看到下图的 Experiment 详细信息，以及所有的 Trial 任务。 查看[这里](../Tutorial/WebUI.md)的更多页面。

![概述](../../img/webui_overview_page.png)

![详细说明](../../img/webui_trialdetail_page.png)

## 系统需求

以下是 NNI 在 Windows 上的最低配置，推荐使用 Windows 10 1809 版。 由于程序变更，NNI 的最低配置会有所更改。

|                | 推荐配置                                      | 最低配置                                  |
| -------------- | ----------------------------------------- | ------------------------------------- |
| **操作系统**       | Windows 10 1809 或更高版本                     |                                       |
| **CPU**        | Intel® Core™ i5 或 AMD Phenom™ II X3 或更高配置 | Intel® Core™ i3 或 AMD Phenom™ X3 8650 |
| **GPU**        | NVIDIA® GeForce® GTX 660 or better        | NVIDIA® GeForce® GTX 460              |
| **Memory**     | 6 GB RAM                                  | 4 GB RAM                              |
| **Storage**    | 30 GB available hare drive space          |                                       |
| **Internet**   | Boardband internet connection             |                                       |
| **Resolution** | 1024 x 768 minimum display resolution     |                                       |

## FAQ

### simplejson failed when installing NNI

Make sure C++ 14.0 compiler installed.
> building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified

### Trial failed with missing DLL in command line or PowerShell

This error caused by missing LIBIFCOREMD.DLL and LIBMMD.DLL and fail to install SciPy. Using Anaconda or Miniconda with Python(64-bit) can solve it.
> ImportError: DLL load failed

### Trial failed on webUI

Please check the trial log file stderr for more details.

If there is a stderr file, please check out. Two possible cases are as follows:

* forget to change the trial command `python3` into `python` in each experiment YAML.
* forget to install experiment dependencies such as TensorFlow, Keras and so on.

### Fail to use BOHB on Windows
Make sure C++ 14.0 compiler installed then try to run `nnictl package install --name=BOHB` to install the dependencies.

### Not supported tuner on Windows
SMAC is not supported currently, the specific reason can be referred to this [GitHub issue](https://github.com/automl/SMAC3/issues/483).

### Use a Windows server as a remote worker
Currently you can't.

Note:

* If there is any error like `Segmentation fault`, please refer to [FAQ](FAQ.md)


## Further reading

* [Overview](../Overview.md)
* [Use command line tool nnictl](Nnictl.md)
* [Use NNIBoard](WebUI.md)
* [Define search space](SearchSpaceSpec.md)
* [Config an experiment](ExperimentConfig.md)
* [How to run an experiment on local (with multiple GPUs)?](../TrainingService/LocalMode.md)
* [How to run an experiment on multiple machines?](../TrainingService/RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](../TrainingService/PaiMode.md)
* [How to run an experiment on Kubernetes through Kubeflow?](../TrainingService/KubeflowMode.md)
* [How to run an experiment on Kubernetes through FrameworkController?](../TrainingService/FrameworkControllerMode.md)