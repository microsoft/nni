# 在 Windows 上安装

## 先决条件

* Python 3.5（或以上）64 位。 在 Windows 上推荐使用 [Anaconda](https://www.anaconda.com/products/individual) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 来管理多个 Python 环境。

* 如果是新安装的 Python 环境，需要安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 来支持 NNI 的依赖项，如 `scikit-learn`。

    ```bat
    pip install cython wheel
    ```

* 安装 git 用于验证安装。

## 安装 NNI

大多数情况下，可以从 pip 包安装和升级 NNI。 这样既方便又快捷。

如果对某个或最新版本的代码感兴趣，可通过源代码安装 NNI。

If you want to contribute to NNI, refer to [setup development environment](SetupNniDeveloperEnvironment.md).

* From pip package

    ```bat
    python -m pip install --upgrade nni
    ```

* From source code

    ```bat
    git clone -b v1.5 https://github.com/Microsoft/nni.git
    cd nni
    powershell -ExecutionPolicy Bypass -file install.ps1
    ```

## Verify installation

The following example is built on TensorFlow 1.x. Make sure **TensorFlow 1.x is used** when running it.

* Clone examples within source code.

    ```bat
    git clone -b v1.5 https://github.com/Microsoft/nni.git
    ```

* Run the MNIST example.

    ```bat
    nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml
    ```

    Note:  If you are familiar with other frameworks, you can choose corresponding example under `examples\trials`. It needs to change trial command `python3` to `python` in each example YAML, since default installation has `python.exe`, not `python3.exe` executable.

* Wait for the message `INFO: Successfully started experiment!` in the command line. This message indicates that your experiment has been successfully started. You can explore the experiment using the `Web UI url`.

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

* Open the `Web UI url` in your browser, you can view detailed information about the experiment and all the submitted trial jobs as shown below. [Here](../Tutorial/WebUI.md) are more Web UI pages.

![overview](../../img/webui_overview_page.png)

![detail](../../img/webui_trialdetail_page.png)

## System requirements

Below are the minimum system requirements for NNI on Windows, Windows 10.1809 is well tested and recommend. Due to potential programming changes, the minimum system requirements for NNI may change over time.

|          | 推荐配置                                      | 最低配置                                  |
| -------- | ----------------------------------------- | ------------------------------------- |
| **操作系统** | Windows 10 1809 或更高版本                     |                                       |
| **CPU**  | Intel® Core™ i5 或 AMD Phenom™ II X3 或更高配置 | Intel® Core™ i3 或 AMD Phenom™ X3 8650 |
| **GPU**  | NVIDIA® GeForce® GTX 660 或更高配置            | NVIDIA® GeForce® GTX 460              |
| **内存**   | 6 GB                                      | 4 GB                                  |
| **存储**   | 30 GB 可用的磁盘空间                             |                                       |
| **网络**   | 宽带连接                                      |                                       |
| **分辨率**  | 1024 x 768 以上                             |                                       |

## FAQ

### simplejson failed when installing NNI

Make sure a C++ 14.0 compiler is installed.
> building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified

### Trial failed with missing DLL in command line or PowerShell

This error is caused by missing LIBIFCOREMD.DLL and LIBMMD.DLL and failure to install SciPy. Using Anaconda or Miniconda with Python(64-bit) can solve it.
> ImportError: DLL load failed

### Trial failed on webUI

Please check the trial log file stderr for more details.

If there is a stderr file, please check it. Two possible cases are:

* forgetting to change the trial command `python3` to `python` in each experiment YAML.
* forgetting to install experiment dependencies such as TensorFlow, Keras and so on.

### Fail to use BOHB on Windows

Make sure a C++ 14.0 compiler is installed when trying to run `nnictl package install --name=BOHB` to install the dependencies.

### Not supported tuner on Windows

SMAC is not supported currently; for the specific reason refer to this [GitHub issue](https://github.com/automl/SMAC3/issues/483).

### Use Windows as a remote worker

Refer to [Remote Machine mode](../TrainingService/RemoteMachineMode.md).

### Segmentation fault (core dumped) when installing

Refer to [FAQ](FAQ.md).

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
