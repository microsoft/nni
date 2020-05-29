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
  git clone -b v1.6 https://github.com/Microsoft/nni.git
  cd nni
  powershell -ExecutionPolicy Bypass -file install.ps1
  ```

## 验证安装

以下示例基于 TensorFlow 1.x 。确保运行环境中使用的的是 **TensorFlow 1.x**。

* 通过克隆源代码下载示例。

  ```bash
  git clone -b v1.6 https://github.com/Microsoft/nni.git
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

* 在浏览器中打开 `Web UI url`，可看到下图的实验详细信息，以及所有的尝试任务。 查看[这里](../Tutorial/WebUI.md)的更多页面。

![概述](../../img/webui_overview_page.png)

![详细说明](../../img/webui_trialdetail_page.png)

## 系统需求

以下是 NNI 在 Windows 上的最低配置，推荐使用 Windows 10 1809 版。 由于程序变更，NNI 的最低配置会有所更改。

|          | 推荐配置                                      | 最低配置                                  |
| -------- | ----------------------------------------- | ------------------------------------- |
| **操作系统** | Windows 10 1809 或更高版本                     |                                       |
| **CPU**  | Intel® Core™ i5 或 AMD Phenom™ II X3 或更高配置 | Intel® Core™ i3 或 AMD Phenom™ X3 8650 |
| **GPU**  | NVIDIA® GeForce® GTX 660 或更高配置            | NVIDIA® GeForce® GTX 460              |
| **内存**   | 6 GB                                      | 4 GB                                  |
| **存储**   | 30 GB 可用的磁盘空间                             |                                       |
| **网络**   | 宽带连接                                      |                                       |
| **分辨率**  | 1024 x 768 以上                             |                                       |

## 常见问答

### 安装 NNI 时出现 simplejson 错误

确保安装了 C++ 14.0 编译器。
> building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified

### 在命令行或 PowerShell 中，Trial 因为缺少 DLL 而失败

此错误因为缺少 LIBIFCOREMD.DLL 和 LIBMMD.DLL 文件，且 SciPy 安装失败。 使用 Anaconda 或 Miniconda 和 Python（64位）可解决。
> ImportError: DLL load failed

### Web 界面上的 Trial 错误

检查 Trial 日志文件来了解详情。

如果存在 stderr 文件，也需要查看其内容。 两种可能的情况是：

* 忘记将 Experiment 配置的 Trial 命令中的 `python3` 改为 `python`。
* 忘记安装 Experiment 的依赖，如 TensorFlow，Keras 等。

### 无法在 Windows 上使用 BOHB
确保安装了 C ++ 14.0 编译器然后尝试运行 `nnictl package install --name=BOHB` 来安装依赖项。

### Windows 上不支持的 Tuner
当前不支持 SMAC，原因可参考[此问题](https://github.com/automl/SMAC3/issues/483)。

### 将 Windows 服务器用作远程服务器
目前不支持。

注意：

* 如果遇到 `Segmentation fault` 的错误，请参阅[常见问题](FAQ.md)


## 更多

* [概述](../Overview.md)
* [使用命令行工具 nnictl](Nnictl.md)
* [使用 NNIBoard](WebUI.md)
* [定制搜索空间](SearchSpaceSpec.md)
* [配置 Experiment](ExperimentConfig.md)
* [如何在本机运行 Experiment (支持多 GPU 卡)？](../TrainingService/LocalMode.md)
* [如何在多机上运行 Experiment？](../TrainingService/RemoteMachineMode.md)
* [如何在 OpenPAI 上运行 Experiment？](../TrainingService/PaiMode.md)
* [如何通过 Kubeflow 在 Kubernetes 上运行 Experiment？](../TrainingService/KubeflowMode.md)
* [如何通过 FrameworkController 在 Kubernetes 上运行 Experiment？](../TrainingService/FrameworkControllerMode.md)
