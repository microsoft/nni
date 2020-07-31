# 在 Linux 和 Mac 下安装

## 安装

在 Linux 和 macOS 上安装，遵循以下相同的说明。

### 通过 pip 命令安装 NNI

先决条件：`python 64-bit >= 3.5`

    bash
      python3 -m pip install --upgrade nni

### 通过源代码安装 NNI

如果对某个或最新版本的代码感兴趣，可通过源代码安装 NNI。

先决条件：`python 64-bit >=3.5`, `git`, `wget`

    bash
      git clone -b v1.7 https://github.com/Microsoft/nni.git
      cd nni
      ./install.sh

### 在 Docker 映像中使用 NNI

也可将 NNI 安装到 docker 映像中。 参考[这里](../deployment/docker/README.md)来生成 NNI 的 Docker 映像。 也可通过此命令从 Docker Hub 中直接拉取 NNI 的映像 `docker pull msranni/nni:latest`。

## 验证安装

以下示例基于 TensorFlow 1.x 。确保运行环境中使用的的是 ** TensorFlow 1.x**。

* 通过克隆源代码下载示例。
    
    ```bash
    git clone -b v1.7 https://github.com/Microsoft/nni.git
    ```

* 运行 MNIST 示例。
    
    ```bash
    nnictl create --config nni/examples/trials/mnist-tfv1/config.yml
    ```

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

由于程序变更，NNI 的最低配置会有所更改。

### Linux

|          | 推荐配置                                      | 最低配置                                  |
| -------- | ----------------------------------------- | ------------------------------------- |
| **操作系统** | Ubuntu 16.04 或以上版本                        |                                       |
| **CPU**  | Intel® Core™ i5 或 AMD Phenom™ II X3 或更高配置 | Intel® Core™ i3 或 AMD Phenom™ X3 8650 |
| **GPU**  | NVIDIA® GeForce® GTX 660 或更高配置            | NVIDIA® GeForce® GTX 460              |
| **内存**   | 6 GB                                      | 4 GB                                  |
| **存储**   | 30 GB 可用的磁盘空间                             |                                       |
| **网络**   | 宽带连接                                      |                                       |
| **分辨率**  | 1024 x 768 以上                             |                                       |

### macOS

|          | 推荐配置                     | 最低配置                                               |
| -------- | ------------------------ | -------------------------------------------------- |
| **操作系统** | macOS 10.14.1 或更高版本      |                                                    |
| **CPU**  | Intel® Core™ i7-4770 或更高 | Intel® Core™ i5-760 或更高                            |
| **GPU**  | AMD Radeon™ R9 M395X 或更高 | NVIDIA® GeForce® GT 750M 或 AMD Radeon™ R9 M290 或更高 |
| **内存**   | 8 GB                     | 4 GB                                               |
| **存储**   | 70GB 可用空间 SSD 硬盘         | 70GB 可用空间及 7200 RPM 硬盘                             |
| **网络**   | 宽带连接                     |                                                    |
| **分辨率**  | 1024 x 768 以上            |                                                    |

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