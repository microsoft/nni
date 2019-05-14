# 安装 NNI

当前支持在 Linux，Mac 和 Windows（本机模式）下安装。

## **在 Linux 和 Mac 下安装**

* **通过 pip 命令安装 NNI**
    
    先决条件：`python >= 3.5`
    
    ```bash
    python3 -m pip install --upgrade nni
    ```

* **通过源代码安装 NNI**
    
    先决条件：`python >=3.5`, `git`, `wget`
    
    ```bash
    git clone -b v0.7 https://github.com/Microsoft/nni.git
    cd nni
    ./install.sh
    ```

* **在 docker 映像中安装 NNI**
    
    也可将 NNI 安装到 docker 映像中。 参考[这里](../deployment/docker/README.md)来生成 NNI 的 Docker 映像。 也可通过此命令从 Docker Hub 中直接拉取 NNI 的映像 `docker pull msranni/nni:latest`。

## **在 Windows 上安装**

在第一次使用 PowerShell 运行脚本时，需要用**使用管理员权限**运行如下命令：

    bash
      Set-ExecutionPolicy -ExecutionPolicy Unrestricted

推荐使用 Anaconda 或 Miniconda。

* **通过 pip 命令安装 NNI**
    
    先决条件：`python(64-bit) >= 3.5`
    
    ```bash
    python -m pip install --upgrade nni
    ```

* **通过源代码安装 NNI**
    
    Prerequisite: `python >=3.5`, `git`, `PowerShell`.
    
    you can install NNI as administrator or current user as follows:
    
    ```bash
    git clone -b v0.7 https://github.com/Microsoft/nni.git
    cd nni
    powershell ./install.ps1
    ```

## **系统需求**

Below are the minimum system requirements for NNI on Linux. 由于程序变更，NNI 的最低配置会有所更改。

|          | 最低配置                                  | 推荐配置                                      |
| -------- | ------------------------------------- | ----------------------------------------- |
| **操作系统** | Ubuntu 16.04 或以上版本                    | Ubuntu 16.04 或以上版本                        |
| **CPU**  | Intel® Core™ i3 或 AMD Phenom™ X3 8650 | Intel® Core™ i5 或 AMD Phenom™ II X3 或更高配置 |
| **GPU**  | NVIDIA® GeForce® GTX 460              | NVIDIA® GeForce® GTX 660 或更高配置            |
| **内存**   | 4 GB                                  | 6 GB                                      |
| **存储**   | 30 GB 可用的磁盘空间                         |                                           |
| **网络**   | 宽带连接                                  |                                           |
| **分辨率**  | 1024 x 768 以上                         |                                           |

Below are the minimum system requirements for NNI on macOS. Due to potential programming changes, the minimum system requirements for NNI may change over time.

|          | 最低配置                                               | 推荐配置                     |
| -------- | -------------------------------------------------- | ------------------------ |
| **操作系统** | macOS 10.14.1 (最新版本)                               | macOS 10.14.1 (最新版本)     |
| **CPU**  | Intel® Core™ i5-760 或更高                            | Intel® Core™ i7-4770 或更高 |
| **GPU**  | NVIDIA® GeForce® GT 750M 或 AMD Radeon™ R9 M290 或更高 | AMD Radeon™ R9 M395X 或更高 |
| **内存**   | 4 GB                                               | 8 GB                     |
| **存储**   | 70GB 可用空间及 7200 RPM 硬盘                             | 70GB 可用空间 SSD 硬盘         |
| **网络**   | 宽带连接                                               |                          |
| **分辨率**  | 1024 x 768 以上                                      |                          |

Below are the minimum system requirements for NNI on Windows, Windows 10.1809 is well tested and recommend. Due to potential programming changes, the minimum system requirements for NNI may change over time.

|          | 最低配置                                  | 推荐配置                                      |
| -------- | ------------------------------------- | ----------------------------------------- |
| **操作系统** | Windows 10                            | Windows 10                                |
| **CPU**  | Intel® Core™ i3 或 AMD Phenom™ X3 8650 | Intel® Core™ i5 或 AMD Phenom™ II X3 或更高配置 |
| **GPU**  | NVIDIA® GeForce® GTX 460              | NVIDIA® GeForce® GTX 660 或更高配置            |
| **内存**   | 4 GB                                  | 6 GB                                      |
| **存储**   | 30 GB 可用的磁盘空间                         |                                           |
| **网络**   | 宽带连接                                  |                                           |
| **分辨率**  | 1024 x 768 以上                         |                                           |

## 更多

* [概述](Overview.md)
* [使用命令行工具 nnictl](NNICTLDOC.md)
* [使用 NNIBoard](WebUI.md)
* [定制搜索空间](SearchSpaceSpec.md)
* [配置 Experiment](ExperimentConfig.md)
* [如何在本机运行 Experiment (支持多 GPU 卡)？](LocalMode.md)
* [如何在多机上运行 Experiment？](RemoteMachineMode.md)
* [如何在 OpenPAI 上运行 Experiment？](PAIMode.md)
* [如何通过 Kubeflow 在 Kubernetes 上运行 Experiment？](KubeflowMode.md)
* [如何通过 FrameworkController 在 Kubernetes 上运行 Experiment？](FrameworkControllerMode.md)