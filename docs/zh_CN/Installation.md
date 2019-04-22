# 安装 NNI

Currently we support installation on Linux, Mac and Windows.

## **Installation on Linux & Mac**

* **通过 pip 命令安装 NNI**
    
    先决条件：`python >= 3.5`
    
    ```bash
    python3 -m pip install --upgrade nni
    ```

* **通过源代码安装 NNI**
    
    Prerequisite: `python >=3.5`, `git`, `wget`
    
    ```bash
    git clone -b v0.6 https://github.com/Microsoft/nni.git
    cd nni
    ./install.sh
    ```

* **在 docker 映像中安装 NNI**
    
    也可将 NNI 安装到 docker 映像中。 参考[这里](../deployment/docker/README.md)来生成 NNI 的 Docker 映像。 也可通过此命令从 Docker Hub 中直接拉取 NNI 的映像 `docker pull msranni/nni:latest`。

## **Installation on Windows**

* **Install NNI through pip**
    
    Prerequisite: `python >= 3.5`
    
    ```bash
    python -m pip install --upgrade nni
    ```

* **Install NNI through source code**
    
    Prerequisite: `python >=3.5`, `git`, `powershell`  
    When you use powershell to run script for the first time, you need run powershell as Administrator with this command:
    
    ```bash
    Set-ExecutionPolicy -ExecutionPolicy Unrestricted
    ```
    
    Then you can install nni as administrator or current user as follows:
    
    ```bash
    git clone https://github.com/Microsoft/nni.git
    cd nni
    powershell ./install.ps1
    ```

## **System requirements**

以下是 NNI 在 Linux 下的最低配置。 由于程序变更，NNI 的最低配置会有所更改。

|          | 最低配置                                  | 推荐配置                                      |
| -------- | ------------------------------------- | ----------------------------------------- |
| **操作系统** | Ubuntu 16.04 或以上版本                    | Ubuntu 16.04 或以上版本                        |
| **CPU**  | Intel® Core™ i3 或 AMD Phenom™ X3 8650 | Intel® Core™ i5 或 AMD Phenom™ II X3 或更高配置 |
| **GPU**  | NVIDIA® GeForce® GTX 460              | NVIDIA® GeForce® GTX 660 或更高配置            |
| **内存**   | 4 GB                                  | 6 GB                                      |
| **存储**   | 30 GB 可用的磁盘空间                         |                                           |
| **网络**   | 宽带连接                                  |                                           |
| **分辨率**  | 1024 x 768 以上                         |                                           |

以下是 NNI 在 MacOS 下的最低配置。 由于程序变更，NNI 的最低配置会有所更改。

|          | 最低配置                                               | 推荐配置                     |
| -------- | -------------------------------------------------- | ------------------------ |
| **操作系统** | macOS 10.14.1 (最新版本)                               | macOS 10.14.1 (最新版本)     |
| **CPU**  | Intel® Core™ i5-760 或更高                            | Intel® Core™ i7-4770 或更高 |
| **GPU**  | NVIDIA® GeForce® GT 750M 或 AMD Radeon™ R9 M290 或更高 | AMD Radeon™ R9 M395X 或更高 |
| **内存**   | 4 GB                                               | 8 GB                     |
| **存储**   | 70GB 可用空间及 7200 RPM 硬盘                             | 70GB 可用空间 SSD 硬盘         |
| **网络**   | 宽带连接                                               |                          |
| **分辨率**  | 1024 x 768 以上                                      |                          |

Below are the minimum system requirements for NNI on Windows. Due to potential programming changes, the minimum system requirements for NNI may change over time.

|                      | Minimum Requirements                   | Recommended Specifications                     |
| -------------------- | -------------------------------------- | ---------------------------------------------- |
| **Operating System** | Windows 10                             | Windows 10                                     |
| **CPU**              | Intel® Core™ i3 or AMD Phenom™ X3 8650 | Intel® Core™ i5 or AMD Phenom™ II X3 or better |
| **GPU**              | NVIDIA® GeForce® GTX 460               | NVIDIA® GeForce® GTX 660 or better             |
| **Memory**           | 4 GB RAM                               | 6 GB RAM                                       |
| **Storage**          | 30 GB available hare drive space       |                                                |
| **Internet**         | Boardband internet connection          |                                                |
| **Resolution**       | 1024 x 768 minimum display resolution  |                                                |

## Further reading

* [Overview](Overview.md)
* [Use command line tool nnictl](NNICTLDOC.md)
* [Use NNIBoard](WebUI.md)
* [Define search space](SearchSpaceSpec.md)
* [Config an experiment](ExperimentConfig.md)
* [How to run an experiment on local (with multiple GPUs)?](LocalMode.md)
* [How to run an experiment on multiple machines?](RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](PAIMode.md)
* [How to run an experiment on Kubernetes through Kubeflow?](KubeflowMode.md)
* [How to run an experiment on Kubernetes through FrameworkController?](FrameworkControllerMode.md)