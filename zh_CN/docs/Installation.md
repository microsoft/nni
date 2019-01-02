# **安装 NNI**

当前仅支持 Linux 和 Mac。

## **安装**

* **依赖项**
    
    python >= 3.5 git wget
    
    需要正确安装 Python 的 pip。 可以用 "python3 -m pip -v" 来检查 pip 的版本。

* **通过 pip 命令安装 NNI**
    
    python3 -m pip install --user --upgrade nni

* **通过源代码安装 NNI**
    
    git clone -b v0.4.1 https://github.com/Microsoft/nni.git cd nni source install.sh

* **在 docker 映像中安装 NNI**
    
    也可将 NNI 安装到 docker 映像中。 参考[这里](../deployment/docker/README.md)来生成 NNI 的 docker 映像。 也可通过此命令从 Docker Hub 中直接拉取 NNI 的映像 `docker pull msranni/nni:latest`。

## **系统需求**

以下是 NNI 在 Linux 下的最小需求。 由于程序变更，NNI 的最小需求会有所更改。

|                      | 最小需求                                   | Recommended Specifications                     |
| -------------------- | -------------------------------------- | ---------------------------------------------- |
| **Operating System** | Ubuntu 16.04 or above                  | Ubuntu 16.04 or above                          |
| **CPU**              | Intel® Core™ i3 or AMD Phenom™ X3 8650 | Intel® Core™ i5 or AMD Phenom™ II X3 or better |
| **GPU**              | NVIDIA® GeForce® GTX 460               | NVIDIA® GeForce® GTX 660 or better             |
| **Memory**           | 4 GB RAM                               | 6 GB RAM                                       |
| **Storage**          | 30 GB available hare drive space       |                                                |
| **Internet**         | Boardband internet connection          |                                                |
| **Resolution**       | 1024 x 768 minimum display resolution  |                                                |

Below are the minimum system requirements for NNI on macOS. Due to potential programming changes, the minimum system requirements for NNI may change over time.

|                      | Minimum Requirements                                      | Recommended Specifications     |
| -------------------- | --------------------------------------------------------- | ------------------------------ |
| **Operating System** | macOS 10.14.1 (latest version)                            | macOS 10.14.1 (latest version) |
| **CPU**              | Intel® Core™ i5-760 or better                             | Intel® Core™ i7-4770 or better |
| **GPU**              | NVIDIA® GeForce® GT 750M or AMD Radeon™ R9 M290 or better | AMD Radeon™ R9 M395X or better |
| **Memory**           | 4 GB RAM                                                  | 8 GB RAM                       |
| **Storage**          | 70GB available space 7200 RPM HDD                         | 70GB available space SSD       |
| **Internet**         | Boardband internet connection                             |                                |
| **Resolution**       | 1024 x 768 minimum display resolution                     |                                |

## Further reading

* [Overview](Overview.md)
* [Use command line tool nnictl](NNICTLDOC.md)
* [Use NNIBoard](WebUI.md)
* [Define search space](SearchSpaceSpec.md)
* [Config an experiment](ExperimentConfig.md)
* [How to run an experiment on local (with multiple GPUs)?](tutorial_1_CR_exp_local_api.md)
* [How to run an experiment on multiple machines?](tutorial_2_RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](PAIMode.md)