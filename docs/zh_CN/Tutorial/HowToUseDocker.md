# **如何在 NNI 中使用 Docker**

## 概述

[Docker](https://www.docker.com/) 是一种工具, 可通过启动容器, 使用户能够更轻松地根据自己的操作系统部署和运行应用程序。 Docker 不是虚拟机，它不创建虚拟操作系统，但它允许不同的应用程序使用相同的操作系统内核，并通过容器隔离不同的应用程序。

用户可使用 Docker 来启动 NNI Experiment。 NNI 在 Docker Hub 上也提供了官方的 Docker 映像 [msranni/nni](https://hub.docker.com/r/msranni/nni)。

## 在本机使用docker

### 第一步：Docker 的安装

在开始使用 Docker 运行 NNI Experiment 前，首先需要在本机安装 Docker 运行程序。 [参考这里](https://docs.docker.com/install/linux/docker-ce/ubuntu/)。

### 第二步：启动 Docker 容器

如果已经在本地机器上安装了 Docker 程序，可以启动 Docker 容器来运行 NNI 示例。 因为 NNI 会在 Docker 容器里启动 Web 界面进程，并监听端口，因此需要指定一个在主机和 Docker 容器映射的端口，可在容器外访问 Docker 容器里的进程。 通过访问主机的 IP 和端口，就可以访问容器里的 Web 网页进程了。

例如，通过如下命令来启动 Docker 容器：

    docker run -i -t -p [hostPort]:[containerPort] [image]
    

-i: 使用交互模式启动 Docker。

-t: 为 Docker 分配一个输入终端。

-p: 端口映射，映射主机端口和容器端口。

更多命令信息，可[参考这里](https://docs.docker.com/v17.09/edge/engine/reference/run/)。

注意：

       NNI 目前仅支持本机模式下的 Ubuntu 和 macOS 系统，请使用正确的 Docker 映像类型。 如果想要在 Docker 容器里面使用 GPU，请使用 nvidia-docker。
    

### 第三步：在 Docker 容器里运行 NNI

如果直接使用 NNI 的官方镜像 `msranni/nni` 来启动 Experiment，可以直接使用 `nnictl` 命令。 NNI 官方镜像有最基础的 Python 环境和深度学习框架。

如果使用自己的 Docker 镜像，需要首先[安装 NNI](InstallationLinux.md)。

如果要使用 NNI 的官方示例，可以通过以下 git 命令来克隆 NNI：

    git clone https://github.com/Microsoft/nni.git
    

然后可以进入 `nni/examples/trials` 文件夹来启动 Experiment。

准备好 NNI 的环境后，可使用 `nnictl` 命令开始新的 Experiment。 [参考这里](QuickStart.md)。

## 在远程平台上运行 Docker

NNI 支持在[远程平台](../TrainingService/RemoteMachineMode.md)上启动 Experiment，并在远程机器里运行 Trial。 因为 Docker 可以运行独立的 Ubuntu 系统和 SSH 服务，因此 Docker 容器可以作为远程平台来运行 NNI。

### 第一步：设置 Docker 环境

You should install the Docker software on your remote machine first, please [refer to this](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

To make sure your Docker container can be connected by NNI experiments, you should build your own Docker image to set an SSH server or use images with an SSH configuration. If you want to use a Docker container as an SSH server, you should configure the SSH password login or private key login; please [refer to this](https://docs.docker.com/engine/examples/running_ssh_service/).

注意：

    NNI's official image msranni/nni does not support SSH servers for the time being; you should build your own Docker image with an SSH configuration or use other images as a remote server.
    

### Step 2: Start a Docker container on a remote machine

An SSH server needs a port; you need to expose Docker's SSH port to NNI as the connection port. For example, if you set your container's SSH port as **`A`**, you should map the container's port **`A`** to your remote host machine's other port **`B`**, NNI will connect port **`B`** as an SSH port, and your host machine will map the connection from port **`B`** to port **`A`** then NNI could connect to your Docker container.

For example, you could start your Docker container using the following commands:

    docker run -dit -p [hostPort]:[containerPort] [image]
    

The `containerPort` is the SSH port used in your Docker container and the `hostPort` is your host machine's port exposed to NNI. You can set your NNI's config file to connect to `hostPort` and the connection will be transmitted to your Docker container. For more information about Docker commands, please [refer to this](https://docs.docker.com/v17.09/edge/engine/reference/run/).

注意：

    If you use your own Docker image as a remote server, please make sure that this image has a basic python environment and an NNI SDK runtime environment. If you want to use a GPU in a Docker container, please use nvidia-docker.
    

### Step 3: Run NNI experiments

You can set your config file as a remote platform and set the `machineList` configuration to connect to your Docker SSH server; [refer to this](../TrainingService/RemoteMachineMode.md). Note that you should set the correct `port`, `username`, and `passWd` or `sshKeyPath` of your host machine.

`port:` The host machine's port, mapping to Docker's SSH port.

`username:` The username of the Docker container.

`passWd:` The password of the Docker container.

`sshKeyPath:` The path of the private key of the Docker container.

After the configuration of the config file, you could start an experiment, [refer to this](QuickStart.md).