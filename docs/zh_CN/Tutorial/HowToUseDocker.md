# **如何在 NNI 中使用 Docker**

## 概述

[Docker](https://www.docker.com/) 是一种工具, 可通过启动容器, 使用户能够更轻松地根据自己的操作系统部署和运行应用程序。 Docker 不是虚拟机, 它不创建虚拟操作系统, 但是它允许不同的应用程序使用相同的操作系统内核, 并通过容器隔离不同的应用程序。

用户可以使用docker进行 NNI 实验, NNI 在docker hub上提供了一个官方的镜像 [msranni/nni](https://hub.docker.com/r/msranni/nni)。

## 在本机使用docker

### 第一步：docker的安装

在你开始使用docker进行NNI实验之前，你首先需要在本地机器上安装docker运行程序。 [参考](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

### 第二步：启动docker容器

如果你已经在本地机器上安装了docker程序，你可以启动docker容器来运行NNI实验了。 因为NNI会在docker容器里面启动web UI进程，并且监听一个端口，因此你需要指定一个在主机和docker容器里面的端口映射，这个映射可以让你在容器外面访问docker容器里面的进程。 通过访问主机的ip和端口，你就可以访问容器里面的Web网页进程了。

例如，你可以通过如下命令来启动docker容器：

    docker run -i -t -p [hostPort]:[containerPort] [image]
    

-i: 使用交互模式启动docker

-t: Docker分配一个输入终端。

-p: 端口映射，映射主机端口和容器端口。

可以参考[这里](https://docs.docker.com/v17.09/edge/engine/reference/run/)，获取更多的命令参考。

注意：

       NNI只支持Ubuntu和macOS操作系统，请指定正确的docker镜像。如果你希望在docker里面使用gpu，请使用nvidia-docker。
    

### 步骤3：在docker容器里面运行NNI

如果你直接使用NNI的官方镜像`msranni/nni`来启动实验，你可以直接使用`nnictl`命令。 NNI的官方镜像有最基础的python环境和深度学习框架。

如果你使用你自己的docker镜像，你首先需要安装NNI环境。[参考](Installation.md)

如果你想要使用NNI的官方例子，你可以通过以下git命令来克隆NNI：

    git clone https://github.com/Microsoft/nni.git
    

然后可以进入`nni/examples/trials`文件夹来启动实验。

等你准备完NNI环境，你可以通过`nnictl`命令来启动实验，[参考](QuickStart.md).

## 在远程平台上运行docker

NNI支持在[远程平台](../TrainingService/RemoteMachineMode.md)上启动实验，在远程机器里运行任务。 因为docker可以运行独立的Ubuntu系统和SSH服务，因此docker容器可以作为远程平台来运行NNI.

### 步骤1：设置docker环境

你首先应该在远程机器上安装docker工具，[参考](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

为了保证docker容器可以被NNI实验连接上，你应该在你自己的docker容器里面安装SSH服务，并做SSH相关配置。 如果你想在docker容器里面使用SSH服务，你应该配置SSH密码登录或者私钥登录，[参考](https://docs.docker.com/engine/examples/running_ssh_service/)。

注意：

    NNI的官方镜像msranni/nni暂时不支持SSH服务，你应该构建自己的带有SSH服务的镜像，或者使用其他的带有SSH服务的镜像。
    

### 第二步：在远程机器上启动docker容器

SSH容器需要一个端口，你需要把docker的SSH服务端口暴露给NNI作为连接端口。 例如，如果你设置容器的端口**`A`**作为SSH端口，你应该把端口**`A`**映射到主机的端口**`B`**，NNI会连接端口**`B`**作为SSH服务端口，你的主机会把连接到端口**`B`**的连接映射到端口**`A`**，NNI就可以连接到你的容器中了。

例如，你可以通过如下命令来启动docker容器：

    docker run -dit -p [hostPort]:[containerPort] [image]
    

`containerPort`是在docker容器中指定的端口，`hostPort`是主机的端口。 你可以设置你的NNI配置，连接到`hostPort`，这个连接会被转移到你的docker容器中。 更多的命定信息，可以[参考](https://docs.docker.com/v17.09/edge/engine/reference/run/).

注意：

    如果你使用你自己构建的docker容器，请保证这个容器中有基础的python运行时环境和NNI SDK环境。 如果你想要在docker容器里面使用gpu，请使用nvidia-docker。
    

### 步骤三：运行NNI实验

你可以在你的配置文件中，设置训练平台为远程平台，然后设置`machineList`配置。[参考](../TrainingService/RemoteMachineMode.md)。 注意你应该设置正确的`port`，`username`, `passwd`或者`sshKeyPath`。

`port`: 主机的端口，映射到docker的SSH端口中。

`username`: docker容器的用户名。

`passWd: ` docker容器的密码。

`sshKeyPath:` docker容器私钥的存储路径。

设置完配置文件，你就可以启动实验了，[参考](QuickStart.md)。