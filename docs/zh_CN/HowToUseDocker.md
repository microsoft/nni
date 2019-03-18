# **如何在 NNI 中使用 Docker**

## 概述

[Docker](https://www.docker.com/) 是一种工具, 可通过启动容器, 使用户能够更轻松地根据自己的操作系统部署和运行应用程序。 Docker 不是虚拟机, 它不创建虚拟操作系统, 但是它允许不同的应用程序使用相同的操作系统内核, 并通过容器隔离不同的应用程序。

用户可以使用docker进行 NNI 实验, NNI 在docker hub上提供了一个官方的镜像 [msranni/nni](https://hub.docker.com/r/msranni/nni)。

## 在本地模式使用docker

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

If you start your own docker image, you may need to install NNI package first, please [refer](Installation.md).

If you want to run NNI's offical examples, you may need to clone NNI repo in github using

    git clone https://github.com/Microsoft/nni.git
    

then you could enter `nni/examples/trials` to start an experiment.

After you prepare NNI's environment, you could start a new experiment using `nnictl` command, [refer](QuickStart.md)

## Using docker in remote platform

NNI support starting experiments in [remoteTrainingService](RemoteMachineMode.md), and run trial jobs in remote machines. As docker could start an independent Ubuntu system as SSH server, docker container could be used as the remote machine in NNI's remot mode.

### Step 1: Setting docker environment

You should install a docker software in your remote machine first, please [refer](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

To make sure your docker container could be connected by NNI experiments, you should build your own docker image to set SSH server or use images with SSH configuration. If you want to use docker container as SSH server, you should configure SSH password login or private key login, please [refer](https://docs.docker.com/engine/examples/running_ssh_service/).

Note:

    NNI's offical image msranni/nni does not support SSH server for the time being, you should build your own docker image with SSH configuration or use other images as remote server.
    

### Step2: Start docker container in remote machine

SSH server need a port, you need to expose docker's SSH port to NNI as the connection port. For example, if you set your container's SSH port as **`A`**, you should map container's port **`A`** to your remote host machine's another port **`B`**, NNI will connect port **`B`** as SSH port, and your host machine will map the connection from port **`B`** to port **`A`**, then NNI could connect to your docker container.

For example, you could start your docker container using following commands:

    docker run -dit -p [hostPort]:[containerPort] [image]
    

The `containerPort` is the SSH port used in your docker container, and the `hostPort` is your host machine's port exposed to NNI. You could set your NNI's config file to connect to `hostPort`, and the connection will be transmitted to your docker container. For more information about docker command, please [refer](https://docs.docker.com/v17.09/edge/engine/reference/run/).

Note:

    If you use your own docker image as remote server, please make sure that this image has basic python environment and NNI SDK runtime environment. If you want to use gpu in docker container, please use nvidia-docker.
    

### Step3: Run NNI experiments

You could set your config file as remote platform, and setting the `machineList` configuration to connect your docker SSH server, [refer](RemoteMachineMode.md). Note that you should set correct `port`,`username` and `passwd` or `sshKeyPath` of your host machine.

`port:` The host machine's port, mapping to docker's SSH port.

`username:` The username of docker container.

`passWd:` The password of docker container.

`sshKeyPath:` The path of private key of docker container.

After the configuration of config file, you could start an experiment, [refer](QuickStart.md)