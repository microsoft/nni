**How to Use Docker in NNI**
===

## Overview

[Docker](https://www.docker.com/) is a tool to make it easier for users to deploy and run applications based on their own operating system by starting containers. Docker is not a virtual machine, it does not creating a virtual operating system, bug it allows different applications to use the same OS kernel as the basic system, and isolate different applications by container.

Users could start NNI experiments using docker, and NNI provides a offical docker image [msranni/nni](https://hub.docker.com/r/msranni/nni) in docker hub.

## Using docker in local machine

### step 1. Installation of docker
Before you start using docker to start NNI experiments, you should install a docker software in your local machine. [Refer](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

### step2. Start docker container
If you have installed the docker package in your local machine, you could start a docker container as an instance to run NNI examples. You should notice that because NNI needs a restful server port as a listen port to start web UI, you need to specify the port mapping between your host machine and docker container. By visting the host ip address and port, you could redirect to the web UI process started in docker container, and visit web UI content.

You could start a new docker container using following command:
```
docker run -i -t -p [hostPort]:[containerPort] [image]
``` 
`-i:` Start a docker in an interactive mode.

`-t:` Docker assign the container a input terminal.

`-p:` Port mapping, map host port to a container port.

For more information about docker command, please [refer](https://docs.docker.com/v17.09/edge/engine/reference/run/)

Note:
```
   NNI only support Ubuntu and MacOS system in local mode for the moment, please use correct docker image type.If you want to use gpu in docker container, please use nvidia-docker.
```
### step3. Run NNI in docker container

If you start a docker image using NNI's offical iamge `msranni/nni`, you could directly start NNI experiments by using `nnictl` command. Our offical image has NNI's running environment and basic python and deep learning frameworks environment. 

If you start your own docker image, you may need to install NNI package first, please [refer](https://github.com/Microsoft/nni/blob/master/docs/en_US/Installation.md).

If you want to run NNI's offical examples, you may need clone NNI repo in github using 
```
git clone https://github.com/Microsoft/nni.git
```
then you could enter `nni/examples/trials` to start an experiment.

After you prepare NNI's environment, you could start a new experiment using `nnictl` command, [refer](https://github.com/Microsoft/nni/blob/master/docs/en_US/QuickStart.md)

## Using docker in remote platform

NNI support starting experiments in [remoteTrainingService](https://github.com/Microsoft/nni/blob/master/docs/en_US/RemoteMachineMode.md), and run trial jobs in remote machines. As docker could start an independent Ubuntu system as SSH server, docker container could be the remote machine platform in NNI's remot mode.

### step 1. Setting docker environment

You should install a docker software in your remote machine first, please [refer](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

To make sure your docker container could be connected by NNI experiments, you should build your own docker image to set SSH server configuration or use images with SSH configuration. If you want to use docker container as SSH server, you should configure SSH password login or private key login, please [refer](https://docs.docker.com/engine/examples/running_ssh_service/).

Note:
```
NNI's offical image msranni/nni does not support SSH server for the time being, you should build your own docker image with SSH configuration or use other images as remote server.
```

### step2. Start docker container in remote machine

SSH server need a port, you need to expose docker's SSH port to NNI as the connection port. For example, if you set your container's SSH port as **`A`**, you should map container's **`A`** port to your remote host machine's another port **`B`**, NNI will connect port **`B`** as SSH port, and your host machine will map the connection from port **`B`** to port **`A`**, and then NNI could connect to your docker container.

For example, you could start your docker container using following commands:
```
docker run -dit -p [hostPort]:[containerPort] [image]
```
The `containerPort` is the SSH port used in your docker container, and the `hostPort` is your host machine's port exposed to NNI. You could set your NNI's config file to connect to `hostPort`, and the connection will be transmitted to your docker container.
For more information about docker command, please [refer](https://docs.docker.com/v17.09/edge/engine/reference/run/).

Note:
```
If you use your own docker image as remote server, please make sure that this image has basic python environment and NNI SDK runtime environment. If you want to use gpu in docker container, please use nvidia-docker.
```

### step3. Run NNI experiments

You could set your config file as remote platform, and setting the `machineList` configuration to connect your docker SSH server, [refer](https://github.com/Microsoft/nni/blob/master/docs/en_US/RemoteMachineMode.md). Note that you should set correct `port`,`username` and `passwd` or `sshKeyPath` of your host machine.

`port:` The host machine's port, mapping to docker's SSH port.
`username:` The username of docker container.
`passWd:` The password of docker container.
`sshKeyPath:` The path of private key of docker container.

After the configuration of config file, you could start an experiment, [refer](https://github.com/Microsoft/nni/blob/master/docs/en_US/QuickStart.md)