# Dockerfile 

## 1. 说明

这是 NNI 项目的 Dockerfile 文件。 其中包含了 NNI 以及多个流行的深度学习框架。 在 `Ubuntu 16.04 LTS` 上进行过测试：

    CUDA 9.0
    CuDNN 7.0
    numpy 1.14.3
    scipy 1.1.0
    tensorflow-gpu 1.15.0
    keras 2.1.6
    torch 1.4.0
    scikit-learn 0.23.2
    pandas 0.23.4
    lightgbm 2.2.2
    nni
    

此 Dockerfile 可作为定制的参考。

## 2.如何生成和运行

**使用 `nni/deployment/docker` 的下列命令来生成 docker 映像。**

        docker build -t nni/nni .
    

**运行 docker 映像**

* 如果 docker 容器中没有 GPU，运行下面的命令

        docker run -it nni/nni
    

注意，如果要使用 tensorflow，需要先卸载 tensorflow-gpu，然后在 Docker 容器中安装 tensorflow。 或者修改 `Dockerfile` 来安装没有 GPU 的 tensorflow 版本，并重新生成 Docker 映像。

* 如果 docker 容器中有 GPU，确保安装了 [NVIDIA 容器运行包](https://github.com/NVIDIA/nvidia-docker)，然后运行下面的命令

        nvidia-docker run -it nni/nni
    

或者

        docker run --runtime=nvidia -it nni/nni
    

## 3.拉取 docker 映像

使用下列命令从 docker Hub 中拉取 NNI docker 映像。

    docker pull msranni/nni:latest