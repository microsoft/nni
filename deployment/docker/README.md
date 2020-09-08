Dockerfile
===
## 1.Description
This is the Dockerfile of NNI project. It includes serveral popular deep learning frameworks and NNI. It is tested on `Ubuntu 16.04 LTS`:

```
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
```
You can take this Dockerfile as a reference for your own customized Dockerfile.

## 2.How to build and run
__Use the following command from `nni/deployment/docker` to build docker image__
```
    docker build -t nni/nni .
```
__Run the docker image__
* If does not use GPU in docker container, simply run the following command
```
    docker run -it nni/nni
```
Note that if you want to use tensorflow, please uninstall tensorflow-gpu and install tensorflow in this docker container. Or modify `Dockerfile` to install tensorflow (without gpu) and build docker image.

* If use GPU in docker container, make sure you have installed [NVIDIA Container Runtime](https://github.com/NVIDIA/nvidia-docker), then run the following command
```
    nvidia-docker run -it nni/nni
```
or
```
    docker run --runtime=nvidia -it nni/nni
```

## 3.Directly retrieve the docker image
Use the following command to retrieve the NNI docker image from Docker Hub
```
docker pull msranni/nni:latest
```
