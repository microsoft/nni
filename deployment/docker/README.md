Dockerfile
===
## 1.Description
This is the Dockerfile of nni project. It includes serveral popular deep learning frameworks and NNI. It is tested on `Ubuntu 16.04 LTS`:

```
CUDA 9.0, CuDNN 7.0
numpy 1.14.3,scipy 1.1.0
TensorFlow 1.5.0
Keras 2.1.6
PyTorch 0.4.1
scikit-learn 0.20.0
NNI v0.3
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
