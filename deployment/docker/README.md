Dockerfile
===
## 1.Description
This is the Dockerfile of nni project, including the most kinds of deeplearning frameworks and nni source code.  You can run your nni experiment in this docker container directly.
Dockerfile could build the customized docker image, users could build their customized docker image using this file.
This docker file includes the following libraries on `Ubuntu 16.04 LTS`:

```
CUDA 9.0, CuDNN 7.0
numpy 1.14.3,scipy 1.1.0
TensorFlow 1.5.0
Keras 2.1.6
PyTorch 0.4.1
scikit-learn 0.20.0
NNI v0.3
```

## 2.How to build and run
__Use the following command to build docker image__
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