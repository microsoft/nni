Dockerfile
===
## 1.Description  
This is the Dockerfile of nni project, including the most kinds of deeplearning frameworks and nni source code.  You can run your nni experiment in this docker container directly.
Dockerfile.build.base could build the base Docker image, users can get a docker image with Ubuntu and NNI environment after building this file. 
Dockerfile could build the customized docker image, users could build their customized docker image using this file.
## 2.Including Libraries  

```
Ubuntu 16.04 LTS
CUDA 9.0, CuDNN 7.0
numpy 1.14.3,scipy 1.1.0
TensorFlow 1.5.0
Keras 2.1.6
NNI v0.1
```

## 3 How to run  
    
    docker build -f Dockerfile.build.base -t nni.build.base:cuda9.0-cudnn7-devel-ubuntu16.04 .
    docker build -t nni/nni .
    nvidia-docker run -it nni/nni