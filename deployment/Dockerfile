FROM nni.build.base:cuda9.0-cudnn7-devel-ubuntu16.04

LABEL maintainer='Microsoft NNI Team<nni@microsoft.com>'

#
#Tensorflow 1.10.0
#
RUN pip3 --no-cache-dir install tensorflow-gpu==1.10.0

#
#Keras 2.1.6
#
RUN pip3 --no-cache-dir install Keras==2.1.6

WORKDIR /root