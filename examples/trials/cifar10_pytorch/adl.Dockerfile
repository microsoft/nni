# Dockerfile for building AdaptDL-enabled CIFAR10 image
# Set docker build context to current folder

FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN pip install nni adaptdl tensorboard

COPY ./ /cifar10
