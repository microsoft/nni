# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04

LABEL maintainer='Microsoft NNI Team<nni@microsoft.com>'

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get -y update && \
    apt-get -y install sudo \
    apt-utils \
    git \
    curl \
    vim \
    unzip \
    wget \
    build-essential \
    cmake \
    libopenblas-dev \
    automake \
    openssh-client \
    openssh-server \
    lsof \
    python3.6 \
    python3-dev \
    python3-pip \
    python3-tk \
    libcupti-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#
# generate python script
#
RUN cp /usr/bin/python3 /usr/bin/python

#
# update pip
#
RUN python3 -m pip install --upgrade pip==20.0.2 setuptools==41.0.0

# numpy 1.14.3  scipy 1.1.0
RUN python3 -m pip --no-cache-dir install \
    numpy==1.14.3 scipy==1.1.0

#
# Tensorflow 1.15
#
RUN python3 -m pip --no-cache-dir install tensorflow-gpu==1.15.0

#
# Keras 2.1.6
#
RUN python3 -m pip --no-cache-dir install Keras==2.1.6

#
# PyTorch
#
RUN python3 -m pip --no-cache-dir install torch==1.4.0
RUN python3 -m pip install torchvision==0.5.0

#
# sklearn 0.23.2
#
RUN python3 -m pip --no-cache-dir install scikit-learn==0.23.2

#
# pandas==0.23.4 lightgbm==2.2.2
#
RUN python3 -m pip --no-cache-dir install pandas==0.23.4 lightgbm==2.2.2

#
# Install NNI
#
RUN python3 -m pip --no-cache-dir install nni

#
# install aml package
#
RUN python3 -m pip --no-cache-dir install azureml
RUN python3 -m pip --no-cache-dir install azureml-sdk


ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/root/.local/bin:/usr/bin:/bin:/sbin

WORKDIR /root
