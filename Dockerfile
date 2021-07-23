# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ARG NNI_RELEASE

LABEL maintainer='Microsoft NNI Team<nni@microsoft.com>'

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get -y update
RUN apt-get -y install \
    apt-utils \
    automake \
    build-essential \
    cmake \
    curl \
    git \
    libcupti-dev \
    libopenblas-dev \
    lsof \
    openssh-client \
    openssh-server \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    sudo \
    unzip \
    vim \
    wget
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN ln -s python3 /usr/bin/python

#
# Install libraries
#
RUN python3 -m pip install --upgrade pip==21.1.3 setuptools==57.4.0 wheel==0.36.2
RUN python3 -m pip --no-cache-dir install numpy==1.21.1 scipy==1.7.0
RUN python3 -m pip --no-cache-dir install tensorflow==2.4.2
RUN python3 -m pip --no-cache-dir install torch==1.7.1 torchvision==0.8.2 pytorch-lightning==1.3.8
RUN python3 -m pip --no-cache-dir install scikit-learn==0.24.2
RUN python3 -m pip --no-cache-dir install pandas==1.3.0 lightgbm==3.2.1
RUN python3 -m pip --no-cache-dir install azureml azureml-sdk

#
# Install NNI
#
COPY dist/nni-${NNI_RELEASE}-py3-none-manylinux1_x86_64.whl .
RUN python3 -m pip install nni-${NNI_RELEASE}-py3-none-manylinux1_x86_64.whl

#
# Vision patch. Need del later
#
COPY interim_vision_patch.py .
RUN python3 interim_vision_patch.py

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/root/.local/bin:/usr/bin:/bin:/sbin

WORKDIR /root
