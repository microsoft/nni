# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

ARG NNI_RELEASE

LABEL maintainer='Microsoft NNI Team<nni@microsoft.com>'

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get -y update
RUN apt-get -y install \
    sudo \
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
    libcupti-dev
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

#
# generate python script
#
RUN ln -s python3 /usr/bin/python

#
# update pip
#
RUN python3 -m pip install --upgrade pip==20.2.4 setuptools==50.3.2

# numpy 1.19.5  scipy 1.5.4
RUN python3 -m pip --no-cache-dir install numpy==1.19.5 scipy==1.5.4

#
# TensorFlow
#
RUN python3 -m pip --no-cache-dir install tensorflow==2.3.1

#
# Keras
#
RUN python3 -m pip --no-cache-dir install Keras==2.4.3

#
# PyTorch
#
RUN python3 -m pip --no-cache-dir install torch==1.7.1 torchvision==0.8.2 pytorch-lightning==1.3.3

#
# sklearn 0.24.1
#
RUN python3 -m pip --no-cache-dir install scikit-learn==0.24.1

#
# pandas==0.23.4 lightgbm==2.2.2
#
RUN python3 -m pip --no-cache-dir install pandas==1.1 lightgbm==2.2.2

#
# Install NNI
#
COPY dist/nni-${NNI_RELEASE}-py3-none-manylinux1_x86_64.whl .
RUN python3 -m pip install nni-${NNI_RELEASE}-py3-none-manylinux1_x86_64.whl

# 
# Vision patch. Need del later
# 
COPY test/vso_tools/interim_patch.py .
RUN python3 interim_patch.py

#
# install aml package
#
RUN python3 -m pip --no-cache-dir install azureml
RUN python3 -m pip --no-cache-dir install azureml-sdk

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/root/.local/bin:/usr/bin:/bin:/sbin

WORKDIR /root
