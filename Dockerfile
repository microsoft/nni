# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04

LABEL maintainer='Microsoft NNI Team<nni@microsoft.com>'

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update
RUN apt-get -y install \
    build-essential \
    git \
    openssh-server \
    python3-pip \
    python3.8 \
    python3.8-dev \
    wget

# candidate packages:
#   apt-utils
#   automake
#   cmake
#   curl
#   sudo
#   libcupti-dev
#   libopenblas-dev
#   lsof
#   python3-tk
#   unzip
#   vim

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.8 /usr/bin/python
RUN ln -sf python3.8 /usr/bin/python3

RUN python -m pip --no-cache-dir install --upgrade pip==20.2.4 setuptools==50.3.2

RUN python -m pip --no-cache-dir install \
    azureml==0.2.7 \
    azureml-sdk==1.16.0 \
    lightgbm==3.0.0 \
    numpy==1.19.2 \
    pandas==1.1.3 \
    scikit-learn==0.23.2 \
    scipy==1.5.3 \
    tensorflow==2.3.1 \
    torch==1.6.0 \
    torchvision==0.7.0

RUN python3 -m pip --no-cache-dir install nni

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/root/.local/bin:/usr/bin:/sbin:/bin

WORKDIR /root
